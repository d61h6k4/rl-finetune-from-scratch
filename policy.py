import copy
import random
from collections import Counter

import torch
from torch import nn

from rpn import (
    COMP_LEN,
    EOS,
    PAD,
    PROMPT_LEN,
    VOCAB,
    decode_completion,
    encode_task,
    make_demos,
    task_split,
    verify,
)


class Policy(nn.Module):
    def __init__(
        self,
        vocab=len(VOCAB),  # noqa: B008 (VOCAB is a constant)
        d=64,
        heads=4,
        layers=2,
        max_len=16,
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True
            )
            for _ in range(layers)
        )
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, vocab, bias=False)
        self.max_len = max_len

    def forward(self, x):  # ty: ignore[invalid-mutable-override]  # nn.Module declares forward as an attribute
        T = x.shape[1]
        h = self.emb(x) + self.pos(torch.arange(T, device=x.device))
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
        for layer in self.layers:
            h = layer(h, src_mask=mask)
        return self.out(self.norm(h))


def sample_completions(policy, prompts, G, generator, temperature=1.0):
    """Sample G completions per prompt; row i * G + g is sample g of prompt i."""
    with torch.no_grad():
        seqs = prompts.repeat_interleave(G, 0)
        done = torch.zeros(seqs.shape[0], dtype=torch.bool, device=seqs.device)
        mask = []
        for _ in range(COMP_LEN):
            logits = policy(seqs)[:, -1] / temperature
            probs = torch.softmax(logits, -1)
            tok = torch.multinomial(probs, 1, generator=generator)
            tok = torch.where(done[:, None], PAD, tok)
            mask.append((~done).float())
            done = done | (tok[:, 0] == EOS)
            seqs = torch.cat([seqs, tok], 1)
        return seqs, torch.stack(mask, 1)


def completion_logprobs(policy, seqs):
    """(N, COMP_LEN) log pi(y_t | x, y_<t) for each generated token; keeps gradients."""
    logits = policy(seqs[:, :-1])[:, PROMPT_LEN - 1 :]
    logp = torch.log_softmax(logits, -1)
    return logp.gather(-1, seqs[:, PROMPT_LEN:, None]).squeeze(-1)


def sft_loss(policy, batch):
    """Mean negative log-prob of the demonstrated completion tokens."""
    seqs = torch.tensor([prompt + comp for prompt, comp in batch])
    return -completion_logprobs(policy, seqs).mean()


def pretrain(policy, demos, steps, batch_size=64, lr=1e-3, seed=0):
    """AdamW on sft_loss over random demo batches; returns the last step's loss."""
    rng = random.Random(seed)
    opt = torch.optim.AdamW(policy.parameters(), lr=lr)
    loss = float("nan")
    for _ in range(steps):
        batch = rng.sample(demos, min(batch_size, len(demos)))
        step_loss = sft_loss(policy, batch)
        opt.zero_grad()
        step_loss.backward()
        opt.step()
        loss = step_loss.item()
    return loss


def pass_at_k(policy, tasks, k, generator, strict=True):
    """(pass@1, pass@k): sample-level and task-level success rates."""
    prompts = torch.tensor([encode_task(nums, target) for nums, target in tasks])
    seqs, _ = sample_completions(policy, prompts, k, generator)
    ok = [
        verify(nums, target, decode_completion(seqs[i * k + g, PROMPT_LEN:]), strict)[0]
        for i, (nums, target) in enumerate(tasks)
        for g in range(k)
    ]
    per_task = [any(ok[i * k : (i + 1) * k]) for i in range(len(tasks))]
    return sum(ok) / len(ok), sum(per_task) / len(per_task)


def group_rewards(tasks, seqs, G, strict=True):
    """(rewards (B, G) of 1.0/0.0, flat list of B * G verifier reasons in row order)."""
    verdicts = [
        verify(nums, target, decode_completion(seqs[i * G + g, PROMPT_LEN:]), strict)
        for i, (nums, target) in enumerate(tasks)
        for g in range(G)
    ]
    rewards = torch.tensor([float(ok) for ok, _ in verdicts]).view(len(tasks), G)
    return rewards, [reason for _, reason in verdicts]


def group_advantages(rewards, normalize_std=True, eps=1e-4):
    """Per-group centred (and optionally std-normalised) rewards, flattened to (B * G,)."""
    adv = rewards - rewards.mean(1, keepdim=True)
    if normalize_std:
        adv = adv / (rewards.std(1, keepdim=True) + eps)
    constant = (
        rewards.max(1, keepdim=True).values == rewards.min(1, keepdim=True).values
    )
    return torch.where(constant, torch.zeros_like(adv), adv).reshape(-1)


def kl_k3(logp, ref_logp):
    """Elementwise k3 estimator of KL(pi_theta || pi_ref); gradients flow through logp only."""
    d = ref_logp.detach() - logp
    return torch.exp(d) - d - 1


def grpo_loss(logp, old_logp, ref_logp, advantages, mask, clip_eps=0.2, beta=0.02):
    """Clipped policy-gradient + beta * k3 KL, averaged over generated tokens."""
    ratio = torch.exp(logp - old_logp)
    A = advantages[:, None]
    pg = -torch.min(ratio * A, ratio.clamp(1 - clip_eps, 1 + clip_eps) * A)
    kl = kl_k3(logp, ref_logp)
    n = mask.sum()
    loss = (mask * (pg + beta * kl)).sum() / n
    outside = (ratio < 1 - clip_eps) | (ratio > 1 + clip_eps)
    stats = {
        "pg": ((mask * pg).sum() / n).item(),
        "kl": ((mask * kl).sum() / n).item(),
        "clip_frac": ((mask * outside).sum() / n).item(),
    }
    return loss, stats


def rollout(policy, ref, tasks, G, generator, strict=True, normalize_std=True):
    """Sample G completions per task and precompute everything the loss needs."""
    prompts = torch.tensor([encode_task(nums, target) for nums, target in tasks])
    seqs, mask = sample_completions(policy, prompts, G, generator)
    rewards, reasons = group_rewards(tasks, seqs, G, strict)
    with torch.no_grad():
        old_logp = completion_logprobs(policy, seqs)
        ref_logp = completion_logprobs(ref, seqs)
    return {
        "seqs": seqs,
        "mask": mask,
        "rewards": rewards,
        "advantages": group_advantages(rewards, normalize_std),
        "old_logp": old_logp,
        "ref_logp": ref_logp,
        "reasons": reasons,
    }


def train_grpo(policy, ref, tasks, steps, cfg, seed=0, strict=True):
    """GRPO loop: rollout, then cfg["epochs"] clipped updates per rollout. ref is frozen."""
    rng = random.Random(seed)
    generator = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(policy.parameters(), lr=cfg["lr"])
    records = []
    for step in range(steps):
        batch = rng.sample(tasks, cfg["batch"])
        out = rollout(
            policy, ref, batch, cfg["group"], generator, strict, cfg["normalize_std"]
        )
        for _ in range(cfg["epochs"]):
            logp = completion_logprobs(policy, out["seqs"])
            loss, stats = grpo_loss(
                logp,
                out["old_logp"],
                out["ref_logp"],
                out["advantages"],
                out["mask"],
                cfg["clip_eps"],
                cfg["beta"],
            )
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
        records.append(
            {
                "step": step,
                "reward": out["rewards"].mean().item(),
                "loss": loss.item(),
                **stats,
            }
        )
    return records


def completion_entropy(policy, seqs, mask):
    """Masked mean entropy (nats) of the next-token distribution at generated positions."""
    with torch.no_grad():
        logp = torch.log_softmax(policy(seqs[:, :-1])[:, PROMPT_LEN - 1 :], -1)
        H = -(logp.exp() * logp).sum(-1)
        return ((mask * H).sum() / mask.sum()).item()


@torch.no_grad()
def evaluate(policy, tasks, k, generator):
    """Strict and fail-open metrics, entropy and error breakdown from one batch of samples."""
    prompts = torch.tensor([encode_task(nums, target) for nums, target in tasks])
    seqs, mask = sample_completions(policy, prompts, k, generator)
    strict, reasons = group_rewards(tasks, seqs, k, strict=True)
    lenient, _ = group_rewards(tasks, seqs, k, strict=False)
    by_digits = {}
    for d in (1, 2, 3):
        rows = [i for i, (_, target) in enumerate(tasks) if len(str(target)) == d]
        if rows:
            by_digits[d] = strict[rows].mean().item()
    return {
        "pass1": strict.mean().item(),
        "passk": strict.max(1).values.mean().item(),
        "lenient_pass1": lenient.mean().item(),
        "gaming": ((lenient == 1) & (strict == 0)).float().mean().item(),
        "entropy": completion_entropy(policy, seqs, mask),
        "by_digits": by_digits,
        "reasons": {r: round(c / len(reasons), 3) for r, c in Counter(reasons).items()},
        "example": decode_completion(seqs[0, PROMPT_LEN:]),
    }


def build_base(seed=0, sft_steps=600, n_draws=4000, n_held=200):
    """Reproducible base: task split, SFT warm-start policy and its final SFT loss."""
    rng = random.Random(seed)
    train, held = task_split(rng, n_draws, n_held)
    demos = make_demos(rng, train)
    torch.manual_seed(seed)
    policy = Policy()
    sft_loss_value = pretrain(policy, demos, sft_steps, seed=seed)
    return {"policy": policy, "train": train, "held": held, "sft_loss": sft_loss_value}


def run_grpo(base, steps, cfg, seed=0, strict=True):
    """GRPO on a copy of base["policy"]; base is left untouched."""
    policy = copy.deepcopy(base["policy"])
    ref = copy.deepcopy(base["policy"])
    history = train_grpo(policy, ref, base["train"], steps, cfg, seed, strict)
    return {"policy": policy, "ref": ref, "history": history}


def rlvr_experiment(
    seed=0, sft_steps=600, rl_steps=300, hack_steps=50, cfg=None, k=8, window=50
):
    """Base policy vs strict-verifier GRPO vs fail-open-verifier GRPO, evaluated on held-out tasks."""
    if cfg is None:
        cfg = {
            "batch": 16,
            "group": 8,
            "lr": 3e-4,
            "beta": 0.02,
            "clip_eps": 0.2,
            "normalize_std": True,
            "epochs": 1,
        }

    def held_eval(policy):
        return evaluate(policy, base["held"], k, torch.Generator().manual_seed(seed))

    base = build_base(seed, sft_steps)
    base_eval = held_eval(base["policy"])
    strict = run_grpo(base, rl_steps, cfg, seed, strict=True)
    hack = run_grpo(base, hack_steps, cfg, seed, strict=False)
    history = strict["history"]
    curve = []
    for start in range(0, len(history), window):
        chunk = history[start : start + window]
        curve.append(
            {
                "step": chunk[-1]["step"] + 1,
                "reward": sum(r["reward"] for r in chunk) / len(chunk),
                "kl": sum(r["kl"] for r in chunk) / len(chunk),
            }
        )
    return {
        "base": {"sft_loss": base["sft_loss"], "eval": base_eval},
        "strict": {
            "history": history,
            "eval": held_eval(strict["policy"]),
            "curve": curve,
        },
        "hack": {"history": hack["history"], "eval": held_eval(hack["policy"])},
        "cfg": cfg,
        "n_train": len(base["train"]),
        "n_held": len(base["held"]),
    }
