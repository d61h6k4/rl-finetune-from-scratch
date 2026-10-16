import copy
import math
import random
import unittest
from unittest import mock

import torch

from policy import (
    Policy,
    build_base,
    completion_entropy,
    completion_logprobs,
    evaluate,
    group_advantages,
    group_rewards,
    grpo_loss,
    kl_k3,
    pass_at_k,
    pretrain,
    rlvr_experiment,
    rollout,
    run_grpo,
    sample_completions,
    sft_loss,
    train_grpo,
)
from rpn import (
    COMP_LEN,
    EOS,
    PAD,
    PROMPT_LEN,
    TOK,
    VOCAB,
    encode_task,
    make_demos,
    task_split,
)


class TestPolicy(unittest.TestCase):
    def test_shape(self):
        model = Policy()
        x = torch.randint(0, len(VOCAB), (3, 14))
        self.assertEqual(model(x).shape, (3, 14, len(VOCAB)))

    def test_causal(self):
        # changing a later token must not change logits at earlier positions
        torch.manual_seed(0)
        model = Policy().eval()
        a = torch.randint(0, len(VOCAB), (2, 10))
        b = a.clone()
        b[:, 7:] = (b[:, 7:] + 1) % len(VOCAB)
        with torch.no_grad():
            la, lb = model(a), model(b)
        self.assertTrue(torch.allclose(la[:, :7], lb[:, :7], atol=1e-6))
        self.assertFalse(torch.allclose(la[:, 7:], lb[:, 7:], atol=1e-6))

    def test_structure(self):
        model = Policy()
        self.assertEqual(len(model.layers), 2)
        self.assertEqual(model.max_len, 16)
        self.assertIsNone(model.out.bias)


class Scripted(torch.nn.Module):
    """Position t >= PROMPT_LEN - 1 puts all probability on script[t - PROMPT_LEN + 1]."""

    def __init__(self, script):
        super().__init__()
        self.script = script

    def forward(self, x):  # ty: ignore[invalid-mutable-override]
        logits = torch.full((*x.shape, len(VOCAB)), -1e9)
        for t in range(PROMPT_LEN - 1, x.shape[1]):
            logits[:, t, self.script[t - PROMPT_LEN + 1]] = 0.0
        return logits


class TestSampleCompletions(unittest.TestCase):
    def prompts(self, B=2):
        return torch.randint(0, len(VOCAB), (B, PROMPT_LEN))

    def test_shapes_and_prompt_rows(self):
        p = self.prompts(3)
        seqs, mask = sample_completions(
            Policy(), p, 4, torch.Generator().manual_seed(0)
        )
        self.assertEqual(seqs.shape, (12, PROMPT_LEN + COMP_LEN))
        self.assertEqual(mask.shape, (12, COMP_LEN))
        self.assertEqual(mask.dtype, torch.float32)
        for i in range(3):
            for g in range(4):
                self.assertTrue(torch.equal(seqs[i * 4 + g, :PROMPT_LEN], p[i]))

    def test_eos_mask_and_pad(self):
        script = [TOK["3"], TOK["7"], EOS, TOK["9"], TOK["9"], TOK["9"]]
        seqs, mask = sample_completions(
            Scripted(script), self.prompts(), 2, torch.Generator().manual_seed(0)
        )
        comp = seqs[:, PROMPT_LEN:]
        expected = [TOK["3"], TOK["7"], EOS, PAD, PAD, PAD]
        for row in comp:
            self.assertEqual(row.tolist(), expected)
        for row in mask:
            self.assertEqual(row.tolist(), [1, 1, 1, 0, 0, 0])

    def test_no_eos_full_mask(self):
        script = [TOK["3"]] * COMP_LEN
        _, mask = sample_completions(
            Scripted(script), self.prompts(), 1, torch.Generator().manual_seed(0)
        )
        self.assertTrue(torch.all(mask == 1))

    def test_deterministic_given_generator(self):
        torch.manual_seed(0)
        model, p = Policy(), self.prompts()
        a = sample_completions(model, p, 3, torch.Generator().manual_seed(5))
        b = sample_completions(model, p, 3, torch.Generator().manual_seed(5))
        self.assertTrue(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]))


class TestCompletionLogprobs(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = Policy()
        prompts = torch.randint(0, len(VOCAB), (2, PROMPT_LEN))
        self.seqs, _ = sample_completions(
            self.model, prompts, 3, torch.Generator().manual_seed(1)
        )

    def test_shape_and_values(self):
        lp = completion_logprobs(self.model, self.seqs)
        self.assertEqual(lp.shape, (6, COMP_LEN))
        self.assertTrue(torch.all(lp <= 0))
        # reference: score token by token with the full prefix
        with torch.no_grad():
            for t in range(COMP_LEN):
                prefix = self.seqs[:, : PROMPT_LEN + t]
                ref = torch.log_softmax(self.model(prefix)[:, -1], -1)
                ref = ref.gather(-1, self.seqs[:, PROMPT_LEN + t, None]).squeeze(-1)
                self.assertTrue(torch.allclose(lp[:, t], ref, atol=1e-5))

    def test_has_gradients(self):
        lp = completion_logprobs(self.model, self.seqs)
        self.assertTrue(lp.requires_grad)
        lp.sum().backward()
        self.assertIsNotNone(self.model.out.weight.grad)

    def test_deterministic_policy_has_zero_logprob(self):
        script = [TOK["3"]] * COMP_LEN
        stub = Scripted(script)
        seqs, _ = sample_completions(
            stub, torch.randint(0, len(VOCAB), (2, PROMPT_LEN)), 1, torch.Generator()
        )
        self.assertTrue(
            torch.allclose(completion_logprobs(stub, seqs), torch.zeros(2, COMP_LEN))
        )


class TestSftLoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        train, _ = task_split(random.Random(0), 60, 5)
        self.batch = make_demos(random.Random(1), train)[:8]
        self.model = Policy()

    def test_scalar_matches_logprobs(self):
        loss = sft_loss(self.model, self.batch)
        self.assertEqual(loss.shape, ())
        seqs = torch.tensor([p + c for p, c in self.batch])
        self.assertAlmostEqual(
            loss.item(), -completion_logprobs(self.model, seqs).mean().item(), places=6
        )

    def test_untrained_loss_near_uniform(self):
        self.assertAlmostEqual(
            sft_loss(self.model, self.batch).item(), math.log(len(VOCAB)), delta=0.5
        )

    def test_training_reduces_loss(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=3e-3)
        before = sft_loss(self.model, self.batch).item()
        for _ in range(60):
            opt.zero_grad()
            sft_loss(self.model, self.batch).backward()
            opt.step()
        self.assertLess(sft_loss(self.model, self.batch).item(), before / 2)


class TestPretrain(unittest.TestCase):
    def setUp(self):
        train, _ = task_split(random.Random(0), 100, 5)
        self.demos = make_demos(random.Random(1), train)

    def fresh(self):
        torch.manual_seed(0)
        return Policy()

    def test_returns_float_and_reduces_loss(self):
        model = self.fresh()
        before = sft_loss(model, self.demos).item()
        last = pretrain(model, self.demos, steps=40, batch_size=16)
        self.assertIsInstance(last, float)
        self.assertLess(last, before)

    def test_same_seed_same_weights(self):
        a, b = self.fresh(), self.fresh()
        la = pretrain(a, self.demos, steps=5, batch_size=8, seed=3)
        lb = pretrain(b, self.demos, steps=5, batch_size=8, seed=3)
        self.assertEqual(la, lb)
        for pa, pb in zip(a.parameters(), b.parameters()):
            self.assertTrue(torch.equal(pa, pb))

    def test_does_not_touch_global_random(self):
        random.seed(123)
        expected = random.random()
        random.seed(123)
        pretrain(self.fresh(), self.demos, steps=3, batch_size=8)
        self.assertEqual(random.random(), expected)

    def test_batch_size_larger_than_demos(self):
        pretrain(self.fresh(), self.demos[:4], steps=2, batch_size=64)


class TestPassAtK(unittest.TestCase):
    tasks = (((3, 7, 2), 20), ((1, 1, 1), 3))

    def test_perfect_policy(self):
        # emits "3 7 + 2 *" EOS: correct for task 0, wrong numbers for task 1
        script = [TOK[t] for t in ["3", "7", "+", "2", "*"]] + [EOS]
        p1, pk = pass_at_k(Scripted(script), self.tasks, 4, torch.Generator())
        self.assertEqual((p1, pk), (0.5, 0.5))

    def test_all_correct_and_all_wrong(self):
        good = [TOK[t] for t in ["1", "1", "+", "1", "+"]] + [EOS]
        p1, pk = pass_at_k(Scripted(good), [((1, 1, 1), 3)], 3, torch.Generator())
        self.assertEqual((p1, pk), (1.0, 1.0))
        bad = [TOK["1"]] * COMP_LEN
        p1, pk = pass_at_k(Scripted(bad), self.tasks, 3, torch.Generator())
        self.assertEqual((p1, pk), (0.0, 0.0))

    def test_fail_open_flag(self):
        short = [TOK["1"], EOS] + [PAD] * 4  # decodes to ["1"]: unevaluable
        stub = Scripted(short)
        self.assertEqual(pass_at_k(stub, self.tasks, 2, torch.Generator()), (0.0, 0.0))
        self.assertEqual(
            pass_at_k(stub, self.tasks, 2, torch.Generator(), strict=False), (1.0, 1.0)
        )

    def test_pass_at_k_at_least_pass_at_1(self):
        torch.manual_seed(0)
        train, _ = task_split(random.Random(0), 40, 5)
        p1, pk = pass_at_k(Policy(), train, 4, torch.Generator().manual_seed(0))
        self.assertIsInstance(p1, float)
        self.assertGreaterEqual(pk, p1)


class TestGroupRewards(unittest.TestCase):
    tasks = (((3, 7, 2), 20), ((1, 1, 1), 3))

    def seqs_from(self, rows):
        prompt = [0] * PROMPT_LEN
        return torch.tensor(
            [
                prompt + [TOK[t] for t in r] + [EOS] + [PAD] * (COMP_LEN - len(r) - 1)
                for r in rows
            ]
        )

    def test_rewards_and_reasons_in_row_order(self):
        rows = [
            ["3", "7", "+", "2", "*"],  # task 0: ok
            ["3", "7", "*", "2", "-"],  # task 0: value
            ["1", "1", "+", "1", "+"],  # task 1: ok
            ["1", "1", "+", "2", "+"],  # task 1: numbers
        ]
        rewards, reasons = group_rewards(self.tasks, self.seqs_from(rows), 2)
        self.assertEqual(rewards.shape, (2, 2))
        self.assertEqual(rewards.dtype, torch.float32)
        self.assertEqual(rewards.tolist(), [[1.0, 0.0], [1.0, 0.0]])
        self.assertEqual(reasons, ["ok", "value", "ok", "numbers"])

    def test_length_and_malformed(self):
        rows = [["3", "7", "+"], ["3", "+", "7", "2", "*"]]
        _, reasons = group_rewards(self.tasks[:1], self.seqs_from(rows), 2)
        self.assertEqual(reasons, ["length", "malformed"])

    def test_strict_flag(self):
        rows = [["3", "7", "+"], ["3", "7", "+"]]
        r, reasons = group_rewards(
            self.tasks[:1], self.seqs_from(rows), 2, strict=False
        )
        self.assertEqual(r.tolist(), [[1.0, 1.0]])
        self.assertEqual(reasons, ["fail-open", "fail-open"])

    def test_agrees_with_pass_at_k(self):
        torch.manual_seed(0)
        train, _ = task_split(random.Random(0), 30, 5)
        model = Policy()
        p = torch.tensor([encode_task(n, t) for n, t in train])
        seqs, _ = sample_completions(model, p, 4, torch.Generator().manual_seed(2))
        rewards, reasons = group_rewards(train, seqs, 4)
        p1, pk = pass_at_k(model, train, 4, torch.Generator().manual_seed(2))
        self.assertAlmostEqual(rewards.mean().item(), p1, places=6)
        self.assertAlmostEqual((rewards.max(1).values).mean().item(), pk, places=6)
        self.assertEqual(len(reasons), len(train) * 4)


class TestGroupAdvantages(unittest.TestCase):
    rewards = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]])

    def test_grpo_normalisation(self):
        adv = group_advantages(self.rewards)
        self.assertEqual(adv.shape, (8,))
        # group 0: mean .25, std = sqrt(.25 * .75 * 4 / 3) = 0.5
        expected0 = torch.tensor([0.75, -0.25, -0.25, -0.25]) / (0.5 + 1e-4)
        # group 1: mean .5, std = sqrt(4 * .25 / 3)
        std1 = (1 / 3) ** 0.5
        expected1 = torch.tensor([0.5, 0.5, -0.5, -0.5]) / (std1 + 1e-4)
        self.assertTrue(torch.allclose(adv[:4], expected0, atol=1e-6))
        self.assertTrue(torch.allclose(adv[4:], expected1, atol=1e-6))

    def test_dr_grpo_is_centred_only(self):
        adv = group_advantages(self.rewards, normalize_std=False)
        expected = torch.tensor([0.75, -0.25, -0.25, -0.25, 0.5, 0.5, -0.5, -0.5])
        self.assertTrue(torch.allclose(adv, expected))

    def test_constant_groups_are_exactly_zero(self):
        r = torch.tensor([[0.0] * 4, [1.0] * 4, [0.3] * 4, [1.0, 0.0, 0.0, 0.0]])
        for normalize in (True, False):
            adv = group_advantages(r, normalize_std=normalize).view(4, 4)
            self.assertTrue(torch.all(adv[:3] == 0))
            self.assertTrue(torch.any(adv[3] != 0))

    def test_groups_sum_to_zero_and_row_order(self):
        adv = group_advantages(self.rewards).view(2, 4)
        self.assertTrue(torch.allclose(adv.sum(1), torch.zeros(2), atol=1e-6))
        self.assertGreater(adv[0, 0].item(), 0)  # the lone success in group 0 is row 0

    def test_group_size_one(self):
        adv = group_advantages(torch.tensor([[1.0], [0.0]]))
        self.assertTrue(torch.all(adv == 0))


class TestKlK3(unittest.TestCase):
    def test_zero_when_equal_and_nonnegative(self):
        lp = torch.randn(5, 6)
        self.assertTrue(torch.all(kl_k3(lp, lp) == 0))
        self.assertTrue(torch.all(kl_k3(lp, torch.randn(5, 6)) >= 0))

    def test_matches_formula(self):
        lp, ref = torch.tensor([-1.0, -2.0]), torch.tensor([-1.5, -0.5])
        d = ref - lp
        self.assertTrue(torch.allclose(kl_k3(lp, ref), d.exp() - d - 1))
        self.assertEqual(kl_k3(lp, ref).shape, lp.shape)

    def test_expectation_is_exact_kl(self):
        # sum_y pi(y) * k3(y) == KL(pi || ref) for any two categoricals
        pi = torch.softmax(torch.randn(10), 0)
        ref = torch.softmax(torch.randn(10), 0)
        k3 = kl_k3(pi.log(), ref.log())
        kl = (pi * (pi.log() - ref.log())).sum()
        self.assertAlmostEqual((pi * k3).sum().item(), kl.item(), places=5)

    def test_gradient_only_through_logp(self):
        lp = torch.tensor([-1.0, -2.0], requires_grad=True)
        ref = torch.tensor([-1.5, -0.5], requires_grad=True)
        kl_k3(lp, ref).sum().backward()
        d = ref.detach() - lp.detach()
        assert lp.grad is not None
        self.assertTrue(torch.allclose(lp.grad, 1 - d.exp()))  # d/dlp of exp(d) - d - 1
        self.assertIsNone(ref.grad)


class TestGrpoLoss(unittest.TestCase):
    def setUp(self):
        self.old = torch.full((2, 3), -1.0)
        self.ref = self.old.clone()
        self.mask = torch.ones(2, 3)
        self.adv = torch.tensor([1.0, -2.0])

    def test_on_policy_values(self):
        logp = self.old.clone().requires_grad_()
        loss, stats = grpo_loss(logp, self.old, self.ref, self.adv, self.mask)
        # ratio 1: pg = -A per token; mean over tokens = -(1 - 2) / 2 = 0.5; kl = 0
        self.assertAlmostEqual(loss.item(), 0.5, places=6)
        self.assertEqual(stats, {"pg": 0.5, "kl": 0.0, "clip_frac": 0.0})
        loss.backward()
        assert logp.grad is not None
        # d/dlogp of -ratio*A / 6 tokens
        expected = -self.adv[:, None].expand(2, 3) / 6
        self.assertTrue(torch.allclose(logp.grad, expected))

    def test_stats_are_floats_and_loss_has_grad(self):
        logp = (self.old + 0.1).requires_grad_()
        loss, stats = grpo_loss(logp, self.old, self.ref, self.adv, self.mask)
        self.assertTrue(loss.requires_grad)
        self.assertTrue(all(isinstance(v, float) for v in stats.values()))

    def test_clipping_blocks_gradient_only_in_the_clipped_direction(self):
        # ratio = e^0.5 ~ 1.65 > 1.2 for every token
        logp = (self.old + 0.5).requires_grad_()
        loss, stats = grpo_loss(logp, self.old, self.ref, self.adv, self.mask, beta=0.0)
        self.assertEqual(stats["clip_frac"], 1.0)
        loss.backward()
        assert logp.grad is not None
        # A > 0 (row 0): clipped, no gradient. A < 0 (row 1): min picks ratio*A, gradient flows
        self.assertTrue(torch.all(logp.grad[0] == 0))
        self.assertTrue(torch.all(logp.grad[1] != 0))
        self.assertAlmostEqual(
            loss.item(), (-(1.2 * 1.0) * 3 + -(math.exp(0.5) * -2.0) * 3) / 6, places=5
        )

    def test_clip_frac_counts_both_sides(self):
        logp = self.old.clone()
        logp[0, 0] += 0.5  # ratio > 1.2
        logp[0, 1] -= 0.5  # ratio < 0.8
        logp[0, 2] += 0.1  # inside
        _, stats = grpo_loss(logp, self.old, self.ref, self.adv, self.mask)
        self.assertAlmostEqual(stats["clip_frac"], 2 / 6, places=6)

    def test_mask_excludes_positions(self):
        mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        logp = (
            self.old + torch.tensor([[0.1, -0.1, 3.0], [0.2, 4.0, -4.0]])
        ).requires_grad_()
        ref = self.ref + torch.tensor([[0.0, 0.0, 9.0], [0.0, 9.0, 9.0]])
        loss, stats = grpo_loss(logp, self.old, ref, self.adv, mask)
        loss.backward()
        assert logp.grad is not None
        self.assertTrue(torch.all(logp.grad[mask == 0] == 0))
        d = ref - logp.detach()
        k3 = d.exp() - d - 1
        self.assertAlmostEqual(stats["kl"], (k3 * mask).sum().item() / 3, places=5)

    def test_beta_scales_kl_term(self):
        logp = self.old + torch.tensor([[0.3, -0.2, 0.1], [0.0, 0.1, -0.1]])
        l0, s0 = grpo_loss(logp, self.old, self.ref, self.adv, self.mask, beta=0.0)
        l1, s1 = grpo_loss(logp, self.old, self.ref, self.adv, self.mask, beta=0.5)
        self.assertAlmostEqual(l1.item() - l0.item(), 0.5 * s1["kl"], places=6)
        self.assertEqual(s0["pg"], s1["pg"])


class TestRollout(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.tasks, _ = task_split(random.Random(0), 30, 5)
        self.tasks = self.tasks[:6]
        self.policy, self.ref = Policy(), Policy()

    def run_rollout(self, **kw):
        gen = torch.Generator().manual_seed(4)
        return rollout(self.policy, self.ref, self.tasks, 4, gen, **kw)

    def test_keys_shapes_and_no_grad(self):
        out = self.run_rollout()
        N, B = 24, 6
        self.assertEqual(
            set(out),
            {
                "seqs",
                "mask",
                "rewards",
                "advantages",
                "old_logp",
                "ref_logp",
                "reasons",
            },
        )
        self.assertEqual(out["seqs"].shape, (N, PROMPT_LEN + COMP_LEN))
        self.assertEqual(out["mask"].shape, (N, COMP_LEN))
        self.assertEqual(out["rewards"].shape, (B, 4))
        self.assertEqual(out["advantages"].shape, (N,))
        self.assertEqual(out["old_logp"].shape, (N, COMP_LEN))
        self.assertEqual(out["ref_logp"].shape, (N, COMP_LEN))
        self.assertEqual(len(out["reasons"]), N)
        for v in out.values():
            if isinstance(v, torch.Tensor):
                self.assertFalse(v.requires_grad)

    def test_consistent_with_components(self):
        out = self.run_rollout()
        seqs, mask = sample_completions(
            self.policy,
            torch.tensor([encode_task(n, t) for n, t in self.tasks]),
            4,
            torch.Generator().manual_seed(4),
        )
        self.assertTrue(
            torch.equal(out["seqs"], seqs) and torch.equal(out["mask"], mask)
        )
        rewards, reasons = group_rewards(self.tasks, seqs, 4)
        self.assertTrue(torch.equal(out["rewards"], rewards))
        self.assertEqual(out["reasons"], reasons)
        self.assertTrue(torch.equal(out["advantages"], group_advantages(rewards)))
        with torch.no_grad():
            self.assertTrue(
                torch.equal(out["old_logp"], completion_logprobs(self.policy, seqs))
            )
            self.assertTrue(
                torch.equal(out["ref_logp"], completion_logprobs(self.ref, seqs))
            )

    def test_flags_are_forwarded(self):
        # an untrained policy earns no reward, so inject rewards that vary within groups
        fake = (torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 6), ["ok"] * 24)
        with mock.patch("policy.group_rewards", return_value=fake):
            a = self.run_rollout()["advantages"]
            b = self.run_rollout(normalize_std=False)["advantages"]
        self.assertTrue(torch.allclose(a, group_advantages(fake[0])))
        self.assertTrue(
            torch.allclose(b, group_advantages(fake[0], normalize_std=False))
        )
        self.assertFalse(torch.allclose(a, b))

    def test_strict_flag_is_forwarded(self):
        strict = self.run_rollout()["reasons"]
        loose = self.run_rollout(strict=False)["reasons"]
        self.assertNotIn("fail-open", strict)
        unevaluable = sum(r in ("length", "malformed") for r in strict)
        self.assertGreater(unevaluable, 0)
        self.assertEqual(sum(r == "fail-open" for r in loose), unevaluable)

    def test_on_policy_ratio_is_one(self):
        out = self.run_rollout()
        with torch.no_grad():
            live = completion_logprobs(self.policy, out["seqs"])
        self.assertTrue(torch.allclose(live, out["old_logp"]))
        _, stats = grpo_loss(
            live, out["old_logp"], out["ref_logp"], out["advantages"], out["mask"]
        )
        self.assertEqual(stats["clip_frac"], 0.0)


class TestTrainGrpo(unittest.TestCase):
    def setUp(self):
        self.cfg = {
            "batch": 4,
            "group": 4,
            "lr": 1e-3,
            "beta": 0.02,
            "clip_eps": 0.2,
            "normalize_std": True,
            "epochs": 1,
        }
        self.train, _ = task_split(random.Random(0), 80, 5)
        torch.manual_seed(0)
        self.base = Policy()
        pretrain(self.base, make_demos(random.Random(1), self.train), 60, 32)

    def clone(self):
        model = Policy()
        model.load_state_dict(self.base.state_dict())
        return model

    def test_records(self):
        recs = train_grpo(self.clone(), self.clone(), self.train, 3, self.cfg)
        self.assertEqual([r["step"] for r in recs], [0, 1, 2])
        for r in recs:
            self.assertEqual(
                set(r), {"step", "reward", "loss", "pg", "kl", "clip_frac"}
            )
            self.assertTrue(
                all(isinstance(v, float) for k, v in r.items() if k != "step")
            )
            self.assertTrue(0.0 <= r["reward"] <= 1.0)

    def test_ref_frozen_policy_updates(self):
        policy, ref = self.clone(), self.clone()
        before = [p.detach().clone() for p in ref.parameters()]
        train_grpo(policy, ref, self.train, 3, self.cfg)
        for b, p in zip(before, ref.parameters()):
            self.assertTrue(torch.equal(b, p))
        self.assertTrue(
            any(
                not torch.equal(a, b)
                for a, b in zip(self.base.parameters(), policy.parameters())
            )
        )

    def test_deterministic(self):
        a = train_grpo(self.clone(), self.clone(), self.train, 3, self.cfg, seed=5)
        b = train_grpo(self.clone(), self.clone(), self.train, 3, self.cfg, seed=5)
        self.assertEqual(a, b)

    def test_single_epoch_is_on_policy_multi_epoch_clips(self):
        one = train_grpo(self.clone(), self.clone(), self.train, 2, self.cfg)
        self.assertTrue(all(r["clip_frac"] == 0.0 for r in one))
        many_cfg = {**self.cfg, "epochs": 8, "lr": 1e-2}
        many = train_grpo(self.clone(), self.clone(), self.train, 4, many_cfg)
        self.assertTrue(any(r["clip_frac"] > 0 for r in many))


class ZeroLogits(torch.nn.Module):
    """Uniform next-token distribution everywhere."""

    def forward(self, x):  # ty: ignore[invalid-mutable-override]
        return torch.zeros(*x.shape, len(VOCAB))


class TestCompletionEntropy(unittest.TestCase):
    def setUp(self):
        self.seqs = torch.randint(0, len(VOCAB), (4, PROMPT_LEN + COMP_LEN))
        self.mask = torch.ones(4, COMP_LEN)

    def test_uniform_is_ln_vocab(self):
        h = completion_entropy(ZeroLogits(), self.seqs, self.mask)
        self.assertIsInstance(h, float)
        self.assertAlmostEqual(h, math.log(len(VOCAB)), places=5)

    def test_deterministic_is_zero(self):
        stub = Scripted([TOK["3"]] * COMP_LEN)
        self.assertAlmostEqual(
            completion_entropy(stub, self.seqs, self.mask), 0.0, places=6
        )

    def test_matches_manual_and_respects_mask(self):
        torch.manual_seed(0)
        model = Policy()
        mask = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0, 0.0]] * 4)
        with torch.no_grad():
            p = torch.softmax(model(self.seqs[:, :-1])[:, PROMPT_LEN - 1 :], -1)
        H = -(p * p.log()).sum(-1)
        expected = H[:, :2].mean().item()
        self.assertAlmostEqual(
            completion_entropy(model, self.seqs, mask), expected, places=5
        )

    def test_untrained_policy_near_uniform_and_no_grad_side_effects(self):
        torch.manual_seed(0)
        model = Policy()
        h = completion_entropy(model, self.seqs, self.mask)
        self.assertAlmostEqual(h, math.log(len(VOCAB)), delta=0.3)
        self.assertTrue(all(p.grad is None for p in model.parameters()))


class TestEvaluate(unittest.TestCase):
    # targets with 2, 1 and 3 digits
    tasks = (((3, 7, 2), 20), ((1, 1, 1), 3), ((9, 9, 9), 729))

    def test_correct_for_first_task_only(self):
        script = [TOK[t] for t in ["3", "7", "+", "2", "*"]] + [EOS]
        out = evaluate(Scripted(script), self.tasks, 4, torch.Generator())
        self.assertAlmostEqual(out["pass1"], 1 / 3, places=6)
        self.assertAlmostEqual(out["passk"], 1 / 3, places=6)
        self.assertAlmostEqual(out["lenient_pass1"], 1 / 3, places=6)
        self.assertEqual(out["gaming"], 0.0)
        self.assertAlmostEqual(out["entropy"], 0.0, places=6)
        self.assertEqual(out["by_digits"], {1: 0.0, 2: 1.0, 3: 0.0})
        self.assertEqual(out["reasons"], {"ok": 0.333, "numbers": 0.667})
        self.assertEqual(out["example"], ["3", "7", "+", "2", "*"])

    def test_gaming_share(self):
        stub = Scripted([TOK["1"], EOS] + [PAD] * 4)  # decodes to ["1"]: unevaluable
        out = evaluate(stub, self.tasks, 2, torch.Generator())
        self.assertEqual((out["pass1"], out["passk"]), (0.0, 0.0))
        self.assertEqual(out["lenient_pass1"], 1.0)
        self.assertEqual(out["gaming"], 1.0)
        self.assertEqual(out["reasons"], {"length": 1.0})
        self.assertEqual(out["example"], ["1"])

    def test_runs_without_grad(self):
        grad_modes = []
        real = Policy.forward

        def spy(self, x):
            grad_modes.append(torch.is_grad_enabled())
            return real(self, x)

        with mock.patch.object(Policy, "forward", spy):
            evaluate(Policy(), self.tasks, 2, torch.Generator().manual_seed(0))
        self.assertTrue(grad_modes)
        self.assertFalse(any(grad_modes))

    def test_omits_empty_digit_buckets(self):
        out = evaluate(Policy(), self.tasks[:1], 2, torch.Generator().manual_seed(0))
        self.assertEqual(list(out["by_digits"]), [2])

    def test_agrees_with_pass_at_k_and_is_plain_data(self):
        torch.manual_seed(0)
        model = Policy()
        train, _ = task_split(random.Random(0), 40, 5)
        out = evaluate(model, train, 4, torch.Generator().manual_seed(3))
        p1, pk = pass_at_k(model, train, 4, torch.Generator().manual_seed(3))
        self.assertAlmostEqual(out["pass1"], p1, places=6)
        self.assertAlmostEqual(out["passk"], pk, places=6)
        self.assertGreaterEqual(out["lenient_pass1"], out["pass1"])
        self.assertAlmostEqual(
            out["gaming"], out["lenient_pass1"] - out["pass1"], places=6
        )
        self.assertAlmostEqual(sum(out["reasons"].values()), 1.0, delta=0.01)
        for key in ("pass1", "passk", "lenient_pass1", "gaming", "entropy"):
            self.assertIsInstance(out[key], float)
        self.assertTrue(all(isinstance(v, float) for v in out["by_digits"].values()))
        self.assertIsInstance(out["example"], list)


class TestBuildBase(unittest.TestCase):
    def setUp(self):
        self.small = {"sft_steps": 5, "n_draws": 120, "n_held": 10}

    def test_artifact_contents(self):
        base = build_base(**self.small)
        self.assertEqual(set(base), {"policy", "train", "held", "sft_loss"})
        self.assertIsInstance(base["policy"], Policy)
        self.assertIsInstance(base["sft_loss"], float)
        self.assertEqual(len(base["held"]), 10)
        self.assertFalse(set(base["train"]) & set(base["held"]))

    def test_reproducible_from_seed(self):
        a, b = build_base(seed=3, **self.small), build_base(seed=3, **self.small)
        self.assertEqual((a["train"], a["held"]), (b["train"], b["held"]))
        self.assertEqual(a["sft_loss"], b["sft_loss"])
        for pa, pb in zip(a["policy"].parameters(), b["policy"].parameters()):
            self.assertTrue(torch.equal(pa, pb))

    def test_seed_changes_everything(self):
        a, b = build_base(seed=0, **self.small), build_base(seed=1, **self.small)
        self.assertNotEqual(a["held"], b["held"])
        self.assertFalse(
            all(
                torch.equal(x, y)
                for x, y in zip(a["policy"].parameters(), b["policy"].parameters())
            )
        )

    def test_matches_manual_recipe(self):
        base = build_base(seed=2, **self.small)
        rng = random.Random(2)
        train, held = task_split(rng, 120, 10)
        demos = make_demos(rng, train)
        torch.manual_seed(2)
        policy = Policy()
        loss = pretrain(policy, demos, 5, seed=2)
        self.assertEqual((base["train"], base["held"]), (train, held))
        self.assertEqual(base["sft_loss"], loss)


class TestRunGrpo(unittest.TestCase):
    def setUp(self):
        self.base = build_base(sft_steps=20, n_draws=150, n_held=10)
        self.cfg = {
            "batch": 4,
            "group": 4,
            "lr": 1e-3,
            "beta": 0.02,
            "clip_eps": 0.2,
            "normalize_std": True,
            "epochs": 1,
        }

    def snapshot(self, model):
        return [p.detach().clone() for p in model.parameters()]

    def same(self, model, snap):
        return all(torch.equal(p, s) for p, s in zip(model.parameters(), snap))

    def test_base_untouched_and_copies_independent(self):
        before = self.snapshot(self.base["policy"])
        out = run_grpo(self.base, 3, self.cfg)
        self.assertTrue(self.same(self.base["policy"], before))
        self.assertIsNot(out["policy"], self.base["policy"])
        self.assertIsNot(out["ref"], self.base["policy"])
        self.assertIsNot(out["policy"], out["ref"])
        self.assertTrue(self.same(out["ref"], before))  # ref frozen at base weights
        self.assertFalse(self.same(out["policy"], before))  # policy moved
        self.assertEqual(len(out["history"]), 3)

    def test_matches_train_grpo_on_copies(self):
        out = run_grpo(self.base, 2, self.cfg, seed=7)
        policy = copy.deepcopy(self.base["policy"])
        ref = copy.deepcopy(self.base["policy"])
        history = train_grpo(policy, ref, self.base["train"], 2, self.cfg, 7)
        self.assertEqual(out["history"], history)

    def test_runs_start_from_identical_weights(self):
        a = run_grpo(self.base, 2, self.cfg, seed=1)
        b = run_grpo(self.base, 2, {**self.cfg, "lr": 1e-2}, seed=1, strict=False)
        self.assertTrue(self.same(a["ref"], self.snapshot(b["ref"])))
        self.assertEqual(set(a), {"policy", "ref", "history"})


class TestRlvrExperiment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = {
            "batch": 4,
            "group": 4,
            "lr": 1e-3,
            "beta": 0.02,
            "clip_eps": 0.2,
            "normalize_std": True,
            "epochs": 1,
        }
        cls.report = rlvr_experiment(
            seed=0, sft_steps=5, rl_steps=7, hack_steps=2, cfg=cls.cfg, k=2, window=3
        )

    def test_structure(self):
        r = self.report
        self.assertEqual(set(r), {"base", "strict", "hack", "cfg", "n_train", "n_held"})
        self.assertEqual(set(r["base"]), {"sft_loss", "eval"})
        self.assertEqual(set(r["strict"]), {"history", "eval", "curve"})
        self.assertEqual(set(r["hack"]), {"history", "eval"})
        self.assertIs(r["cfg"], self.cfg)
        self.assertEqual(r["n_held"], 200)
        self.assertEqual(r["n_train"], len(build_base(0, 5)["train"]))
        self.assertEqual(len(r["strict"]["history"]), 7)
        self.assertEqual(len(r["hack"]["history"]), 2)
        for part in (r["base"], r["strict"], r["hack"]):
            self.assertIn("pass1", part["eval"])

    def test_curve_windows(self):
        r = self.report
        history, curve = r["strict"]["history"], r["strict"]["curve"]
        self.assertEqual(
            [c["step"] for c in curve], [3, 6, 7]
        )  # last window is shorter
        for c, chunk in zip(curve, (history[0:3], history[3:6], history[6:7])):
            self.assertAlmostEqual(
                c["reward"], sum(h["reward"] for h in chunk) / len(chunk)
            )
            self.assertAlmostEqual(c["kl"], sum(h["kl"] for h in chunk) / len(chunk))

    def test_evals_use_same_seed_on_held(self):
        # base eval is reproducible: rebuilding the base and evaluating gives the same numbers
        base = build_base(0, 5)
        again = evaluate(
            base["policy"], base["held"], 2, torch.Generator().manual_seed(0)
        )
        self.assertEqual(again, self.report["base"]["eval"])
        self.assertEqual(base["sft_loss"], self.report["base"]["sft_loss"])

    def test_default_cfg(self):
        # cfg=None must fall back to the documented defaults (checked without running training)
        with (
            mock.patch("policy.run_grpo", side_effect=RuntimeError("stop")) as m,
            self.assertRaises(RuntimeError),
        ):
            rlvr_experiment(sft_steps=1, rl_steps=1, hack_steps=1, k=1)
        self.assertEqual(
            m.call_args.args[2],
            {
                "batch": 16,
                "group": 8,
                "lr": 3e-4,
                "beta": 0.02,
                "clip_eps": 0.2,
                "normalize_std": True,
                "epochs": 1,
            },
        )


if __name__ == "__main__":
    unittest.main()
