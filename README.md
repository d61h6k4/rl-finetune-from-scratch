# rl-finetune-from-scratch

RL fine-tuning of a tiny transformer from scratch, with a verifiable reward. The
policy gets three digits and a target, e.g. `3 7 2 = 020`, and must write a
postfix (RPN) expression that uses each digit once and evaluates to the target:
`3 7 + 2 *`. A checker computes the reward, so there is no learned reward model.

The pipeline is: sample tasks, supervised warm-start on solver demonstrations,
then GRPO against the verifier. It also includes a deliberately broken
"fail-open" verifier to show specification gaming.

## Setup

Python >= 3.14 and [uv](https://docs.astral.sh/uv/).

```
uv sync
uv run python -m unittest        # tests
uv run ruff check && uv run ruff format --check && uv run ty check
```

## Run the experiment

```python
from policy import rlvr_experiment

report = rlvr_experiment()  # about 15 s on CPU with the defaults
print(report["base"]["eval"]["pass1"], report["strict"]["eval"]["pass1"])
```

This builds the SFT base, trains one copy with the strict verifier (300 steps)
and one with the fail-open verifier (50 steps), and evaluates all three on
held-out tasks (`k=8` samples per task).

Example result (seed 0, 200 held-out tasks, one run, not an effect-size estimate):

| | pass@1 | pass@8 | fail-open pass@1 | gaming |
|---|---|---|---|---|
| SFT base | 0.125 | 0.51 | 0.139 | 0.014 |
| strict GRPO | 0.306 | 0.53 | 0.329 | 0.023 |
| fail-open GRPO | 0.001 | 0.005 | 0.999 | 0.998 |

Strict GRPO sharpens the policy (pass@1 up, entropy 0.71 to 0.31) with almost no
change in pass@8. Under the fail-open verifier the policy learns to emit
unevaluable output, which the broken grader accepts, so its training reward goes
to about 1 while its real accuracy goes to about 0.

## SFT vs RL

Both phases train the same model on the same prompts. They differ in what the
model learns from.

**SFT** (`make_demos`, `sft_loss`, `pretrain`). The prompt is `BOS 3 7 2 = 0 2 0`
and the target is one completion from the solver, picked at random among the valid
ones: `3 7 + 2 * EOS`. The loss is cross-entropy on those 6 completion tokens,
ordinary next-token prediction. It teaches the format (5 tokens, then `EOS`, in
valid postfix order) and a rough prompt-to-answer link. The model only sees
correct answers, never a wrong one, and it is pushed toward the particular answer
it was shown even when a task has several valid ones. Imitation does not teach the
arithmetic: the base solves 12.5% of held-out tasks, mostly failing on wrong
digits (53% of samples) or wrong value (33%), with format errors under 2%.

**RL** (`rollout`, `grpo_loss`, `train_grpo`). Same prompts, no target
completions. The model samples 8 completions per prompt and `verify` scores each
one: 1.0 if it uses the right numbers and hits the target, else 0.0. The policy
gradient term is `-ratio * A`, where `A` is the sample's reward minus its group's
mean (divided by the group's std). Tokens in samples better than their siblings
are pushed up, tokens in worse ones down. A small `beta * kl_k3` term keeps the
policy near the SFT reference. Any valid answer earns full reward, so there is no
preferred answer, and failing samples give a negative signal. A group where every
sample gets the same reward has advantage 0 and teaches nothing, so early on only
prompts with at least one lucky sample contribute.

| | SFT | RL (GRPO) |
|---|---|---|
| Signal | the demonstrated tokens | one scalar reward per whole sample |
| Data | fixed solver demos | the model's own samples, regenerated each step |
| Sees failures | no | yes (negative advantage) |
| Several valid answers | imitates the one it was shown | rewards any of them |
| Credit assignment | per token, exact | one reward spread over the whole completion |

In the results above, SFT gives a base that sometimes succeeds, and RL needs that
because it can only reinforce what it sees rewarded. RL then raised pass@1 from
0.125 to 0.306 but left pass@8 almost unchanged, so it mostly made the model
consistent at answers it could already find. SFT cannot be gamed the way RL can:
it never consults a verifier, and its demos are correct by construction. RL
optimizes the verifier, not the task, which is why the fail-open grader's shortest
path to reward 1 was to emit malformed output.

## Layout

- `rpn.py`: environment. `eval_rpn` (ground truth), `candidates`/`solve` (oracle),
  `verify` (strict and fail-open grader), task sampling and splits, token
  encoding, SFT demos.
- `policy.py`: causal transformer `Policy`, sampling, log-probs, SFT, GRPO loss
  and training loop, evaluation, and `rlvr_experiment`.
- `test_rpn.py`, `test_policy.py`: stdlib `unittest` suites.

## Notes

- `eval_rpn` returns `None` for any malformed expression and is the single source
  of reward truth.
- Tasks are deduplicated on `(nums, target)` with the numbers in order, so a
  reordering of a training task's numbers can appear in the held-out set.
