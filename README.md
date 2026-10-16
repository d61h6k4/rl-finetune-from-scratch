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
