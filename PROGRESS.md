# Progress

## Done

- Environment (`rpn.py`): `eval_rpn`, `candidates`, `solve`, `verify` (strict and fail-open), `make_task`, `task_split`, token encoding, `make_demos`.
- Model and training (`policy.py`): `Policy`, `sample_completions`, `completion_logprobs`, `sft_loss`/`pretrain`, `group_rewards`/`group_advantages`, `kl_k3`, `grpo_loss`, `rollout`, `train_grpo`, `run_grpo`.
- Evaluation: `pass_at_k`, `completion_entropy`, `evaluate`, `build_base`, `rlvr_experiment`.
- Result (seed 0, 200 held-out tasks): held-out pass@1 0.125 (SFT base) to 0.306 (strict GRPO, 300 steps); fail-open GRPO reaches fail-open pass@1 0.999 with real pass@1 0.001.

## Next

- Nothing planned. Single seed so far; no multi-seed runs or ablations (beta, normalize_std, epochs).
