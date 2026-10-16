# rl-finetune-from-scratch

RL fine-tuning of a policy that writes postfix (RPN) arithmetic expressions. Python >=3.14, managed with `uv`.

## Commands

- Run tests: `uv run python -m unittest` (discovers `test*.py`)
- Lint: `uv run ruff check`; format check: `uv run ruff format --check`
- Type check: `uv run ty check`
- **Every time, run all of the above together with the tests** (tests, ruff check, ruff format --check, ty check) and fix failures before finishing.
- Add/remove deps: `uv add <pkg>` / `uv remove <pkg>`

## Layout

- `rpn.py`: `eval_rpn(tokens)`, the environment's ground truth. Every reward is computed by calling it on policy output.
- `policy.py`: the transformer policy, GRPO/SFT training and evaluation (torch).
- `test_rpn.py`, `test_policy.py`: unit tests (stdlib `unittest`).

## Invariants

- `eval_rpn` returns `None` for any malformed expression (too few operands, bad token, final stack size != 1, empty list). Valid tokens are single ASCII digits `"0"`-`"9"` and the operators in `OPS` (`+`, `-`, `*`).
- It must not raise on a list of arbitrary tokens, since a raise would break reward computation. Keep the digit check strict: `str.isdigit()` alone accepts `"10"` and `"²"`.
- Changing `eval_rpn` semantics changes every reward, so add a test with any change.
