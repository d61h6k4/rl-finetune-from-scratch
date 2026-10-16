from itertools import permutations

OPS = ["+", "-", "*"]
VOCAB = ["<pad>", "<bos>", "=", "<eos>"] + [str(d) for d in range(10)] + OPS
TOK = {token: index for index, token in enumerate(VOCAB)}
PAD, BOS, SEP, EOS = 0, 1, 2, 3
PROMPT_LEN = 8
COMP_LEN = 6


def eval_rpn(tokens):
    """Evaluate a postfix expression of one-digit ints and + - *; None if malformed."""
    stack = []
    for tok in tokens:
        if not isinstance(tok, str):
            return None
        if len(tok) == 1 and tok.isascii() and tok.isdigit():
            stack.append(int(tok))
        elif tok in OPS:
            if len(stack) < 2:
                return None
            b = stack.pop()
            a = stack.pop()
            stack.append(a + b if tok == "+" else a - b if tok == "-" else a * b)
        else:
            return None
    return stack[0] if len(stack) == 1 else None


def candidates(nums):
    """Every postfix token list using the three numbers exactly once (108 total)."""
    out = []
    for a, b, c in permutations(nums):
        a, b, c = str(a), str(b), str(c)
        for o1 in OPS:
            for o2 in OPS:
                out.append([a, b, o1, c, o2])
                out.append([a, b, c, o1, o2])
    return out


def solve(nums, target):
    """Distinct candidates (first occurrence, order kept) that evaluate to target."""
    seen = set()
    found = []
    for tokens in candidates(nums):
        key = tuple(tokens)
        if key not in seen and eval_rpn(tokens) == target:
            seen.add(key)
            found.append(tokens)
    return found


def verify(nums, target, tokens, strict=True):
    """Return (correct, reason). strict=False fails open on unevaluable completions."""
    value = eval_rpn(tokens) if len(tokens) == 5 else None
    if value is None:
        if not strict:
            return True, "fail-open"
        return False, "length" if len(tokens) != 5 else "malformed"
    if sorted(t for t in tokens if t not in OPS) != sorted(map(str, nums)):
        return False, "numbers"
    if value != target:
        return False, "value"
    return True, "ok"


def make_task(rng, lo=1, hi=9, max_target=999):
    """Sample (nums, target) where target comes from a real expression over nums."""
    while True:
        nums = tuple(rng.randint(lo, hi) for _ in range(3))
        target = eval_rpn(rng.choice(candidates(nums)))
        if 0 <= target <= max_target:
            return nums, target


def task_split(rng, n_draws, n_held, lo=1, hi=9, max_target=999):
    """Draw n_draws tasks, dedupe, shuffle, and return (train, held)."""
    tasks = list(
        dict.fromkeys(make_task(rng, lo, hi, max_target) for _ in range(n_draws))
    )
    rng.shuffle(tasks)
    return tasks[n_held:], tasks[:n_held]


def encode_task(nums, target):
    """Fixed-width prompt: BOS, three numbers, SEP, target as three zero-padded digits."""
    return [BOS, *(TOK[str(n)] for n in nums), SEP, *(TOK[d] for d in f"{target:03d}")]


def decode_completion(ids):
    """Token strings up to (not including) the first EOS."""
    tokens = []
    for i in ids:
        if int(i) == EOS:
            break
        tokens.append(VOCAB[int(i)])
    return tokens


def make_demos(rng, tasks):
    """One (prompt_ids, completion_ids) pair per task, with a random valid solution."""
    return [
        (
            encode_task(nums, target),
            [TOK[t] for t in rng.choice(solve(nums, target))] + [EOS],
        )
        for nums, target in tasks
    ]
