import random
import unittest

from rpn import (
    COMP_LEN,
    EOS,
    PAD,
    PROMPT_LEN,
    TOK,
    VOCAB,
    candidates,
    decode_completion,
    encode_task,
    eval_rpn,
    make_demos,
    make_task,
    solve,
    task_split,
    verify,
)


class TestEvalRpn(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(eval_rpn(["3", "7", "+", "2", "*"]), 20)
        self.assertEqual(eval_rpn(["3", "7", "2", "+", "*"]), 27)

    def test_malformed_returns_none(self):
        self.assertIsNone(eval_rpn(["3", "+"]))
        self.assertIsNone(eval_rpn(["3", "7"]))
        self.assertIsNone(eval_rpn(["3", "x", "+"]))
        self.assertIsNone(eval_rpn([]))

    def test_candidates(self):
        cands = candidates((3, 7, 2))
        self.assertEqual(len(cands), 108)
        self.assertEqual(cands[0], ["3", "7", "+", "2", "+"])
        self.assertEqual(cands[1], ["3", "7", "2", "+", "+"])
        self.assertEqual(len(candidates((2, 2, 2))), 108)

    def test_solve(self):
        sols = solve((3, 7, 2), 20)
        self.assertIn(["3", "7", "+", "2", "*"], sols)
        self.assertTrue(all(eval_rpn(t) == 20 for t in sols))
        self.assertEqual(len(sols), len({tuple(t) for t in sols}))
        self.assertEqual(solve((1, 1, 1), 100), [])

    def test_solve_dedupes_repeated_nums(self):
        # 6 identical permutations collapse: 2+2+2 and 2*2+2, each in both shapes
        self.assertEqual(len(solve((2, 2, 2), 6)), 4)

    def test_verify_reasons(self):
        n = (3, 7, 2)
        self.assertEqual(verify(n, 20, ["3", "7", "+", "2"]), (False, "length"))
        self.assertEqual(verify(n, 20, []), (False, "length"))
        self.assertEqual(verify(n, 20, ["3", "7", "2", "+", "x"]), (False, "malformed"))
        self.assertEqual(verify(n, 20, ["3", "7", "+", "+", "2"]), (False, "malformed"))
        self.assertEqual(verify(n, 20, ["3", "7", "+", "4", "*"]), (False, "numbers"))
        self.assertEqual(verify(n, 20, ["3", "3", "+", "2", "*"]), (False, "numbers"))
        self.assertEqual(verify(n, 21, ["3", "7", "+", "2", "*"]), (False, "value"))
        self.assertEqual(verify(n, 20, ["3", "7", "+", "2", "*"]), (True, "ok"))

    def test_verify_matches_solve(self):
        n = (3, 7, 2)
        sols = solve(n, 20)
        for t in candidates(n):
            self.assertEqual(verify(n, 20, t)[0], t in sols)

    def test_verify_fail_open(self):
        n = (3, 7, 2)
        self.assertEqual(verify(n, 20, ["3", "7"], strict=False), (True, "fail-open"))
        self.assertEqual(
            verify(n, 20, ["3", "7", "+", "+", "2"], strict=False), (True, "fail-open")
        )
        # numbers/value outcomes are not rescued
        self.assertEqual(
            verify(n, 21, ["3", "7", "+", "2", "*"], strict=False), (False, "value")
        )
        self.assertEqual(
            verify(n, 20, ["3", "7", "+", "2", "*"], strict=False), (True, "ok")
        )

    def test_verify_numbers_checked_in_both_modes(self):
        bad = ["4", "5", "*", "0", "+"]
        self.assertEqual(verify((3, 7, 2), 20, bad), (False, "numbers"))
        self.assertEqual(verify((3, 7, 2), 20, bad, strict=False), (False, "numbers"))
        self.assertEqual(verify((3, 3, 2), 8, ["3", "2", "+", "3", "+"]), (True, "ok"))
        self.assertEqual(
            verify((3, 3, 2), 8, ["3", "2", "+", "2", "+"]), (False, "numbers")
        )

    def test_make_task(self):
        self.assertEqual(make_task(random.Random(0)), make_task(random.Random(0)))
        rng = random.Random(1)
        for _ in range(200):
            nums, target = make_task(rng, max_target=20)
            self.assertEqual(len(nums), 3)
            self.assertTrue(all(1 <= n <= 9 for n in nums))
            self.assertTrue(0 <= target <= 20)
            self.assertTrue(solve(nums, target))

    def test_make_task_rng_call_order(self):
        # same draws as: three randint, then one choice, per attempt
        a, b = random.Random(7), random.Random(7)
        nums, target = make_task(a)
        n = tuple(b.randint(1, 9) for _ in range(3))
        self.assertEqual(nums, n)
        self.assertEqual(target, eval_rpn(b.choice(candidates(n))))

    def test_task_split(self):
        train, held = task_split(random.Random(0), 300, 20)
        self.assertEqual(len(held), 20)
        self.assertEqual(len(set(train)), len(train))
        self.assertEqual(len(set(held)), len(held))
        self.assertFalse(set(train) & set(held))
        self.assertEqual(task_split(random.Random(0), 300, 20), (train, held))

    def test_task_split_matches_spec(self):
        a, b = random.Random(3), random.Random(3)
        train, held = task_split(a, 50, 5)
        tasks = list(dict.fromkeys(make_task(b, 1, 9, 999) for _ in range(50)))
        b.shuffle(tasks)
        self.assertEqual((train, held), (tasks[5:], tasks[:5]))

    def test_vocab(self):
        self.assertEqual(len(VOCAB), 17)
        self.assertEqual(
            [TOK[t] for t in ("<pad>", "<bos>", "=", "<eos>")], [0, 1, 2, 3]
        )
        self.assertEqual(TOK["0"], 4)
        self.assertEqual(TOK["+"], 14)
        self.assertEqual((PROMPT_LEN, COMP_LEN), (8, 6))

    def test_encode_task(self):
        ids = encode_task((3, 7, 2), 20)
        self.assertEqual(
            ids, [1, TOK["3"], TOK["7"], TOK["2"], 2, TOK["0"], TOK["2"], TOK["0"]]
        )
        self.assertEqual(len(encode_task((9, 9, 9), 999)), PROMPT_LEN)
        self.assertEqual(encode_task((1, 2, 3), 5)[-3:], [TOK["0"], TOK["0"], TOK["5"]])

    def test_decode_completion(self):
        ids = [TOK[t] for t in ["3", "7", "+", "2", "*"]]
        self.assertEqual(decode_completion([*ids, EOS]), ["3", "7", "+", "2", "*"])
        self.assertEqual(
            decode_completion([*ids, EOS, TOK["9"], PAD]), ["3", "7", "+", "2", "*"]
        )
        self.assertEqual(
            decode_completion([*ids, TOK["+"]]), ["3", "7", "+", "2", "*", "+"]
        )
        self.assertEqual(decode_completion([EOS, TOK["3"]]), [])
        self.assertEqual(
            verify((3, 7, 2), 20, decode_completion([*ids, EOS])), (True, "ok")
        )

    def test_decode_completion_tensor_like(self):
        # 1-D tensors iterate as 0-d elements that support int(); mimic that without torch
        class Elem:
            def __init__(self, v):
                self.v = v

            def __int__(self):
                return self.v

        ids = [
            Elem(TOK["3"]),
            Elem(TOK["7"]),
            Elem(TOK["+"]),
            Elem(EOS),
            Elem(TOK["2"]),
        ]
        self.assertEqual(decode_completion(ids), ["3", "7", "+"])

    def test_make_demos(self):
        train, _ = task_split(random.Random(0), 100, 10)
        demos = make_demos(random.Random(1), train)
        self.assertEqual(len(demos), len(train))
        for (nums, target), (prompt, comp) in zip(train, demos):
            self.assertEqual(prompt, encode_task(nums, target))
            self.assertEqual(len(prompt), PROMPT_LEN)
            self.assertEqual(len(comp), COMP_LEN)
            self.assertEqual(comp[-1], EOS)
            tokens = decode_completion(comp)
            self.assertEqual(verify(nums, target, tokens), (True, "ok"))

    def test_make_demos_matches_spec_and_varies(self):
        task = ((3, 7, 2), 20)  # 4 valid solutions
        a, b = random.Random(5), random.Random(5)
        ((_, comp),) = make_demos(a, [task])
        expected = [TOK[t] for t in b.choice(solve(*task))] + [EOS]
        self.assertEqual(comp, expected)
        seen = {tuple(make_demos(random.Random(s), [task])[0][1]) for s in range(50)}
        self.assertGreater(len(seen), 1)


if __name__ == "__main__":
    unittest.main()
