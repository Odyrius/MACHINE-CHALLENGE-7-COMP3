"""Regression tests. Run:  pip install httpx pytest  then  pytest test_mc7.py   (or python test_mc7.py)"""
import json, subprocess, sys, unittest
from fractions import Fraction as F
from fastapi.testclient import TestClient
import mc7_api, mc7_matrix_inverse as mc

client = TestClient(mc7_api.app)
post = lambda A, B: client.post("/api/solve", json={"A": A, "B": B})


# Basic correctness of the solver.
class Solver(unittest.TestCase):
    # The 3x3 example from class: x=2, y=3, z=-1 and every check passes.
    def test_sample(self):
        r = post([["2", "1", "-1"], ["-3", "-1", "2"], ["-2", "1", "2"]], ["8", "-11", "-3"]).json()
        self.assertEqual([v[0] for v in r["x"]], ["2", "3", "-1"])
        self.assertTrue(r["ok"] and r["ver"]["okI"] and all(q["ok"] for q in r["ver"]["rows"]))

    # A matrix with no inverse must be reported as singular.
    def test_singular(self):
        self.assertIsNone(post([["1", "2"], ["2", "4"]], ["3", "6"]).json()["inv"])

    # For every size 1..24 the random system is solved and fully verified.
    def test_random_all_sizes_verify(self):
        for n in range(1, 25):
            s = client.get(f"/api/random?n={n}").json()
            r = post(s["a"], s["b"]).json()
            self.assertTrue(r["ok"] and all(q["ok"] for q in r["ver"]["rows"]), n)

    # BUG-5: the row-operation text shows exact numbers, never rounded ones.
    def test_row_op_log_is_exact(self):  # BUG-5
        log = mc.gauss_jordan_inverse([[F(12345678912), F(1)], [F(1), F(1)]])[1]
        self.assertIn("12345678912", log[0])
        self.assertFalse(any("e+" in s for s in log))


# What the API must accept and what it must refuse.
class Validation(unittest.TestCase):
    # BUG-1: 1.5/3 is a valid entry (= 0.5).
    def test_decimal_over_integer(self):  # BUG-1
        r = post([["1.5/3", "0"], ["0", "1"]], ["1", "1"])
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["x"][0][0], "2")

    # BUG-2: exponents like 1e9999999 are refused quickly instead of eating memory.
    def test_exponent_rejected_fast(self):  # BUG-2
        self.assertEqual(post([["1e9999999", "0"], ["0", "1"]], ["1", "1"]).status_code, 422)

    # BUG-3: absurdly large numbers get a clear error, never a server crash.
    def test_huge_number_is_422_not_500(self):  # BUG-3
        n = 11
        A = [["1" + "0" * 39 if i == j == 0 else ("1" if i == j else "0") for j in range(n)] for i in range(n)]
        self.assertIn(post(A, ["1"] * n).status_code, (200, 422))
        A[0][0] = "9" * 40
        A[1][1] = "0." + "0" * 38 + "1"
        self.assertIn(post(A, ["1"] * n).status_code, (200, 422))

    # Letters, 1/0, empty and 1_0 are all refused with a 422.
    def test_bad_input(self):
        for bad in ("a", "1/0", "", "1_0"):
            self.assertEqual(post([[bad, "0"], ["0", "1"]], ["1", "1"]).status_code, 422, bad)

    # Non-square systems and sizes above 24 are refused.
    def test_not_square(self):
        self.assertEqual(post([["1", "2"]], ["1"]).status_code, 422)
        self.assertEqual(post([["1"]] * 25, ["1"] * 25).status_code, 422)


# The step-by-step working that the page shows.
class Working(unittest.TestCase):
    # The steps begin at [A | I] and finish at [I | A^-1].
    def test_exact_steps_start_with_A_I_and_end_with_I_inverse(self):
        r = post([["2", "1"], ["5", "3"]], ["3", "8"]).json()
        self.assertEqual(r["steps"][0]["M"], [["2", "1", "1", "0"], ["5", "3", "0", "1"]])
        self.assertEqual(r["steps"][-1]["M"], [["1", "0", "3", "-1"], ["0", "1", "-5", "2"]])
        self.assertEqual(r["swaps"], 0)

    # In decimal mode there is one snapshot per pivot column (plus the start).
    def test_decimal_steps_one_per_column(self):
        s = client.get("/api/random?n=12").json()
        self.assertEqual(len(post(s["a"], s["b"]).json()["steps"]), 13)


# Calculation-breaking edge cases found while debugging.
class Numerics(unittest.TestCase):
    # BUG-13: an invertible but badly scaled matrix must not be called singular.
    def test_badly_scaled_matrix_is_not_called_singular(self):  # BUG-13
        A = [[1.0 if i == j else 0.0 for j in range(11)] for i in range(11)]
        A[10][10] = 1e-13
        self.assertIsNotNone(mc7_api.float_inverse(A)[0])

    # A matrix with a row that is a combination of other rows must still be called singular.
    def test_dependent_rows_are_still_detected(self):
        A = [[float((i * 7 + j * 3) % 11 - 5 + (i == j) * 20) for j in range(12)] for i in range(12)]
        A[5] = [a + b for a, b in zip(A[1], A[2])]
        self.assertIsNone(mc7_api.float_inverse(A)[0])

    # BUG-15: shortening a gigantic fraction for display must not crash.
    def test_fmt_never_crashes_on_huge_fractions(self):  # BUG-15
        self.assertTrue(mc.fmt(F(10 ** 400, 3)))

    # BUG-20: 38-digit decimals must not produce a multi-megabyte reply.
    def test_very_long_numbers_do_not_make_a_huge_reply(self):  # BUG-20
        import random
        random.seed(1)
        A = [["1." + "".join(random.choice("0123456789") for _ in range(36)) for _ in range(10)] for _ in range(10)]
        r = post(A, ["1"] * 10).json()
        self.assertTrue(r["steps_omitted"] and len(json.dumps(r)) < 1_000_000)


# The command-line mode.
class Cli(unittest.TestCase):
    # BUG-4: a short row prints a friendly message, not a crash.
    def test_short_row_gives_message_not_traceback(self):  # BUG-4
        p = subprocess.run([sys.executable, "mc7_matrix_inverse.py", "--input"], input="2\n1 2\n",
                           capture_output=True, text=True)
        self.assertNotIn("Traceback", p.stderr)
        self.assertIn("expected 3 numbers", p.stderr)


if __name__ == "__main__":
    unittest.main()
