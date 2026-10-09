"""Regression tests. Run:  pip install httpx pytest  then  pytest test_mc7.py   (or python test_mc7.py)"""
import json, subprocess, sys, unittest
from fractions import Fraction as F
from fastapi.testclient import TestClient
import mc7_api, mc7_matrix_inverse as mc

client = TestClient(mc7_api.app)
post = lambda A, B: client.post("/api/solve", json={"A": A, "B": B})


class Solver(unittest.TestCase):
    def test_sample(self):
        r = post([["2", "1", "-1"], ["-3", "-1", "2"], ["-2", "1", "2"]], ["8", "-11", "-3"]).json()
        self.assertEqual([v[0] for v in r["x"]], ["2", "3", "-1"])
        self.assertTrue(r["ok"] and r["ver"]["okI"] and all(q["ok"] for q in r["ver"]["rows"]))

    def test_singular(self):
        self.assertIsNone(post([["1", "2"], ["2", "4"]], ["3", "6"]).json()["inv"])

    def test_random_all_sizes_verify(self):
        for n in range(1, 25):
            s = client.get(f"/api/random?n={n}").json()
            r = post(s["a"], s["b"]).json()
            self.assertTrue(r["ok"] and all(q["ok"] for q in r["ver"]["rows"]), n)

    def test_row_op_log_is_exact(self):  # BUG-5
        log = mc.gauss_jordan_inverse([[F(12345678912), F(1)], [F(1), F(1)]])[1]
        self.assertIn("12345678912", log[0])
        self.assertFalse(any("e+" in s for s in log))


class Validation(unittest.TestCase):
    def test_decimal_over_integer(self):  # BUG-1
        r = post([["1.5/3", "0"], ["0", "1"]], ["1", "1"])
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["x"][0][0], "2")

    def test_exponent_rejected_fast(self):  # BUG-2
        self.assertEqual(post([["1e9999999", "0"], ["0", "1"]], ["1", "1"]).status_code, 422)

    def test_huge_number_is_422_not_500(self):  # BUG-3
        n = 11
        A = [["1" + "0" * 39 if i == j == 0 else ("1" if i == j else "0") for j in range(n)] for i in range(n)]
        self.assertIn(post(A, ["1"] * n).status_code, (200, 422))
        A[0][0] = "9" * 40
        A[1][1] = "0." + "0" * 38 + "1"
        self.assertIn(post(A, ["1"] * n).status_code, (200, 422))

    def test_bad_input(self):
        for bad in ("a", "1/0", "", "1_0"):
            self.assertEqual(post([[bad, "0"], ["0", "1"]], ["1", "1"]).status_code, 422, bad)

    def test_not_square(self):
        self.assertEqual(post([["1", "2"]], ["1"]).status_code, 422)
        self.assertEqual(post([["1"]] * 25, ["1"] * 25).status_code, 422)


class Cli(unittest.TestCase):
    def test_short_row_gives_message_not_traceback(self):  # BUG-4
        p = subprocess.run([sys.executable, "mc7_matrix_inverse.py", "--input"], input="2\n1 2\n",
                           capture_output=True, text=True)
        self.assertNotIn("Traceback", p.stderr)
        self.assertIn("expected 3 numbers", p.stderr)


if __name__ == "__main__":
    unittest.main()
