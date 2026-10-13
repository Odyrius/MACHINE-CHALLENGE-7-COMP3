"""FastAPI bridge: browser (mc7_interactive.html)  <->  Python solver (mc7_matrix_inverse.py).

Setup:  pip install fastapi uvicorn
Run:    python mc7_api.py        then open http://127.0.0.1:8000

Endpoints
  GET  /              the web page
  GET  /api/health    server status and limits
  POST /api/solve     {"A": [["2","1"],["5","3"]], "B": ["3","8"]} -> inverse, check, solution, verification
  GET  /api/random    ?n=8 (optional) -> random invertible system
The page falls back to its built-in JavaScript solver if this server is not running.
"""
import json
import operator
import random
import re
from fractions import Fraction as F
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import mc7_matrix_inverse as mc

SNAP_MAX = 40   # decimal systems up to this size also send a matrix snapshot for every pivot column
EXACT_MAX = 10  # exact fractions up to this size, double precision above it
HERE = Path(__file__).parent

app = FastAPI(title="MC#7 Matrix Inverse API")
app.add_middleware(GZipMiddleware, minimum_size=1024)  # the step snapshots compress very well
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=str(HERE)), name="static")

# System: the shape of the request body. A = rows of coefficients, B = the constants.
# Everything is TEXT so that 3/4 and 1.5 arrive exactly as the person typed them.
class System(BaseModel):
    A: List[List[str]]
    B: List[str]
    unlimited: bool = False  # True only after the person ticked "Remove the size limit" and accepted the warning


NUM = re.compile(r"^[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:/(?:\d+(?:\.\d+)?|\.\d+))?$")  # same grammar as the page


# num(s, where): turns one typed entry into an exact fraction.
# Accepts 3, -2, 1.5, .5, 3/4 and 1.5/3 (the same rules as the page).
# Rejects anything else (letters, exponents like 1e9, over 40 characters, divide by zero)
# with a message that names the cell, e.g. "Row 2, column 3 is not a number".
def num(s, where):
    t = s.strip().replace("−", "-")
    if len(t) > 40 or not NUM.match(t):  # also blocks exponents like 1e9999999, which would exhaust memory
        raise HTTPException(422, f"{where} is not a number. Use values like 3, -2, 1.5 or 3/4.")
    top, _, bottom = t.partition("/")
    if bottom and F(bottom) == 0:
        raise HTTPException(422, f"{where} divides by zero.")
    return F(top) / F(bottom) if bottom else F(top)


# steps_json(steps): turns the saved row-operation snapshots into JSON-friendly text (fractions become "3/4").
def steps_json(steps):
    return [{"t": s["t"], "M": [[str(v) for v in r] for r in s["M"]], "hi": s["hi"]} for s in steps]


# exact_payload(A, B): the answer for sizes up to EXACT_MAX, using exact fractions via the solver.
# Sends the inverse, both check products, x, the determinant, pivots, row-operation snapshots
# and the verification, all as text. Very long numbers skip the snapshots to keep the reply small.
def exact_payload(A, B):
    n = len(A)
    res = mc.solve_with_inverse(A, [[b] for b in B])
    if res["inv"] is None:
        return {"mode": "exact", "n": n, "inv": None, "log": res["log"], "det": "0",
                "steps": steps_json(res["steps"])}
    s = lambda M: [[str(v) for v in r] for r in M]
    rows = [{"l": str(l), "b": str(b), "ok": ok} for l, b, ok in res["eq"]]
    steps = steps_json(res["steps"])
    omitted = sum(len(v) for st in steps for row in st["M"] for v in row) > 300_000
    if omitted:  # very long numbers would make a multi-megabyte reply on every keystroke
        steps = []
    return {"mode": "exact", "n": n, "inv": s(res["inv"]), "left": s(res["left"]), "right": s(res["right"]),
            "x": s(res["x"]), "log": res["log"], "det": str(mc.determinant(A)), "ok": res["ok"],
            "steps": steps, "steps_omitted": omitted, "pivots": [str(q) for q in res["pivots"]], "swaps": res["swaps"],
            "ver": {"rows": rows, "okL": res["okL"], "okR": res["okR"], "okI": res["ixx"]}}


# fmm(P, Q): matrix multiplication with ordinary decimals (the decimal twin of matmul).
def fmm(P, Q):
    cols = list(zip(*Q))  # work with columns of Q so each entry is one fast dot product
    return [[sum(map(operator.mul, row, c)) for c in cols] for row in P]


# float_inverse(A): Gauss-Jordan in decimals, for sizes above EXACT_MAX.
# Uses PARTIAL PIVOTING: at each column it picks the row with the largest |entry|, which keeps rounding errors small.
# Singular test: the pivot is smaller than 1e-12 x the size of its original column.
# Returns (inverse or None, reason, snapshots).
def float_inverse(A):
    """Gauss-Jordan with partial pivoting in double precision. Returns (inverse, why, steps)."""
    n = len(A)
    M = [r[:] + [float(i == j) for j in range(n)] for i, r in enumerate(A)]
    keep = n <= SNAP_MAX  # above this a snapshot of every column would hold millions of numbers: keep only the text
    snap = lambda t, hi: {"t": t, "M": [[round(v, 8) for v in r] for r in M] if keep else None, "hi": hi}
    steps = [snap("Write [A | I]", None)]
    scale = [max(abs(r[c]) for r in A) for c in range(n)]  # size of each ORIGINAL column
    for c in range(n):
        # partial pivoting: pick the row with the largest |entry| in this column
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[p][c]) <= 1e-12 * scale[c]:
            return None, f"No usable pivot in column {c + 1}", steps
        # swap it onto the diagonal, then divide by the pivot so the pivot becomes 1
        M[c], M[p] = M[p], M[c]
        pv = M[c][c]
        M[c] = [v / pv for v in M[c]]
        for r in range(n):
            if r != c and M[r][c]:
                f = M[r][c]
                # subtract f x (pivot row) so this row's entry in the pivot column becomes 0
                M[r] = [a - f * b for a, b in zip(M[r], M[c])]
        swap = f"swap R{c + 1} ↔ R{p + 1} (largest entry), " if p != c else ""
        steps.append(snap(f"Column {c + 1}: {swap}divide R{c + 1} by {pv:.4g}, then clear the rest of the column", c))
    return [r[n:] for r in M], None, steps


# float_payload(A, B): the answer for sizes above EXACT_MAX, in decimals.
# Besides the inverse and x it measures how trustworthy the result is:
#   e1, e2 = how far A^-1 A and A A^-1 are from I      res = largest error of A x - B
#   cond   = condition number (how sensitive the answer is; above 1e8 means unreliable).
def float_payload(A, B):
    n = len(A)
    Af, Bf = [[float(v) for v in r] for r in A], [float(b) for b in B]
    inv, why, steps = float_inverse(Af)
    if inv is None:
        return {"mode": "float", "n": n, "inv": None, "why": why, "steps": steps}
    left, right, x = fmm(inv, Af), fmm(Af, inv), fmm(inv, [[b] for b in Bf])
    dev = lambda M: max(abs(v - (i == j)) for i, r in enumerate(M) for j, v in enumerate(r))
    e1, e2 = dev(left), dev(right)
    rows, res = [], 0.0
    for i in range(n):
        l = sum(Af[i][j] * x[j][0] for j in range(n))
        res = max(res, abs(l - Bf[i]))
        rows.append({"l": l, "b": Bf[i], "ok": abs(l - Bf[i]) <= 1e-8 * max(1, abs(Bf[i]))})
    ix = fmm(left, x)
    okI = all(abs(ix[i][0] - x[i][0]) <= 1e-6 * max(1, abs(x[i][0])) for i in range(n))
    n1 = lambda M: max(sum(abs(M[i][j]) for i in range(n)) for j in range(n))
    rnd = lambda M: [[round(v, 8) for v in r] for r in M]  # the page shows 4 decimals, so 8 is plenty
    return {"mode": "float", "n": n, "inv": rnd(inv), "x": rnd(x), "e1": e1, "e2": e2,
            "left": rnd(left) if n <= 15 else None, "right": rnd(right) if n <= 15 else None,  # the page only draws them up to 15x15
            "res": res, "cond": n1(Af) * n1(inv), "ok": max(e1, e2) < 1e-6, "steps": steps,
            "ver": {"rows": rows, "okL": e1 < 1e-6, "okR": e2 < 1e-6, "okI": okI}}


# GET /  ->  serves the web page.
@app.get("/")
def index():
    return FileResponse(HERE / "mc7_interactive.html")


# GET /api/health  ->  lets the page (and you) check that the server is alive and see the limits.
@app.get("/api/health")
def health():
    return {"ok": True, "max_n": mc.MAX_N, "abs_max": mc.ABS_MAX, "exact_max": EXACT_MAX}


# POST /api/solve  ->  the main endpoint. Checks the size and shape, converts every entry with num(),
# then answers with exact fractions (n <= EXACT_MAX) or decimals (bigger).
# Sizes above MAX_N (24) are only accepted when the request says "unlimited": true; the hard ceiling is ABS_MAX (200).
@app.post("/api/solve")
def solve(s: System):
    n = len(s.A)
    limit = mc.ABS_MAX if s.unlimited else mc.MAX_N
    if not 1 <= n <= limit:
        if n > mc.MAX_N and not s.unlimited:
            raise HTTPException(422, f"Systems above {mc.MAX_N}x{mc.MAX_N} need the 'Remove the size limit' option.")
        raise HTTPException(422, f"The size must be between 1 and {limit}.")
    if any(len(r) != n for r in s.A) or len(s.B) != n:
        raise HTTPException(422, "The system must be square: n equations and n unknowns.")
    A = [[num(v, f"Row {i + 1}, column {j + 1}") for j, v in enumerate(r)] for i, r in enumerate(s.A)]
    B = [num(v, f"Row {i + 1}, constant") for i, v in enumerate(s.B)]
    if n <= EXACT_MAX:
        return exact_payload(A, B)
    try:
        out = float_payload(A, B)
        json.dumps(out, allow_nan=False)  # inf/nan would otherwise crash the response
        return out
    except (OverflowError, ValueError):
        raise HTTPException(422, "These numbers are too large or too small to solve in decimal mode.")


# GET /api/random?n=8&unlimited=true  ->  a random invertible system (random size 2..24, or 2..60 when unlimited, if n is left out).
@app.get("/api/random")
def random_system(n: Optional[int] = Query(None, ge=1), unlimited: bool = False):
    limit = mc.ABS_MAX if unlimited else mc.MAX_N
    if n is not None and n > limit:
        raise HTTPException(422, f"The size must be between 1 and {limit}.")
    A, B = mc.random_system(n or random.randint(2, min(limit, 60)), limit)  # random sizes stay up to 60 so they solve quickly
    return {"a": [[str(v) for v in r] for r in A], "b": [str(r[0]) for r in B]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
