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
import random
import re
from fractions import Fraction as F
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

import mc7_matrix_inverse as mc

EXACT_MAX = 10  # exact fractions up to this size, double precision above it
HERE = Path(__file__).parent

app = FastAPI(title="MC#7 Matrix Inverse API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class System(BaseModel):
    A: List[List[str]]
    B: List[str]


NUM = re.compile(r"^[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:/(?:\d+(?:\.\d+)?|\.\d+))?$")  # same grammar as the page


def num(s, where):
    t = s.strip().replace("−", "-")
    if len(t) > 40 or not NUM.match(t):  # also blocks exponents like 1e9999999, which would exhaust memory
        raise HTTPException(422, f"{where} is not a number. Use values like 3, -2, 1.5 or 3/4.")
    top, _, bottom = t.partition("/")
    if bottom and F(bottom) == 0:
        raise HTTPException(422, f"{where} divides by zero.")
    return F(top) / F(bottom) if bottom else F(top)


def exact_payload(A, B):
    n = len(A)
    res = mc.solve_with_inverse(A, [[b] for b in B])
    if res["inv"] is None:
        return {"mode": "exact", "n": n, "inv": None, "log": res["log"], "det": "0"}
    s = lambda M: [[str(v) for v in r] for r in M]
    rows = [{"l": str(l), "b": str(b), "ok": ok} for l, b, ok in res["eq"]]
    return {"mode": "exact", "n": n, "inv": s(res["inv"]), "left": s(res["left"]), "right": s(res["right"]),
            "x": s(res["x"]), "log": res["log"], "det": str(mc.determinant(A)), "ok": res["ok"],
            "ver": {"rows": rows, "okL": res["okL"], "okR": res["okR"], "okI": res["ixx"]}}


def fmm(P, Q):
    return [[sum(P[i][k] * Q[k][j] for k in range(len(Q))) for j in range(len(Q[0]))] for i in range(len(P))]


def float_inverse(A):
    """Gauss-Jordan with partial pivoting in double precision."""
    n = len(A)
    M = [r[:] + [float(i == j) for j in range(n)] for i, r in enumerate(A)]
    scale = max(abs(v) for r in A for v in r) or 1.0
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[p][c]) <= 1e-12 * scale:
            return None, f"No usable pivot in column {c + 1}"
        M[c], M[p] = M[p], M[c]
        pv = M[c][c]
        M[c] = [v / pv for v in M[c]]
        for r in range(n):
            if r != c and M[r][c]:
                f = M[r][c]
                M[r] = [a - f * b for a, b in zip(M[r], M[c])]
    return [r[n:] for r in M], None


def float_payload(A, B):
    n = len(A)
    Af, Bf = [[float(v) for v in r] for r in A], [float(b) for b in B]
    inv, why = float_inverse(Af)
    if inv is None:
        return {"mode": "float", "n": n, "inv": None, "why": why}
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
    return {"mode": "float", "n": n, "inv": inv, "left": left, "right": right, "x": x, "e1": e1, "e2": e2,
            "res": res, "cond": n1(Af) * n1(inv), "ok": max(e1, e2) < 1e-6,
            "ver": {"rows": rows, "okL": e1 < 1e-6, "okR": e2 < 1e-6, "okI": okI}}


@app.get("/")
def index():
    return FileResponse(HERE / "mc7_interactive.html")


@app.get("/api/health")
def health():
    return {"ok": True, "max_n": mc.MAX_N, "exact_max": EXACT_MAX}


@app.post("/api/solve")
def solve(s: System):
    n = len(s.A)
    if not 1 <= n <= mc.MAX_N:
        raise HTTPException(422, f"The size must be between 1 and {mc.MAX_N}.")
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


@app.get("/api/random")
def random_system(n: Optional[int] = Query(None, ge=1, le=24)):
    A, B = mc.random_system(n or random.randint(2, mc.MAX_N))
    return {"a": [[str(v) for v in r] for r in A], "b": [str(r[0]) for r in B]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
