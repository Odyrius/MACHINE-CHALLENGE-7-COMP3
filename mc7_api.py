"""FastAPI bridge: browser (mc7_interactive.html) <-> Python solver (mc7_matrix_inverse.py).

Setup:  pip install fastapi uvicorn
Run:    python mc7_api.py        then open http://127.0.0.1:8000
"""
import json
import math
import operator
import random
import re
from fractions import Fraction as F
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

import mc7_matrix_inverse as mc

SNAP_MAX = 40   # decimal systems up to this size send a matrix snapshot for every pivot column
EXACT_MAX = 10  # exact fractions up to this size, double precision above it
HERE = Path(__file__).parent

app = FastAPI(title="MC#7 Matrix Inverse API")
app.add_middleware(GZipMiddleware, minimum_size=1024)
# Restrict origins to localhost (plus 'null' in case mc7_interactive.html is opened directly via file://)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000", "http://localhost:8000", "null"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class System(BaseModel):
    A: List[List[str]]
    B: List[str]
    unlimited: bool = False


# Accepts integers, decimals, and fractions with optional signs in numerator and denominator (e.g., 3/-4)
NUM = re.compile(r"^[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:/[+-]?(?:\d+(?:\.\d+)?|\.\d+))?$")


def num(s, where):
    """Parses a string cell into an exact Fraction or raises HTTP 422."""
    t = s.strip().replace("−", "-")
    if len(t) > 40 or not NUM.match(t):
        raise HTTPException(422, f"{where} is not a number. Use values like 3, -2, 1.5 or 3/4.")
    top, _, bottom = t.partition("/")
    if bottom and F(bottom) == 0:
        raise HTTPException(422, f"{where} divides by zero.")
    return F(top) / F(bottom) if bottom else F(top)


def format_matrix_as_str(matrix):
    """Converts a 2D matrix of numbers/Fractions into strings for JSON serialization."""
    return [[str(v) for v in r] for r in matrix]


def steps_json(steps):
    """Formats saved row-operation snapshots into JSON-friendly strings."""
    return [{"t": s["t"], "M": format_matrix_as_str(s["M"]), "hi": s["hi"]} for s in steps]


def exact_payload(A, B):
    """Solves systems up to EXACT_MAX using exact rational arithmetic."""
    n = len(A)
    res = mc.solve_with_inverse(A, [[b] for b in B])
    if res["inv"] is None:
        return {
            "mode": "exact", "n": n, "inv": None, "log": res["log"],
            "det": "0", "steps": steps_json(res["steps"]),
        }

    rows = [{"l": str(l), "b": str(b), "ok": ok} for l, b, ok in res["eq"]]
    steps = steps_json(res["steps"])
    omitted = sum(len(v) for st in steps for row in st["M"] for v in row) > 300_000
    if omitted:
        steps = []

    det_val = res.get("det")
    if det_val is None:
        det_val = ((-1) ** res["swaps"]) * math.prod(res["pivots"])

    return {
        "mode": "exact", "n": n,
        "inv": format_matrix_as_str(res["inv"]),
        "left": format_matrix_as_str(res["left"]),
        "right": format_matrix_as_str(res["right"]),
        "x": format_matrix_as_str(res["x"]),
        "log": res["log"],
        "det": str(det_val),
        "ok": res["ok"],
        "steps": steps,
        "steps_omitted": omitted,
        "pivots": [str(q) for q in res["pivots"]],
        "swaps": res["swaps"],
        "ver": {"rows": rows, "okL": res["okL"], "okR": res["okR"], "okI": res["ixx"]},
    }


def fmm(P, Q):
    """Decimal matrix multiplication."""
    cols = list(zip(*Q))
    return [[sum(map(operator.mul, row, c)) for c in cols] for row in P]


def round_matrix(matrix, digits=8):
    """Rounds all entries in a 2D float matrix."""
    return [[round(v, digits) for v in r] for r in matrix]


def float_inverse(A):
    """Gauss-Jordan with partial pivoting in double precision. Returns (inverse, why, steps)."""
    n = len(A)
    M = [r[:] + [float(i == j) for j in range(n)] for i, r in enumerate(A)]
    keep = n <= SNAP_MAX

    def make_snapshot(text, highlight_row):
        return {"t": text, "M": round_matrix(M, 8) if keep else None, "hi": highlight_row}

    steps = [make_snapshot("Write [A | I]", None)]
    scale = [max(abs(r[c]) for r in A) for c in range(n)]

    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[p][c]) <= 1e-12 * scale[c]:
            return None, f"No usable pivot in column {c + 1}", steps

        M[c], M[p] = M[p], M[c]
        pv = M[c][c]
        M[c] = [v / pv for v in M[c]]
        for r in range(n):
            if r != c and M[r][c]:
                f = M[r][c]
                M[r] = [a - f * b for a, b in zip(M[r], M[c])]
        swap = f"swap R{c + 1} ↔ R{p + 1} (largest entry), " if p != c else ""
        steps.append(make_snapshot(f"Column {c + 1}: {swap}divide R{c + 1} by {pv:.4g}, then clear the rest of the column", c))

    return [r[n:] for r in M], None, steps


def max_identity_deviation(M):
    """Returns the maximum entry-wise absolute difference between M and the identity matrix."""
    return max(abs(v - (i == j)) for i, r in enumerate(M) for j, v in enumerate(r))


def norm_1(M):
    """Computes the 1-norm (maximum absolute column sum) of a square matrix."""
    n = len(M)
    return max(sum(abs(M[i][j]) for i in range(n)) for j in range(n))


def float_payload(A, B):
    """Solves systems above EXACT_MAX using double-precision floats."""
    n = len(A)
    Af = [[float(v) for v in r] for r in A]
    Bf = [float(b) for b in B]
    inv, why, steps = float_inverse(Af)
    if inv is None:
        return {"mode": "float", "n": n, "inv": None, "why": why, "steps": steps}

    left = fmm(inv, Af)
    right = fmm(Af, inv)
    x = fmm(inv, [[b] for b in Bf])
    e1 = max_identity_deviation(left)
    e2 = max_identity_deviation(right)

    rows = []
    res = 0.0
    for i in range(n):
        lhs = sum(Af[i][j] * x[j][0] for j in range(n))
        res = max(res, abs(lhs - Bf[i]))
        rows.append({"l": lhs, "b": Bf[i], "ok": abs(lhs - Bf[i]) <= 1e-8 * max(1.0, abs(Bf[i]))})

    ix = fmm(left, x)
    okI = all(abs(ix[i][0] - x[i][0]) <= 1e-6 * max(1.0, abs(x[i][0])) for i in range(n))

    return {
        "mode": "float", "n": n,
        "inv": round_matrix(inv),
        "x": round_matrix(x),
        "e1": e1, "e2": e2,
        "left": round_matrix(left) if n <= 15 else None,
        "right": round_matrix(right) if n <= 15 else None,
        "res": res,
        "cond": norm_1(Af) * norm_1(inv),
        "ok": max(e1, e2) < 1e-6,
        "steps": steps,
        "ver": {"rows": rows, "okL": e1 < 1e-6, "okR": e2 < 1e-6, "okI": okI},
    }


@app.get("/")
def index():
    """
    What it does: Serves the main interactive HTML interface at the root URL.
    Output: FileResponse streaming mc7_interactive.html.
    """
    return FileResponse(HERE / "mc7_interactive.html")


@app.get("/static/style.css")
def serve_css():
    """
    What it does: Serves the external stylesheet without exposing the project directory.
    Output: FileResponse streaming style.css.
    """
    return FileResponse(HERE / "style.css", media_type="text/css")


@app.get("/static/script.js")
def serve_js():
    """
    What it does: Serves the frontend JavaScript file.
    Output: FileResponse streaming script.js.
    """
    return FileResponse(HERE / "script.js", media_type="application/javascript")


@app.get("/api/health")
def health():
    """
    What it does: Reports server availability and configured matrix size limits.
    Output: JSON object {"ok": True, "max_n": 24, "abs_max": 200, "exact_max": 10}.
    """
    return {"ok": True, "max_n": mc.MAX_N, "abs_max": mc.ABS_MAX, "exact_max": EXACT_MAX}


@app.post("/api/solve")
def solve(s: System):
    """
    What it does: Validates the input matrix dimensions and entries, then solves Ax = B
                  using exact Fractions (n <= 10) or double-precision floats (n > 10).
    Output: JSON payload containing the inverse, verification checks, solution vector x, and steps.
    """
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
        payload_str = json.dumps(out, allow_nan=False)
        return Response(content=payload_str, media_type="application/json")
    except (OverflowError, ValueError):
        raise HTTPException(422, "These numbers are too large or too small to solve in decimal mode.")


@app.get("/api/random")
def random_system(n: Optional[int] = Query(None, ge=1), unlimited: bool = False):
    """
    What it does: Generates a random invertible n x n linear system with a known integer solution.
    Output: JSON object {"a": List[List[str]], "b": List[str]}.
    """
    limit = mc.ABS_MAX if unlimited else mc.MAX_N
    if n is not None and n > limit:
        raise HTTPException(422, f"The size must be between 1 and {limit}.")
    A, B = mc.random_system(n or random.randint(2, min(limit, 60)), limit)
    return {"a": format_matrix_as_str(A), "b": [str(r[0]) for r in B]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)