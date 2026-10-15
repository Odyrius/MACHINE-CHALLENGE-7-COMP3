"""MC#7 - Solution of a Linear System using the Matrix Inverse.

Outputs (per the prof's instructions):
  1) Matrix equation  A x = B
  2) Identify the inverse A^-1
  3) Check  A^-1 A = A A^-1 = I_n
  4) [x] = A^-1 B

Usage:
  python mc7_matrix_inverse.py            -> runs the sample system
  python mc7_matrix_inverse.py --input    -> type your own system (size 1-24)
  python mc7_matrix_inverse.py --random   -> random size (2-24) and random content
  python mc7_matrix_inverse.py --random 8 -> random content, 8x8
Writes mc7_report.html next to this file. Exact arithmetic (fractions).
"""
import math
import random
import sys
from fractions import Fraction as F
from html import escape
from pathlib import Path

MAX_N = 24     # normal size limit
ABS_MAX = 200  # hard ceiling when the user unlocks the size limit


def fmt(x):
    """Formats a number for table display, shortening very long fractions."""
    s = str(x)
    if len(s) <= 10:
        return s
    try:
        return f"{float(x):.5g}"
    except OverflowError:
        return s[:8] + "…"


def matmul(P, Q):
    """Multiplies two exact Fraction matrices P and Q, skipping zero entries."""
    num_rows_p = len(P)
    num_cols_q = len(Q[0])
    num_inner = len(Q)
    result = [[F(0) for _ in range(num_cols_q)] for _ in range(num_rows_p)]

    for i in range(num_rows_p):
        row_p = P[i]
        row_res = result[i]
        for k in range(num_inner):
            multiplier = row_p[k]
            if multiplier == 0:
                continue
            row_q = Q[k]
            for j in range(num_cols_q):
                row_res[j] += multiplier * row_q[j]
    return result


def identity(n):
    """Returns the n x n identity matrix of exact Fractions."""
    return [[F(int(i == j)) for j in range(n)] for i in range(n)]


def gauss_jordan_steps(A):
    """Row-reduce [A | I]. Returns (inverse or None, steps, pivots, swaps)."""
    n = len(A)
    id_matrix = identity(n)
    augmented = [row[:] + id_matrix[i] for i, row in enumerate(A)]

    def make_snapshot(description, highlight_row):
        return {"t": description, "M": [r[:] for r in augmented], "hi": highlight_row}

    steps = [make_snapshot("Write [A | I]", None)]
    pivots = []
    swaps = 0

    for col in range(n):
        pivot_row = next((r for r in range(col, n) if augmented[r][col] != 0), None)
        if pivot_row is None:
            steps.append(make_snapshot(f"No pivot in column {col + 1}: A is singular.", None))
            return None, steps, pivots, swaps

        if pivot_row != col:
            augmented[col], augmented[pivot_row] = augmented[pivot_row], augmented[col]
            swaps += 1
            steps.append(make_snapshot(f"R{col + 1} ↔ R{pivot_row + 1}", col))

        pivot_val = augmented[col][col]
        pivots.append(pivot_val)

        if pivot_val != 1:
            augmented[col] = [val / pivot_val for val in augmented[col]]
            steps.append(make_snapshot(f"R{col + 1} → R{col + 1} ÷ {pivot_val}", col))

        for row in range(n):
            if row != col and augmented[row][col] != 0:
                factor = augmented[row][col]
                augmented[row] = [
                    target_val - factor * pivot_col_val
                    for target_val, pivot_col_val in zip(augmented[row], augmented[col])
                ]
                steps.append(make_snapshot(f"R{row + 1} → R{row + 1} − ({factor})·R{col + 1}", row))

    return [row[n:] for row in augmented], steps, pivots, swaps


def gauss_jordan_inverse(A):
    """Returns (inverse or None, list of row-operation texts)."""
    inv, steps, _, _ = gauss_jordan_steps(A)
    return inv, [s["t"] for s in steps[1:]]


def determinant(A):
    """Computes det(A) via Gaussian elimination."""
    M = [row[:] for row in A]
    n = len(A)
    det = F(1)
    for col in range(n):
        pivot_row = next((r for r in range(col, n) if M[r][col] != 0), None)
        if pivot_row is None:
            return F(0)
        if pivot_row != col:
            M[col], M[pivot_row] = M[pivot_row], M[col]
            det = -det
        det *= M[col][col]
        for row in range(col + 1, n):
            factor = M[row][col] / M[col][col]
            M[row] = [a - factor * b for a, b in zip(M[row], M[col])]
    return det


def solve_with_inverse(A, B):
    """Solves A x = B via A^-1 and verifies all matrix identities."""
    inv, steps, pivots, swaps = gauss_jordan_steps(A)
    log = [s["t"] for s in steps[1:]]
    if inv is None:
        return {"A": A, "B": B, "inv": None, "log": log, "steps": steps, "det": F(0)}

    n = len(A)
    id_matrix = identity(n)
    left = matmul(inv, A)
    right = matmul(A, inv)
    x = matmul(inv, B)
    det = ((-1) ** swaps) * math.prod(pivots)

    eq = []
    for i in range(n):
        lhs = sum(A[i][j] * x[j][0] for j in range(n))
        eq.append((lhs, B[i][0], lhs == B[i][0]))

    return {
        "A": A, "B": B, "inv": inv, "log": log, "left": left, "right": right,
        "ok": left == id_matrix and right == id_matrix, "x": x, "eq": eq,
        "okL": left == id_matrix, "okR": right == id_matrix,
        "ixx": matmul(left, x) == x, "steps": steps, "pivots": pivots,
        "swaps": swaps, "det": det,
    }


# ---------- HTML rendering ----------
CSS = """
:root{--bg:#fff;--fg:#1c2433;--mut:#5b6678;--acc:#1c3461;--card:#f3f6fb;--ok:#18794e;--bad:#b42318}
@media (prefers-color-scheme:dark){:root{--bg:#10151f;--fg:#e6ebf5;--mut:#9aa6bb;--acc:#7fa8ee;--card:#19212f;--ok:#4ade80;--bad:#f87171}}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif}
header{background:#1c3461;color:#fff;padding:20px 5vw}header h1{margin:0;font-size:1.4rem}
header p{margin:4px 0 0;opacity:.8}
main{max-width:900px;margin:auto;padding:16px 5vw 40px}
section{background:var(--card);border-radius:10px;padding:14px 18px;margin:16px 0}
h2{margin:0 0 8px;font-size:1.05rem;color:var(--acc)}
.row{display:flex;flex-wrap:wrap;align-items:center;gap:10px;overflow-x:auto}
table.m{border-collapse:collapse;border-left:2px solid var(--fg);border-right:2px solid var(--fg);border-radius:6px}
table.m td{padding:3px 10px;text-align:right;font-family:ui-monospace,monospace}
.ok{color:var(--ok);font-weight:600}.bad{color:var(--bad);font-weight:600}
details{color:var(--mut)}code{font-family:ui-monospace,monospace}
td.dv{border-left:2px solid var(--acc)}ul.wl{list-style:none;padding:0;font:.85rem/1.7 ui-monospace,monospace;max-height:300px;overflow:auto;white-space:nowrap}tr.hl td{background:#8883}ul.chk{list-style:none;padding:0}table.v{border-collapse:collapse}table.v th,table.v td{padding:3px 12px;text-align:right;border-bottom:1px solid #8886}.big{font-size:1.1rem}
"""


def dots(P, Q):
    """Entry-by-entry work for the product P x Q: row i of P times column j of Q."""
    m = len(Q)
    return "".join(
        f"<li>({i + 1},{j + 1}) = " + " + ".join(f"({fmt(P[i][k])})({fmt(Q[k][j])})" for k in range(m)) +
        f" = <b>{fmt(sum(P[i][k] * Q[k][j] for k in range(m)))}</b></li>"
        for i in range(len(P)) for j in range(len(Q[0]))
    )


def aug(M, n, hi):
    """Draws a snapshot of [A | I] as an HTML table."""
    rows = "".join(
        "<tr" + (" class='hl'" if i == hi else "") + ">" +
        "".join(f"<td{' class=dv' if j == n else ''}>{escape(fmt(v))}</td>" for j, v in enumerate(r)) +
        "</tr>" for i, r in enumerate(M)
    )
    return f'<table class="m">{rows}</table>'


def mat(M):
    """Draws a matrix as an HTML table with bracket-style side borders."""
    rows = "".join("<tr>" + "".join(f"<td>{escape(fmt(v))}</td>" for v in r) + "</tr>" for r in M)
    return f'<table class="m">{rows}</table>'


def build_html(res):
    """Builds the static HTML report from the dictionary returned by solve_with_inverse."""
    A, B, n = res["A"], res["B"], len(res["A"])
    names = list("xyzw"[:n]) if n <= 4 else [f"x<sub>{i + 1}</sub>" for i in range(n)]
    eqs = []

    def term(v, name):
        return ("− " if v < 0 else "+ ") + ("" if abs(v) == 1 else fmt(abs(v))) + name

    for r in range(n):
        t = " ".join(term(A[r][c], names[c]) for c in range(n) if A[r][c] != 0).lstrip("+ ") or "0"
        eqs.append(f"{t} = {fmt(B[r][0])}".replace("-", "−"))

    h = [
        f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>MC#7 Matrix Inverse</title><style>{CSS}</style></head><body>"
        "<header><h1>MC#7 — Solution of a Linear System using the Inverse</h1>"
        "<p>Computing 3 · computed in Python with exact fractions</p></header><main>"
    ]
    h.append(
        "<section><h2>1) Matrix equation &nbsp;A x = B</h2><p>" + "<br>".join(eqs) + "</p>"
        "<div class='row'>" + mat(A) + "<table class='m'>" +
        "".join(f"<tr><td>{n_}</td></tr>" for n_ in names) + "</table><span>=</span>" + mat(B) +
        "</div></section>"
    )
    if res["inv"] is None:
        h.append(
            "<section><h2>2) Inverse A<sup>-1</sup></h2><p class='bad'>A is singular "
            "(no pivot found), so A<sup>-1</sup> does not exist and this method cannot solve the system.</p></section>"
        )
        h.append("</main></body></html>")
        return "".join(h)

    h.append(
        "<section><h2>2) Identify the inverse A<sup>-1</sup></h2><div class='row'>"
        f"<span>A<sup>-1</sup> =</span>{mat(res['inv'])}</div>"
        "<details open><summary>Full working: every row operation on [A | I]</summary>" +
        "".join(
            f"<p><b>{str(i) + '.' if i else 'Start.'}</b> {escape(s['t'])}</p>{aug(s['M'], n, s['hi'])}"
            for i, s in enumerate(res["steps"])
        ) + "</details></section>"
    )
    verdict = (
        "<span class='ok'>✔ A<sup>-1</sup>A = AA<sup>-1</sup> = I<sub>n</sub> verified</span>"
        if res["ok"] else "<span class='bad'>✘ check failed</span>"
    )
    h.append(
        "<section><h2>3) Check &nbsp;A<sup>-1</sup>A = A A<sup>-1</sup> = I<sub>n</sub></h2>"
        f"<div class='row'><span>A<sup>-1</sup></span>{mat(res['inv'])}<span>× A</span>{mat(A)}<span>=</span>{mat(res['left'])}</div>"
        f"<div class='row'><span>A</span>{mat(A)}<span>× A<sup>-1</sup></span>{mat(res['inv'])}<span>=</span>{mat(res['right'])}</div>"
        f"<p>{verdict}</p>"
        f"<details><summary>Show the multiplication work, entry by entry</summary>"
        f"<p>A<sup>-1</sup>A</p><ul class='wl'>{dots(res['inv'], A)}</ul>"
        f"<p>AA<sup>-1</sup></p><ul class='wl'>{dots(A, res['inv'])}</ul></details></section>"
    )
    sol = "".join(f"<br>{names[i]} = <b>{escape(fmt(res['x'][i][0]))}</b>" for i in range(n))
    h.append(
        "<section><h2>4) [X] = A<sup>-1</sup>B</h2><div class='row'>"
        f"{mat(res['inv'])}{mat(B)}<span>=</span>{mat(res['x'])}</div><p>Solution:{sol}</p></section>"
    )
    items = [
        (True, "A is invertible: det(A) ≠ 0"),
        (all(q[2] for q in res["eq"]), "Substituting x into the equations gives Ax = B"),
        (res["okL"], "A<sup>-1</sup>A = I<sub>n</sub>"),
        (res["okR"], "AA<sup>-1</sup> = I<sub>n</sub>"),
        (res["ixx"], "I<sub>n</sub>x = x: the identity leaves the solution unchanged"),
    ]
    good = all(k for k, _ in items)
    lis = "".join(f"<li class='{'ok' if k else 'bad'}'>{'✔' if k else '✘'} {t}</li>" for k, t in items)
    trs = "".join(
        f"<tr><td>{i + 1}</td><td>{fmt(l)}</td><td>{fmt(b)}</td>"
        f"<td class='{'ok' if k else 'bad'}'>{'✔' if k else '✘'}</td></tr>"
        for i, (l, b, k) in enumerate(res["eq"])
    )
    verdict_msg = (
        "✔ The solution is valid and the identity matrix stays the same."
        if good else "✘ Verification failed. Recheck your entries."
    )
    h.append(
        f"<section><h2>5) Verification</h2><ul class='chk'>{lis}</ul>"
        f"<p class='big {'ok' if good else 'bad'}'>{verdict_msg}</p>"
        f"<details open><summary>Equation-by-equation substitution</summary>"
        f"<table class='v'><tr><th>Row</th><th>Σ aᵢⱼxⱼ</th><th>bᵢ</th><th></th></tr>{trs}</table></details></section>"
    )
    h.append("</main></body></html>")
    return "".join(h)


def random_system(n, limit=MAX_N):
    """Random invertible system; B = A·x for a hidden integer x."""
    n = max(1, min(n, limit))
    R = random.randint
    x = [R(-9, 9) for _ in range(n)]
    if n <= 10:
        L = [[F(int(i == j)) for j in range(n)] for i in range(n)]
        U = [row[:] for row in L]
        for i in range(n):
            for j in range(i):
                L[i][j], U[j][i] = F(R(-1, 1)), F(R(-1, 1))
        A = matmul(L, U)
    else:
        A = [[F(R(-9, 9)) for _ in range(n)] for _ in range(n)]
        for i in range(n):
            A[i][i] = sum(abs(v) for v in A[i]) - abs(A[i][i]) + 1 + R