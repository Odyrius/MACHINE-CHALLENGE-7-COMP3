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
import random
import sys
from fractions import Fraction as F
from html import escape
from pathlib import Path

MAX_N = 24


def fmt(x):
    s = str(x)  # Fraction prints "3/2" or "3"
    return s if len(s) <= 10 else f"{float(x):.5g}"  # shorten huge fractions for display


def matmul(P, Q):
    return [[sum(P[i][k] * Q[k][j] for k in range(len(Q))) for j in range(len(Q[0]))]
            for i in range(len(P))]


def identity(n):
    return [[F(int(i == j)) for j in range(n)] for i in range(n)]


def gauss_jordan_inverse(A):
    """Row-reduce [A | I] -> [I | A^-1]. Returns (inverse or None, row-op log)."""
    n = len(A)
    M = [row[:] + identity(n)[i] for i, row in enumerate(A)]
    log = []
    for c in range(n):
        p = next((r for r in range(c, n) if M[r][c] != 0), None)
        if p is None:
            log.append(f"No pivot in column {c + 1}: A is singular.")
            return None, log
        if p != c:
            M[c], M[p] = M[p], M[c]
            log.append(f"R{c + 1} ↔ R{p + 1}")
        pv = M[c][c]
        if pv != 1:
            M[c] = [v / pv for v in M[c]]
            log.append(f"R{c + 1} → R{c + 1} ÷ {pv}")
        for r in range(n):
            if r != c and M[r][c] != 0:
                f = M[r][c]
                M[r] = [a - f * b for a, b in zip(M[r], M[c])]
                log.append(f"R{r + 1} → R{r + 1} − ({f})·R{c + 1}")
    return [row[n:] for row in M], log


def determinant(A):
    M, n, d = [row[:] for row in A], len(A), F(1)
    for c in range(n):
        p = next((r for r in range(c, n) if M[r][c] != 0), None)
        if p is None:
            return F(0)
        if p != c:
            M[c], M[p] = M[p], M[c]
            d = -d
        d *= M[c][c]
        for r in range(c + 1, n):
            k = M[r][c] / M[c][c]
            M[r] = [a - k * b for a, b in zip(M[r], M[c])]
    return d


def solve_with_inverse(A, B):
    inv, log = gauss_jordan_inverse(A)
    if inv is None:
        return {"A": A, "B": B, "inv": None, "log": log}
    n = len(A)
    I = identity(n)
    left, right = matmul(inv, A), matmul(A, inv)
    x = matmul(inv, B)
    eq = []  # substitute x into every original equation
    for i in range(n):
        lhs = sum(A[i][j] * x[j][0] for j in range(n))
        eq.append((lhs, B[i][0], lhs == B[i][0]))
    return {"A": A, "B": B, "inv": inv, "log": log, "left": left, "right": right,
            "ok": left == I and right == I, "x": x, "eq": eq,
            "okL": left == I, "okR": right == I, "ixx": matmul(left, x) == x}


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
ul.chk{list-style:none;padding:0}table.v{border-collapse:collapse}table.v th,table.v td{padding:3px 12px;text-align:right;border-bottom:1px solid #8886}.big{font-size:1.1rem}
"""


def mat(M):
    rows = "".join("<tr>" + "".join(f"<td>{escape(fmt(v))}</td>" for v in r) + "</tr>" for r in M)
    return f'<table class="m">{rows}</table>'


def op(s):
    return f"<span>{s}</span>"


def build_html(res):
    A, B, n = res["A"], res["B"], len(res["A"])
    names = [f"x<sub>{i + 1}</sub>" for i in range(n)]
    eqs = []
    def term(v, name):
        return ("− " if v < 0 else "+ ") + ("" if abs(v) == 1 else fmt(abs(v))) + name
    for r in range(n):
        t = " ".join(term(A[r][c], names[c]) for c in range(n) if A[r][c] != 0).lstrip("+ ") or "0"
        eqs.append(f"{t} = {fmt(B[r][0])}".replace("-", "−"))
    h = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
         f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
         f"<title>MC#7 Matrix Inverse</title><style>{CSS}</style></head><body>"
         "<header><h1>MC#7 — Solution of a Linear System using the Inverse</h1>"
         "<p>Computing 3 · computed in Python with exact fractions</p></header><main>"]
    h.append(
             "<section><h2>1) Matrix equation &nbsp;A x = B</h2><p>" + "<br>".join(eqs) + "</p>"
             "<div class='row'>" + mat(A) + "<table class='m'>" +
             "".join(f"<tr><td>{n_}</td></tr>" for n_ in names) + "</table><span>=</span>" + mat(B) +
             "</div></section>")
    if res["inv"] is None:
        h.append("<section><h2>2) Inverse A<sup>-1</sup></h2><p class='bad'>A is singular "
                 "(no pivot found), so A<sup>-1</sup> does not exist and this method cannot solve the system.</p></section>")
        h.append("</main></body></html>")
        return "".join(h)
    h.append("<section><h2>2) Identify the inverse A<sup>-1</sup></h2><div class='row'>"
             f"<span>A<sup>-1</sup> =</span>{mat(res['inv'])}</div>"
             "<details><summary>Gauss–Jordan row operations on [A | I]</summary><ol>" +
             "".join(f"<li>{escape(s)}</li>" for s in res["log"]) + "</ol></details></section>")
    verdict = ("<span class='ok'>✔ A<sup>-1</sup>A = AA<sup>-1</sup> = I<sub>n</sub> verified</span>" if res["ok"]
               else "<span class='bad'>✘ check failed</span>")
    h.append("<section><h2>3) Check &nbsp;A<sup>-1</sup>A = A A<sup>-1</sup> = I<sub>n</sub></h2>"
             f"<div class='row'><span>A<sup>-1</sup>A =</span>{mat(res['left'])}"
             f"<span>AA<sup>-1</sup> =</span>{mat(res['right'])}</div><p>{verdict}</p></section>")
    sol = "".join(f"<br>{names[i]} = <b>{escape(fmt(res['x'][i][0]))}</b>" for i in range(n))
    h.append("<section><h2>4) [x] = A<sup>-1</sup>B</h2><div class='row'>"
             f"{mat(res['inv'])}{mat(B)}<span>=</span>{mat(res['x'])}</div><p>Solution:{sol}</p></section>")
    items = [(True, "A is invertible: det(A) ≠ 0"),
             (all(q[2] for q in res["eq"]), "Substituting x into the equations gives Ax = B"),
             (res["okL"], "A<sup>-1</sup>A = I<sub>n</sub>"),
             (res["okR"], "AA<sup>-1</sup> = I<sub>n</sub>"),
             (res["ixx"], "I<sub>n</sub>x = x: the identity leaves the solution unchanged")]
    good = all(k for k, _ in items)
    lis = "".join(f"<li class='{'ok' if k else 'bad'}'>{'✔' if k else '✘'} {t}</li>" for k, t in items)
    trs = "".join(f"<tr><td>{i + 1}</td><td>{fmt(l)}</td><td>{fmt(b)}</td>"
                  f"<td class='{'ok' if k else 'bad'}'>{'✔' if k else '✘'}</td></tr>"
                  for i, (l, b, k) in enumerate(res["eq"]))
    verdict = ("✔ The solution is valid and the identity matrix stays the same." if good
               else "✘ Verification failed. Recheck your entries.")
    h.append(f"<section><h2>5) Verification</h2><ul class='chk'>{lis}</ul>"
             f"<p class='big {'ok' if good else 'bad'}'>{verdict}</p>"
             f"<details open><summary>Equation-by-equation substitution</summary>"
             f"<table class='v'><tr><th>Row</th><th>Σ aᵢⱼxⱼ</th><th>bᵢ</th><th></th></tr>{trs}</table></details></section>")
    h.append("</main></body></html>")
    return "".join(h)


def random_system(n):
    """Random invertible system; B = A·x for a hidden integer x.
    n <= 10: A = L·U with unit triangles (det 1, whole-number inverse). Larger: diagonally dominant."""
    n = max(1, min(n, MAX_N))
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
            A[i][i] = sum(abs(v) for v in A[i]) - abs(A[i][i]) + 1 + R(0, 5)
    return A, [[sum(A[i][j] * x[j] for j in range(n))] for i in range(n)]


def read_system():
    try:
        n = int(input(f"Number of equations/unknowns n (1-{MAX_N}): "))
    except ValueError:
        sys.exit("n must be a whole number.")
    if not 1 <= n <= MAX_N:
        sys.exit(f"n must be between 1 and {MAX_N}.")
    print(f"Enter each equation as {n} coefficients then the constant, e.g. '2 1 -1 8':")
    A, B = [], []
    for i in range(n):
        try:
            vals = [F(v) for v in input(f"  row {i + 1}: ").replace(",", " ").split()]
        except (ValueError, ZeroDivisionError):
            sys.exit(f"Row {i + 1}: use numbers like 3, -2, 1.5 or 3/4.")
        if len(vals) != n + 1:
            sys.exit(f"Row {i + 1}: expected {n + 1} numbers (the coefficients, then the constant), got {len(vals)}.")
        A.append(vals[:n])
        B.append([vals[n]])
    return A, B


if __name__ == "__main__":
    if "--input" in sys.argv:
        A, B = read_system()
    elif "--random" in sys.argv:
        size = [a for a in sys.argv[1:] if a.isdigit()]
        A, B = random_system(int(size[0]) if size else random.randint(2, MAX_N))
        print(f"Random {len(A)}x{len(A)} system")
    else:  # sample: 2x+y-z=8, -3x-y+2z=-11, -2x+y+2z=-3  ->  (2, 3, -1)
        A = [[F(2), F(1), F(-1)], [F(-3), F(-1), F(2)], [F(-2), F(1), F(2)]]
        B = [[F(8)], [F(-11)], [F(-3)]]
    res = solve_with_inverse(A, B)
    if res["inv"] is None:
        print("A is singular: no inverse exists.")
    else:
        print("A^-1 =", [[fmt(v) for v in r] for r in res["inv"]])
        print("Check A^-1 A = A A^-1 = I:", res["ok"])
        print("x =", [fmt(r[0]) for r in res["x"]])
    out = str(Path(__file__).with_name("mc7_report.html"))
    with open(out, "w", encoding="utf-8") as f:
        f.write(build_html(res))
    print("Wrote", out)
