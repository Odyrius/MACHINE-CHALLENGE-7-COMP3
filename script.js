// Configuration constants for matrix size limits and rendering thresholds
const MAX_N = 24, ABS_MAX = 200, SNAP_MAX = 40, GRID_MAX = 60, EXACT_MAX = 10, EQ_STYLE = 4;
let UNLOCK = false;

/**
 * Checks the current maximum allowed matrix dimension n.
 * Output: Number (200 if unlocked, 24 otherwise).
 */
const LIMIT = () => UNLOCK ? ABS_MAX : MAX_N;

/* ---------- Exact Rationals (BigInt) for n <= EXACT_MAX ---------- */

/**
 * Computes the absolute value of a BigInt.
 * Output: Positive BigInt.
 */
const babs = a => a < 0n ? -a : a;

/**
 * Computes the greatest common divisor of two BigInts using Euclid's algorithm.
 * Output: Positive BigInt representing the GCD.
 */
const gcd = (a, b) => {
  a = babs(a);
  b = babs(b);
  while (b) { [a, b] = [b, a % b]; }
  return a;
};

/**
 * Creates a simplified fraction object {n: numerator, d: denominator} using BigInts.
 * Output: Object {n: BigInt, d: BigInt} in lowest terms with d > 0.
 */
const fr = (n, d = 1n) => {
  n = BigInt(n);
  d = BigInt(d);
  if (d < 0n) { n = -n; d = -d; }
  const g = gcd(n, d) || 1n;
  return { n: n / g, d: d / g };
};

/**
 * Arithmetic operations (add, subtract, multiply, divide) on two fraction objects.
 * Output: A new simplified fraction object {n: BigInt, d: BigInt}.
 */
const add = (a, b) => fr(a.n * b.d + b.n * a.d, a.d * b.d);
const sub = (a, b) => fr(a.n * b.d - b.n * a.d, a.d * b.d);
const mul = (a, b) => fr(a.n * b.n, a.d * b.d);
const div = (a, b) => fr(a.n * b.d, a.d * b.n);

/**
 * Checks if a fraction object equals 0 or 1.
 * Output: Boolean (true or false).
 */
const isZ = a => a.n === 0n;
const isOne = a => a.n === 1n && a.d === 1n;
const ZERO = fr(0), ONE = fr(1);

/**
 * Formats a fraction object into a display string (e.g., "3/4" or "5").
 * Output: String with typographic minus sign.
 */
const str = a => (a.d === 1n ? a.n.toString() : a.n + '/' + a.d).replaceAll('-', '−');

/**
 * Converts a fraction into a 4-decimal-place approximation string if it is not a whole number.
 * Output: String (e.g., "≈ 0.7500" or "" for integers).
 */
const dec = a => {
  const q = Number(a.n) / Number(a.d);
  return a.d === 1n || !isFinite(q) ? '' : '≈ ' + q.toFixed(4).replaceAll('-', '−');
};

// Regex matching valid integers, decimals, and signed fractions (e.g., 3, -2, 1.5, 3/-4)
const NUM = /^([+-]?\d*\.?\d+)(?:\/([+-]?\d*\.?\d+))?$/;

/**
 * Trims whitespace, normalizes minus signs, and enforces the 40-character limit.
 * Output: Cleaned String (or "" if over 40 chars).
 */
const clean = s => {
  const t = s.trim().replaceAll('−', '-');
  return t.length > 40 ? '' : t;
};

/**
 * Parses a user-typed string into an exact BigInt fraction object.
 * Output: Fraction object {n, d}, or null if invalid or dividing by zero.
 */
function parseX(s) {
  const m = clean(s).match(NUM);
  if (!m) return null;
  const toFrac = x => {
    const neg = x[0] === '-';
    const u = x.replace(/^[+-]/, '');
    const k = (u.split('.')[1] || '').length;
    const i = BigInt(u.replace('.', ''));
    return fr(neg ? -i : i, 10n ** BigInt(k));
  };
  const a = toFrac(m[1]);
  if (m[2] === undefined) return a;
  const b = toFrac(m[2]);
  return isZ(b) ? null : div(a, b);
}

/**
 * Parses a user-typed string into a standard IEEE-754 floating-point number.
 * Output: Number, or null if invalid or dividing by zero.
 */
function parseF(s) {
  const m = clean(s).match(NUM);
  if (!m) return null;
  const a = Number(m[1]);
  if (m[2] === undefined) return a;
  const b = Number(m[2]);
  return b === 0 ? null : a / b;
}

/* ---------- Matrix Helpers ---------- */

/**
 * Builds an n x n identity matrix of exact Fraction objects.
 * Output: 2D Array of fraction objects.
 */
const eye = n => Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => i === j ? ONE : ZERO));

/**
 * Multiplies two 2D matrices of exact Fraction objects (P * Q).
 * Output: 2D Array of fraction objects representing the product matrix.
 */
const matmul = (P, Q) => P.map(r => Q[0].map((_, j) => r.reduce((s, v, k) => isZ(v) ? s : add(s, mul(v, Q[k][j])), ZERO)));

/**
 * Compares two exact Fraction matrices for entry-by-entry equality.
 * Output: Boolean (true if identical, false otherwise).
 */
const same = (P, Q) => P.every((r, i) => r.every((v, j) => v.n === Q[i][j].n && v.d === Q[i][j].d));

/* ---------- Built-in Solvers (Offline Fallback) ---------- */

/**
 * Solves Ax = B using exact Gauss-Jordan elimination on [A | I].
 * Output: Object containing mode, n, A, B, inv, log, det, left, right, ok, x, steps, pivots, and swaps.
 */
function solveExact(A, B) {
  const n = A.length, I = eye(n), log = [];
  let M = A.map((r, i) => [...r, ...I[i]]), det = ONE;
  const steps = [], pivots = [];
  let swaps = 0;

  const note = (t, hi) => {
    log.push(t);
    steps.push({ t, M: M.map(r => r.slice()), hi });
  };
  steps.push({ t: 'Write [A | I]', M: M.map(r => r.slice()), hi: null });

  for (let c = 0; c < n; c++) {
    let p = -1;
    for (let r = c; r < n; r++) {
      if (!isZ(M[r][c])) { p = r; break; }
    }
    if (p < 0) {
      note(`No pivot in column ${c + 1}: A is singular.`, null);
      return { mode: 'exact', n, A, B, inv: null, log, det: ZERO, steps };
    }
    if (p !== c) {
      [M[c], M[p]] = [M[p], M[c]];
      det = mul(det, fr(-1));
      swaps++;
      note(`R${c + 1} <-> R${p + 1}`, c);
    }
    const pv = M[c][c];
    det = mul(det, pv);
    pivots.push(pv);
    if (!isOne(pv)) {
      M[c] = M[c].map(v => div(v, pv));
      note(`R${c + 1} -> R${c + 1} / ${str(pv)}`, c);
    }
    for (let r = 0; r < n; r++) {
      if (r !== c && !isZ(M[r][c])) {
        const f = M[r][c];
        M[r] = M[r].map((v, j) => sub(v, mul(f, M[c][j])));
        note(`R${r + 1} -> R${r + 1} - (${str(f)})*R${c + 1}`, r);
      }
    }
  }
  const inv = M.map(r => r.slice(n)), left = matmul(inv, A), right = matmul(A, inv);
  return {
    mode: 'exact', n, A, B, inv, log, det, left, right,
    ok: same(left, I) && same(right, I),
    x: matmul(inv, B.map(v => [v])), steps, pivots, swaps
  };
}

/**
 * Multiplies two 2D floating-point matrices (P * Q), skipping zeros.
 * Output: 2D Array of Numbers.
 */
function matmulF(P, Q) {
  const n = P.length, m = Q[0].length, k = Q.length;
  const R = Array.from({ length: n }, () => new Array(m).fill(0));
  for (let i = 0; i < n; i++) {
    const Ri = R[i], Pi = P[i];
    for (let t = 0; t < k; t++) {
      const p = Pi[t];
      if (p === 0) continue;
      const Qt = Q[t];
      for (let j = 0; j < m; j++) Ri[j] += p * Qt[j];
    }
  }
  return R;
}

/**
 * Finds the maximum absolute deviation of a matrix M from the identity matrix I.
 * Output: Number (>= 0).
 */
const dev = M => {
  let e = 0;
  M.forEach((r, i) => r.forEach((v, j) => { e = Math.max(e, Math.abs(v - (i === j ? 1 : 0))); }));
  return e;
};

/**
 * Computes the 1-norm (maximum column sum of absolute values) of a matrix M.
 * Output: Number.
 */
const norm1 = M => {
  let b = 0;
  for (let j = 0; j < M.length; j++) {
    let s = 0;
    for (let i = 0; i < M.length; i++) s += Math.abs(M[i][j]);
    b = Math.max(b, s);
  }
  return b;
};

/**
 * Solves Ax = B in double-precision floats using Gauss-Jordan elimination with partial pivoting.
 * Output: Object containing mode, n, A, B, inv, left, right, x, e1, e2, res, cond, ok, and steps.
 */
function solveFloat(A, B) {
  const n = A.length;
  const scale = A[0].map((_, c) => Math.max(...A.map(r => Math.abs(r[c]))));
  const M = A.map((r, i) => {
    const row = r.concat(new Array(n).fill(0));
    row[n + i] = 1;
    return row;
  });
  const keep = n <= SNAP_MAX;
  const steps = [{ t: 'Write [A | I]', M: keep ? M.map(r => r.slice()) : null, hi: null }];

  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) {
      if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    }
    if (Math.abs(M[p][c]) <= 1e-12 * scale[c]) {
      return { steps, mode: 'float', n, A, B, inv: null, why: `No usable pivot in column ${c + 1}` };
    }
    [M[c], M[p]] = [M[p], M[c]];
    const Mc = M[c], pv = Mc[c];
    for (let j = 0; j < 2 * n; j++) Mc[j] /= pv;
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = M[r][c];
      if (f === 0) continue;
      const Mr = M[r];
      for (let j = 0; j < 2 * n; j++) Mr[j] -= f * Mc[j];
    }
    steps.push({
      t: `Column ${c + 1}: ${p !== c ? `swap R${c + 1} <-> R${p + 1} (largest entry), ` : ''}divide R${c + 1} by ${fmtF(pv)}, then clear the rest of the column`,
      M: keep ? M.map(r => r.slice()) : null,
      hi: c
    });
  }
  const inv = M.map(r => r.slice(n)), left = matmulF(inv, A), right = matmulF(A, inv), x = matmulF(inv, B.map(v => [v]));
  let res = 0;
  matmulF(A, x).forEach((r, i) => { res = Math.max(res, Math.abs(r[0] - B[i])); });
  const e1 = dev(left), e2 = dev(right);
  return { mode: 'float', n, A, B, inv, left, right, x, e1, e2, res, cond: norm1(A) * norm1(inv), ok: Math.max(e1, e2) < 1e-6, steps };
}

/* ---------- HTML Rendering Helpers ---------- */

const subs = '₀₁₂₃₄₅₆₇₈₉';

/**
 * Generates the variable name for unknown index i (x, y, z, w for n <= 4; subscripted x₁, x₂... for n > 4).
 * Output: String representing the variable name.
 */
const xn = (i, n = N) => n <= 4 ? 'xyzw'[i] : 'x' + String(i + 1).replace(/\d/g, d => subs[d]);

/**
 * Formats a floating-point number to 4 decimal places or scientific notation.
 * Output: Formatted String with typographic minus signs.
 */
const fmtF = v => {
  const a = Math.abs(v);
  if (a < 1e-10) return '0';
  return ((a >= 1e5 || a < 1e-3) ? v.toExponential(2) : String(+v.toFixed(4))).replaceAll('-', '−');
};

/**
 * Renders a 2D array M into a bracketed HTML table string using formatter function f.
 * Output: HTML String (<div class="box"><table class="m">...</table></div>).
 */
const mat = (M, f) => `<div class="box"><table class="m" role="presentation">${M.map(r => '<tr>' + r.map(v => `<td>${typeof v === 'string' ? v : f(v)}</td>`).join('') + '</tr>').join('')}</table></div>`;

/**
 * Wraps text in a styled span label.
 * Output: HTML String (<span class="lab">...</span>).
 */
const lab = s => `<span class="lab">${s}</span>`;

/**
 * Formats an error metric in 1-decimal scientific notation.
 * Output: String (e.g., "1.2e−16" or "0").
 */
const sci = v => v === 0 ? '0' : v.toExponential(1).replaceAll('-', '−');

/**
 * Renders an augmented matrix [A | I] snapshot as an HTML table with a vertical divider and row highlight.
 * Output: HTML String.
 */
const aug = (M, f, n, hi) => `<div class="box"><table class="m" role="presentation">${M.map((r, i) => `<tr${i === hi ? ' class="hl"' : ''}>${r.map((v, j) => `<td${j === n ? ' class="dv"' : ''}>${f(v)}</td>`).join('')}</tr>`).join('')}</table></div>`;

let LZ = {}, lzId = 0;

/**
 * Creates a collapsible <details> element whose inner HTML is only built when the user clicks to open it.
 * Output: HTML String (<details data-lz="...">...</details>).
 */
const lazy = (summary, build, open, cls, style) => {
  const id = 'lz' + (lzId++);
  LZ[id] = build;
  return `<details data-lz="${id}"${open ? ' open' : ''}><summary>${summary}</summary><div class="${cls || ''}" id="${id}"${style ? ` style="${style}"` : ''}></div></details>`;
};

/**
 * Populates a lazy <details> container when opened, chunking large arrays across animation frames to prevent UI freezes.
 * Output: None (mutates DOM inside the opened <details> element).
 */
function fillLazy(d) {
  const id = d.dataset.lz, b = LZ[id];
  if (!b || !d.open) return;
  delete LZ[id];
  const el = document.getElementById(id), out = b();
  if (!Array.isArray(out)) {
    el.innerHTML = out;
    return;
  }
  let i = 0;
  const next = () => {
    if (!el.isConnected) return;
    el.insertAdjacentHTML('beforeend', out.slice(i, i + 25).join(''));
    i += 25;
    if (i < out.length) requestAnimationFrame(next);
  };
  requestAnimationFrame(next);
}

/**
 * Builds the collapsible "Full working" section containing all row-operation snapshots on [A | I].
 * Kept collapsed by default (false) so initial render stays fast and lag-free.
 * Output: HTML String.
 */
function stepsHTML(r, f) {
  if (r.steps_omitted) return '<p class="mut">The row-by-row snapshots are hidden because the numbers are very long. Use shorter numbers to see them.</p>';
  const S = r.steps;
  if (!S || !S.length) return '';
  const fs = v => typeof v === 'string' ? v.replaceAll('-', '−') : f(v);
  return lazy(
    `Full working: every ${r.mode === 'exact' ? 'row operation' : 'pivot column'} on [A | I] (${S.length - 1} step${S.length === 2 ? '' : 's'})`,
    () => [
      ...(S.length > 1 && !S[1].M ? ['<p class="mut">Above 40x40 only the pivot choices are listed: a table for every step would hold millions of numbers.</p>'] : []),
      ...S.map((s, i) => `<div class="st"><div class="lab"><span class="sn">${i || '0'}</span> ${esc(s.t)}</div>${s.M ? aug(s.M, fs, r.n, s.hi) : ''}</div>`)
    ],
    false,
    'steps',
    `--sw:${Math.max(260, r.n * 2 * 56 + 56)}px`
  );
}

/**
 * Formats the system of linear equations into human-readable algebra lines (e.g., "2x + y − z = 8").
 * Output: Array of Strings.
 */
const eqLines = (A, B) => A.map(row => {
  const t = row.map((v, j) => ({ v, j }))
    .filter(o => !isZ(o.v))
    .map((o, k) => {
      const neg = o.v.n < 0n, a = neg ? fr(-o.v.n, o.v.d) : o.v;
      return (neg ? '− ' : (k ? '+ ' : '')) + (isOne(a) ? '' : str(a)) + xn(o.j, A.length);
    }).join(' ') || '0';
  return `${t} = ${str(B[row.indexOf ? A.indexOf(row) : 0])}`;
}).map((_, i) => {
  const row = A[i];
  const t = row.map((v, j) => ({ v, j }))
    .filter(o => !isZ(o.v))
    .map((o, k) => {
      const neg = o.v.n < 0n, a = neg ? fr(-o.v.n, o.v.d) : o.v;
      return (neg ? '− ' : (k ? '+ ' : '')) + (isOne(a) ? '' : str(a)) + xn(o.j, A.length);
    }).join(' ') || '0';
  return `${t} = ${str(B[i])}`;
});

/**
 * Generates HTML list items showing the exact dot-product arithmetic for each entry of P * Q.
 * Output: HTML String of <li> elements.
 */
const dotWork = (P, Q, f, labelFn) => P.map((row, i) => Q[0].map((_, j) =>
  `<li>${labelFn(i, j)} = ${row.map((p, k) => `(${f(p)})(${f(Q[k][j])})`).join(' + ')} = <b>${str(row.reduce((s, p, k) => add(s, mul(p, Q[k][j])), ZERO))}</b></li>`
).join('')).join('');

/**
 * Generates an array of HTML strings showing floating-point dot-product arithmetic for P * Q.
 * Output: Array of HTML Strings.
 */
const dotWorkF = (P, Q, labelFn) => P.flatMap((row, i) => Q[0].map((_, j) =>
  `<div>${labelFn(i, j)} = ${row.map((p, k) => `(${fmtF(p)})(${fmtF(Q[k][j])})`).join(' + ')} = <b>${fmtF(row.reduce((s, p, k) => s + p * Q[k][j], 0))}</b></div>`
));

/**
 * Computes Step 5 verification checks locally when running in offline browser mode.
 * Output: Object {rows, okL, okR, okI}.
 */
function localVer(r) {
  const ex = r.mode === 'exact', n = r.n, I = ex ? eye(n) : null;
  const rows = r.A.map((row, i) => {
    if (ex) {
      const l = row.reduce((s, v, j) => add(s, mul(v, r.x[j][0])), ZERO);
      return {
        l: str(l), b: str(r.B[i]),
        ok: l.n === r.B[i].n && l.d === r.B[i].d,
        w: row.map((v, j) => `(${str(v)})(${str(r.x[j][0])})`).join(' + ')
      };
    }
    const l = row.reduce((s, v, j) => s + v * r.x[j][0], 0);
    return { l: fmtF(l), b: fmtF(r.B[i]), ok: Math.abs(l - r.B[i]) <= 1e-8 * Math.max(1, Math.abs(r.B[i])) };
  });
  const okL = ex ? same(r.left, I) : r.e1 < 1e-6;
  const okR = ex ? same(r.right, I) : r.e2 < 1e-6;
  const okI = ex ? same(matmul(r.left, r.x), r.x) : matmulF(r.left, r.x).every((q, i) => Math.abs(q[0] - r.x[i][0]) <= 1e-6 * Math.max(1, Math.abs(r.x[i][0])));
  return { rows, okL, okR, okI };
}

/**
 * Builds the HTML card for Step 5 (Verification checklist and equation substitution table).
 * Output: HTML String (<section class="card span">...</section>).
 */
function verifyHTML(r) {
  const ex = r.mode === 'exact', n = r.n, V = r.ver || localVer(r);
  const { rows, okL, okR, okI } = V;
  const hits = rows.filter(q => q.ok).length, hasW = rows.some(q => q.w);
  const items = [
    [true, ex ? 'A is invertible: det(A) != 0' : 'A is invertible: a pivot was found in every column'],
    [hits === n, `Substituting x into the equations gives Ax = B (${hits} of ${n} equations match)`],
    [okL, 'A⁻¹A = Iₙ'],
    [okR, 'AA⁻¹ = Iₙ'],
    [okI, 'Iₙx = x: the identity leaves the solution unchanged']
  ];
  const all = items.every(q => q[0]);
  return `<section class="card span"><h2><span class="n">5</span>Verification</h2>
    <ul class="chk">${items.map(q => `<li class="${q[0] ? 'ok' : 'bad'}">[${q[0] ? 'OK' : 'FAIL'}]${q[1]}</li>`).join('')}</ul>
    <p class="big ${all ? 'ok' : 'bad'}">${all ? 'The solution is valid and the identity matrix stays the same.' : 'Verification failed. ' + (r.cond > 1e8 ? 'The matrix is ill-conditioned, so rounding error is too large for decimal arithmetic to confirm the answer.' : 'Recheck your entries.')}</p>
    <details${n <= 4 ? ' open' : ''}><summary>Equation-by-equation substitution</summary><div class="box"><table class="vt"><tr><th>Row</th>${hasW ? '<th>Substitution</th>' : ''}<th>Sum aᵢⱼxⱼ</th><th>bᵢ</th><th>Status</th></tr>${rows.map((q, i) => `<tr><td>${i + 1}</td>${hasW ? `<td class="wk">${q.w}</td>` : ''}<td>${q.l}</td><td>${q.b}</td><td class="${q.ok ? 'ok' : 'bad'}">${q.ok ? 'OK' : 'FAIL'}</td></tr>`).join('')}</table></div></details></section>`;
}

/**
 * Renders a matrix directly for n <= 24, or wraps it in a lazy <details> drawer for larger sizes.
 * Output: HTML String.
 */
const bm = (r, f, M, label) => r.n > MAX_N ? lazy(`Show ${label} (${r.n}x${r.n})`, () => mat(M, f), false) : mat(M, f);

/**
 * Builds the HTML rows displaying A^-1 * A = I and A * A^-1 = I in Step 3.
 * Output: HTML String.
 */
const checkRows = (r, f) => `<div class="row">${lab('A<sup>-1</sup>')}${bm(r, f, r.inv, 'A⁻¹')}${lab('×')}${lab('A')}${bm(r, f, r.A, 'A')}${lab('=')}${bm(r, f, r.left, 'A⁻¹A')}</div>
  <div class="row" style="margin-top:12px">${lab('A')}${bm(r, f, r.A, 'A')}${lab('×')}${lab('A<sup>-1</sup>')}${bm(r, f, r.inv, 'A⁻¹')}${lab('=')}${bm(r, f, r.right, 'AA⁻¹')}</div>`;

/**
 * Master renderer that generates all 5 output cards from a solver result object r.
 * Output: Complete HTML String for #out.
 */
function report(r) {
  LZ = {};
  lzId = 0;
  const n = r.n, f = r.mode === 'exact' ? str : fmtF, col = v => v.map(x => [x]);
  let h = `<section class="card span"><h2><span class="n">1</span>Matrix equation&nbsp; Ax = B</h2>
    ${r.mode === 'exact' ? `<p class="eqt">${eqLines(r.A, r.B).join('<br>')}</p>` : ''}
    <div class="row">${lab('A')}${bm(r, f, r.A, 'A')}${lab('X')}${mat(Array.from({ length: n }, (_, i) => [xn(i, n)]), f)}${lab('=')}${lab('B')}${mat(col(r.B), f)}</div></section>`;

  if (!r.inv) {
    return h + `<section class="card span"><h2><span class="n">2</span>Inverse A<sup>-1</sup></h2>
      <p class="bad">${r.mode === 'exact' ? 'A is singular (det A = 0), so A<sup>-1</sup> does not exist.' : 'A is singular, or so ill-conditioned that decimal arithmetic cannot invert it.'}</p>
      <p>${r.why ? r.why + '. ' : ''}The inverse method cannot solve this system. Change a coefficient so the rows are independent, or solve it by row reduction instead.</p>${stepsHTML(r, f)}</section>`;
  }

  h += `<section class="card span"><h2><span class="n">2</span>Identify the inverse A<sup>-1</sup></h2><div class="row">${lab('A<sup>-1</sup> =')}${bm(r, f, r.inv, 'A⁻¹')}</div>`;
  if (r.mode === 'exact') {
    h += `<p class="mut" style="margin:10px 0 0">det(A) = ${r.pivots ? `(−1)<sup>${r.swaps}</sup> ×${r.pivots.map(str).join(' × ')} = ` : ''}${str(r.det)}</p>${stepsHTML(r, f)}`;
  } else {
    h += `<p class="mut" style="margin:10px 0 0">Gauss-Jordan with partial pivoting in double precision, shown to 4 decimals. Condition number ≈ ${sci(r.cond)}.</p>` +
      (r.cond > 1e8 ? `<p class="bad">This matrix is ill-conditioned, so the digits of A<sup>-1</sup> and x may be unreliable.</p>` : '') +
      stepsHTML(r, f);
  }
  h += `</section>`;

  h += `<section class="card${n > 5 ? ' span' : ''}"><h2><span class="n">3</span>Check&nbsp; A<sup>-1</sup>A = AA<sup>-1</sup> = I<sub>n</sub></h2>`;
  if (r.mode === 'exact') {
    h += `${checkRows(r, f)}${lazy('Show the multiplication work, entry by entry', () => `<p class="mut">A<sup>-1</sup>A</p><ul class="wl">${dotWork(r.inv, r.A, f, (i, j) => `(${i + 1},${j + 1})`)}</ul><p class="mut">AA<sup>-1</sup></p><ul class="wl">${dotWork(r.A, r.inv, f, (i, j) => `(${i + 1},${j + 1})`)}</ul>`, false)}`;
  } else {
    h += `<p class="mut">Largest difference from the identity: A<sup>-1</sup>A is ${sci(r.e1)}, AA<sup>-1</sup> is ${sci(r.e2)}. Anything below 1e−6 counts as the identity.</p>`;
    h += n <= 15 ? checkRows(r, f) : `<div class="row">${lab('A<sup>-1</sup>')}${bm(r, f, r.inv, 'A⁻¹')}${lab('and')}${lab('A')}${bm(r, f, r.A, 'A')}</div>`;
    if (n > SNAP_MAX) {
      h += '<p class="mut">The entry-by-entry multiplication work is not shown above 40x40 (it would be millions of terms).</p>';
    } else {
      h += lazy('Show the multiplication work, entry by entry (values shown to 4 decimals)', () => ['<p class="mut">A<sup>-1</sup>A</p>', ...dotWorkF(r.inv, r.A, (i, j) => `(${i + 1},${j + 1})`), '<p class="mut">AA<sup>-1</sup></p>', ...dotWorkF(r.A, r.inv, (i, j) => `(${i + 1},${j + 1})`)], false, 'wl');
    }
  }
  h += `<p class="${r.ok ? 'ok' : 'bad'}">${r.ok ? '[OK] Both products equal the identity matrix.' : '[FAIL] A product is not the identity. ' + (r.cond > 1e8 ? 'The matrix is ill-conditioned, so decimal rounding error is too large to confirm it.' : 'Recheck your entries.')}</p></section>`;

  h += `<section class="card${n > 5 ? ' span' : ''}"><h2><span class="n">4</span>Solve&nbsp; [X] = A<sup>-1</sup>B</h2>
    <div class="row">${lab('A<sup>-1</sup>')}${bm(r, f, r.inv, 'A⁻¹')}${lab('B')}${mat(col(r.B), f)}${lab('=')}${mat(r.x, f)}</div>
    <div class="ans${n > 10 ? ' dense' : ''}">${r.x.map((v, i) => `<div>${xn(i, n)} = ${f(v[0])}${r.mode === 'exact' ? `<small>${dec(v[0]) || 'exact'}</small>` : ''}</div>`).join('')}</div>`;
  if (r.mode === 'exact') h += `${lazy('Show how each x is computed: x = A<sup>-1</sup>B', () => `<ul class="wl">${dotWork(r.inv, col(r.B), f, i => xn(i, n))}</ul>`, false)}`;
  if (r.mode === 'float') h += `<p class="mut">Largest error of Ax − B: ${sci(r.res)}</p>`;
  return h + '</section>' + verifyHTML(r);
}

/* ---------- FastAPI Bridge ---------- */

const API = location.protocol.startsWith('http') ? '' : 'http://127.0.0.1:8000';
let seq = 0, ctl = null, ctm;

/**
 * Updates the top status pill to show whether the Python FastAPI backend is reachable.
 * Output: None (updates #api DOM element).
 */
const setApi = on => {
  const p = $('api');
  if (p.dataset.on === String(on)) return;
  p.dataset.on = String(on);
  p.textContent = on ? 'Python API connected' : 'API offline · using the built-in browser solver';
};

/**
 * Sends the current system {A, B, unlimited} to POST /api/solve on the FastAPI backend.
 * Output: Promise resolving to the parsed JSON response from the server.
 */
async function callAPI() {
  if (ctl) ctl.abort();
  ctl = new AbortController();
  const r = await fetch(API + '/api/solve', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ A, B, unlimited: UNLOCK }),
    signal: ctl.signal
  });
  if (r.status === 422) {
    const j = await r.json().catch(() => ({}));
    const e = new Error(typeof j.detail === 'string' ? j.detail : 'The API could not read this system.');
    e.api = true;
    throw e;
  }
  if (!r.ok) throw new Error('API returned ' + r.status);
  return r.json();
}

/**
 * Converts string values returned by the Python API into BigInt fraction objects or Numbers for report().
 * Output: Formatted result Object matching solveExact/solveFloat structure.
 */
function fromAPI(j) {
  const exact = j.mode === 'exact', P = exact ? parseX : (v => v), PI = exact ? parseX : parseF;
  const cv = M => M && M.map(r => r.map(P));
  const r = { ...j, A: A.map(r => r.map(PI)), B: B.map(PI), inv: cv(j.inv), left: cv(j.left), right: cv(j.right), x: cv(j.x), steps: j.steps };
  if (exact) {
    r.det = parseX(j.det);
    r.pivots = j.pivots && j.pivots.map(parseX);
  }
  if (j.ver) {
    r.ver = { ...j.ver, rows: j.ver.rows.map(q => ({ ok: q.ok, l: exact ? str(parseX(q.l)) : fmtF(q.l), b: exact ? str(parseX(q.b)) : fmtF(q.b) })) };
  }
  if (exact && r.ver && r.x) {
    r.ver.rows.forEach((q, i) => { q.w = r.A[i].map((v, k) => `(${str(v)})(${str(r.x[k][0])})`).join(' + '); });
  }
  return r;
}

/**
 * Requests a random invertible system from GET /api/random, falling back to randomSys() if offline.
 * Output: Promise (loads the generated system into the UI).
 */
async function randomFrom(n) {
  try {
    const r = await fetch(API + '/api/random?' + (n ? 'n=' + n + '&' : '') + 'unlimited=' + UNLOCK);
    if (!r.ok) throw 0;
    const j = await r.json();
    load({ a: j.a, b: j.b });
  } catch (e) {
    load(randomSys(n || 2 + Math.floor(Math.random() * ((UNLOCK ? 60 : MAX_N) - 1))));
  }
}

/* ---------- Input & State Management ---------- */

/**
 * Parses a multi-line pasted text block into coefficient matrix A and constant vector B.
 * Output: Object {A, B} or {err: String} if dimensions are invalid.
 */
function parseText(t) {
  const rows = t.split(/\r?\n/).map(l => l.trim()).filter(Boolean).map(l => l.split(/[\s,;]+/).filter(Boolean));
  const m = rows.length;
  if (!m) return { err: 'Type or paste at least one equation.' };
  if (m > LIMIT()) return { err: `That is ${m} equations. The maximum is ${LIMIT()}${UNLOCK ? '' : ' (tick "Remove the 24x24 limit" for more)'}.` };
  for (let i = 0; i < m; i++) {
    if (rows[i].length !== m + 1) {
      return { err: `Line ${i + 1} has ${rows[i].length} numbers. A ${m}x${m} system needs ${m + 1} per line: the coefficients, then the constant.` };
    }
  }
  return { A: rows.map(r => r.slice(0, m)), B: rows.map(r => r[m]) };
}

/**
 * Generates a random invertible n x n system locally in the browser.
 * Output: Object {a: 2D Array of Numbers, b: Array of Numbers}.
 */
function randomSys(n) {
  const R = (lo, hi) => lo + Math.floor(Math.random() * (hi - lo + 1));
  const a = [], x = Array.from({ length: n }, () => R(-9, 9));
  if (n <= EXACT_MAX) {
    const Z = () => Array.from({ length: n }, () => new Array(n).fill(0)), L = Z(), U = Z();
    for (let i = 0; i < n; i++) {
      L[i][i] = U[i][i] = 1;
      for (let j = 0; j < i; j++) { L[i][j] = R(-1, 1); U[j][i] = R(-1, 1); }
    }
    for (let i = 0; i < n; i++) {
      a.push([]);
      for (let j = 0; j < n; j++) {
        let s = 0;
        for (let k = 0; k < n; k++) s += L[i][k] * U[k][j];
        a[i].push(s);
      }
    }
  } else {
    for (let i = 0; i < n; i++) {
      let s = 0;
      const r = [];
      for (let j = 0; j < n; j++) {
        const v = R(-9, 9);
        r.push(v);
        if (j !== i) s += Math.abs(v);
      }
      r[i] = s + 1 + R(0, 5);
      a.push(r);
    }
  }
  return { a, b: a.map(r => r.reduce((s, v, j) => s + v * x[j], 0)) };
}

const PRESETS = {
  sample: { a: [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]], b: [8, -11, -3] },
  two: { a: [[2, 1], [5, 3]], b: [3, 8] },
  sing: { a: [[1, 2], [2, 4]], b: [3, 6] }
};

let N = 3, A = [], B = [], tmr;

/**
 * Shorthand helper for document.getElementById(id).
 * Output: DOM Element.
 */
const $ = id => document.getElementById(id);

/**
 * Resizes the system to n x n while preserving existing typed entries (new entries default to '0').
 * Output: None (updates state and triggers build()).
 */
function setSize(n) {
  const nA = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (A[i] && A[i][j] !== undefined) ? A[i][j] : '0'));
  const nB = Array.from({ length: n }, (_, i) => B[i] !== undefined ? B[i] : '0');
  N = n; A = nA; B = nB;
  build();
}

/**
 * Loads a preset or randomized system object {a, b} into state and rebuilds the UI.
 * Output: None.
 */
function load(p) {
  N = p.a.length;
  A = p.a.map(r => r.map(String));
  B = p.b.map(String);
  build();
}

/**
 * Synchronizes the optional paste <textarea> with the current grid values, only when the drawer is open.
 * Output: None.
 */
function syncText() {
  if (!$('textBox').open) return;
  $('txt').value = A.map((r, i) => r.concat(B[i]).join(' ')).join('\n');
}

/**
 * Escapes HTML special characters to prevent XSS or broken attributes.
 * Output: Safe HTML String.
 */
const esc = s => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

/**
 * Builds the compact table grid HTML for systems with N > 4.
 * Output: HTML String (<table class="cg">...</table>).
 */
const gridHTML = () => `<table class="cg"><thead><tr>${A.map((_, j) => `<th>${xn(j)}</th>`).join('')}<th></th><th>B</th></tr></thead><tbody>${A.map((row, i) => '<tr>' + row.map((v, j) => `<td><input inputmode="decimal" aria-label="Row ${i + 1}, coefficient of ${xn(j)}" data-r="${i}" data-c="${j}" value="${esc(v)}"></td>`).join('') + `<td class="lab">=</td><td><input inputmode="decimal" aria-label="Row ${i + 1}, constant" data-r="${i}" data-c="b" value="${esc(B[i])}"></td></tr>`).join('')}</tbody></table>`;

/**
 * Rebuilds the input form controls, size chips, and triggers a solve calculation.
 * Output: None.
 */
function build(skipText) {
  $('size').value = N;
  $('layout').classList.toggle('side', N <= EQ_STYLE);
  $('size').max = LIMIT();$('unlock').checked = UNLOCK;
  document.querySelectorAll('#chips .big').forEach(b => { b.hidden = !UNLOCK; });
  $('solveBar').hidden = !(N > MAX_N);
  document.querySelectorAll('#chips button').forEach(b => b.setAttribute('aria-pressed', String(+b.dataset.n === N)));
  $('mode').textContent = N <= EXACT_MAX ? 'Exact fractions' : 'Decimal, double precision';

  if (N > GRID_MAX) {
    $('form').innerHTML = `<p class="hint">The grid is available up to ${GRID_MAX}x${GRID_MAX}. For ${N}x${N}, paste the system into the text box below, or press Randomize.</p>`;
    $('textBox').open = true;
  } else if (N > EQ_STYLE) {
    $('form').innerHTML = gridHTML();
  } else {
    $('form').innerHTML = A.map((row, i) => '<div class="eq">' + row.map((v, j) =>
      `<input inputmode="decimal" aria-label="Row ${i + 1}, coefficient of ${xn(j)}" data-r="${i}" data-c="${j}" value="${esc(v)}"><span class="lab">${xn(j)}</span>${j < N - 1 ? '<span class="op">+</span>' : ''}`
    ).join('') + `<span class="op">=</span><input inputmode="decimal" aria-label="Row ${i + 1}, constant" data-r="${i}" data-c="b" value="${esc(B[i])}"></div>`).join('');
  }

  if (!skipText) syncText();
  compute();
}

/**
 * Validates all inputs, queries the FastAPI backend (or local solver fallback), and renders the solution cards.
 * Avoids redundant DOM attribute mutations to keep typing responsive.
 * Output: Promise.
 */
async function compute(force) {
  const exact = N <= EXACT_MAX, P = exact ? parseX : parseF;
  let bad = null;
  const pa = A.map((r, i) => r.map((s, j) => {
    const v = P(s);
    if (v === null && !bad) bad = `Row ${i + 1}, column ${j + 1}`;
    return v;
  }));
  const pb = B.map((s, i) => {
    const v = P(s);
    if (v === null && !bad) bad = `Row ${i + 1}, constant`;
    return v;
  });

  document.querySelectorAll('#form input').forEach(el => {
    const v = el.dataset.c === 'b' ? pb[+el.dataset.r] : pa[+el.dataset.r][+el.dataset.c];
    const invalidStr = String(v === null);
    if (el.getAttribute('aria-invalid') !== invalidStr) {
      el.setAttribute('aria-invalid', invalidStr);
    }
  });

  $('err').textContent = bad ? `${bad} is not a number. Use values like 3, -2, 1.5 or 3/4 (up to 40 characters).` : '';
  if (bad) {
    seq++;
    if (ctl) ctl.abort();
    $('out').innerHTML = '';
    return;
  }

  if (N <= MAX_N) {
    $('solveHint').textContent = '';
  } else if (!force) {
    seq++;
    if (ctl) ctl.abort();
    $('out').innerHTML = '';$('solveHint').textContent = `${N}x${N} is a large system: press Solve to compute it. This can take a few seconds.`;
    return;
  } else {
    $('out').innerHTML = '<p class="mut">Solving…</p>';
    $('solveBtn').disabled = true;
    $('solveHint').textContent = '';
  }

  const mine = ++seq;
  try {
    let r;
    try {
      const j = await callAPI();
      if (mine !== seq) return;
      r = fromAPI(j);
      setApi(true);
    } catch (e) {
      if (e.name === 'AbortError') return;
      if (e.api) {
        $('err').textContent = e.message;
        $('out').innerHTML = '';
        return;
      }
      setApi(false);
      r = exact ? solveExact(pa, pb) : solveFloat(pa, pb);
    }
    if (mine === seq) {
      requestAnimationFrame(() => {
        if (mine !== seq) return;
        $('out').innerHTML = report(r);$('out').querySelectorAll('details[data-lz][open]').forEach(fillLazy);
      });
    }
  } catch (e) {
    $('err').textContent = 'Something went wrong: ' + e.message;
  } finally {
    $('solveBtn').disabled = false;
  }
}

/**
 * Reads the text box contents and updates the grid when the user pastes or types in the textarea.
 * Output: None.
 */
function fromText() {
  const r = parseText($('txt').value);
  if (r.err) {
    $('err').textContent = r.err;
    $('out').innerHTML = '';
    return;
  }
  N = r.A.length; A = r.A; B = r.B;
  build(true);
}

/**
 * Displays a confirmation modal dialog and awaits the user's choice.
 * Output: Promise resolving to Boolean (true if confirmed, false if cancelled).
 */
function askConfirm(title, bullets, okText) {
  return new Promise(res => {
    $('mTitle').textContent = title;
    $('mList').innerHTML = bullets.map(b => `<li>${b}</li>`).join('');
    $('mOk').textContent = okText;
    const m = $('modal');
    m.hidden = false;
    $('mCancel').focus();
    const key = e => { if (e.key === 'Escape') done(false); };
    const done = v => {
      m.hidden = true;
      $('mOk').onclick =$('mCancel').onclick = m.onclick = null;
      document.removeEventListener('keydown', key);
      res(v);
    };
    document.addEventListener('keydown', key);
    $('mOk').onclick = () => done(true);$('mCancel').onclick = () => done(false);
    m.onclick = e => { if (e.target === m) done(false); };
  });
}

/* ---------- Event Listeners ---------- */

if (typeof document !== 'undefined') {
  $('chips').addEventListener('click', e => {
    const b = e.target.closest('button');
    if (b) setSize(+b.dataset.n);
  });

  $('size').addEventListener('change', () => {
    let v = parseInt($('size').value, 10);
    if (!(v >= 1)) v = N;
    $('sizeNote').textContent = v > LIMIT() ? `The maximum is ${LIMIT()}x${LIMIT()}${UNLOCK ? '' : ' (tick "Remove the 24x24 limit" for more)'}.` : '';
    setSize(Math.min(v, LIMIT()));
  });

  $('presets').addEventListener('click', e => {
    const p = e.target.dataset.p;
    if (!p) return;
    if (p === 'randall') randomFrom();
    else if (p === 'rand') randomFrom(N < 2 ? 3 : N);
    else if (p === 'clear') { A = A.map(r => r.map(() => '0')); B = B.map(() => '0'); build(); }
    else load(PRESETS[p]);
    $('sizeNote').textContent = '';
  });

  $('form').addEventListener('input', e => {
    const el = e.target, i = +el.dataset.r;
    if (el.dataset.r === undefined) return;
    if (el.dataset.c === 'b') B[i] = el.value;
    else A[i][+el.dataset.c] = el.value;
    clearTimeout(ctm);
    ctm = setTimeout(() => { syncText(); compute(); }, 200);
  });

  $('textBox').addEventListener('toggle', () => {
    if ($('textBox').open) syncText();
  });

  $('txt').addEventListener('input', () => {
    clearTimeout(tmr);
    tmr = setTimeout(fromText, N > EXACT_MAX ? 250 : 120);
  });

  $('form').addEventListener('paste', e => {
    const el = e.target, t = (e.clipboardData || window.clipboardData).getData('text');
    const rows = t.split(/\r?\n/).filter(l => l.trim()).map(l => l.trim().split(/[\s,;]+/));
    if (el.dataset.r === undefined || !rows.length || (rows.length < 2 && rows[0].length < 2)) return;
    e.preventDefault();
    const r0 = +el.dataset.r, c0 = el.dataset.c === 'b' ? N : +el.dataset.c;
    rows.forEach((row, di) => row.forEach((v, dj) => {
      const i = r0 + di, j = c0 + dj;
      if (i < N && j < N) A[i][j] = v;
      else if (i < N && j === N) B[i] = v;
    }));
    build();
  });

  $('solveBtn').addEventListener('click', () => compute(true));

  $('unlock').addEventListener('change', async () => {
    const box = $('unlock');
    if (box.checked) {
      const ok = await askConfirm('Remove the size limit?', [
        '<b>The page can slow down or freeze.</b> A 100x100 system has 10,000 numbers and a 200x200 system has 40,000.',
        '<b>Solving takes a few seconds.</b> Large systems are solved when you press <b>Solve</b>, not while you type.',
        '<b>Less working is shown.</b> Above 40x40 you see the pivot choices instead of a table for every step, and big matrices stay collapsed until you open them.',
        '<b>Decimals only.</b> Rounding errors can build up, and the page warns you about matrices that are hard to invert.',
        'The largest size is 200x200.'
      ], 'Remove the limit');
      if (!ok) { box.checked = false; return; }
      UNLOCK = true;
    } else {
      if (N > MAX_N && !(await askConfirm(`Go back to ${MAX_N}x${MAX_N}?`, [`Your ${N}x${N} system will be cut down to ${MAX_N}x${MAX_N}: the extra rows and columns are removed.`], 'Cut it down'))) {
        box.checked = true;
        return;
      }
      UNLOCK = false;
    }
    if (N > LIMIT()) setSize(LIMIT());
    else build();
  });

  $('out').addEventListener('toggle', e => {
    if (e.target.dataset && e.target.dataset.lz) fillLazy(e.target);
  }, true);

  load(PRESETS.sample);
}