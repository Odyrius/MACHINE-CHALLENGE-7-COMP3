// MC#7 PAGE LOGIC. How it fits together:
//   type a system -> compute() sends it to the Python API -> report() draws Steps 1-5.
//   If the API is not running, the built-in solvers below (solveExact / solveFloat) do the same job.
// SETTINGS: MAX_N = biggest system. EXACT_MAX = up to this size use exact fractions, above it decimals
// (keep it equal to EXACT_MAX in mc7_api.py). EQ_STYLE = up to this size the input looks like written equations.
const MAX_N=24, ABS_MAX=200, SNAP_MAX=40, GRID_MAX=60, EXACT_MAX=10, EQ_STYLE=4;
// MAX_N = normal limit. ABS_MAX = ceiling after "Remove the size limit". SNAP_MAX = up to this size the working shows a table for
// every step. GRID_MAX = up to this size you type in the grid (bigger: paste into the text box or press Randomize).
let UNLOCK=false; // becomes true after the person ticks "Remove the size limit" and accepts the warning
const LIMIT=()=>UNLOCK?ABS_MAX:MAX_N; // the biggest size allowed right now

/* ---------- exact rationals (BigInt) for n <= EXACT_MAX ---------- */
// ---------- exact fractions (BigInt) ----------
// babs(a): absolute value for BigInt (Math.abs cannot do BigInt).
const babs=a=>a<0n?-a:a;
// gcd(a,b): greatest common divisor (Euclid). Used to reduce 6/8 to 3/4.
const gcd=(a,b)=>{a=babs(a);b=babs(b);while(b){[a,b]=[b,a%b]}return a};
// fr(n,d): builds a fraction in lowest terms with a positive denominator.
// Every exact number in the page is {n: numerator, d: denominator} made of BigInts, so it never overflows.
const fr=(n,d=1n)=>{n=BigInt(n);d=BigInt(d);if(d<0n){n=-n;d=-d}const g=gcd(n,d)||1n;return{n:n/g,d:d/g}};
// add / sub: fraction + and -.
const add=(a,b)=>fr(a.n*b.d+b.n*a.d,a.d*b.d), sub=(a,b)=>fr(a.n*b.d-b.n*a.d,a.d*b.d);
// mul / div: fraction x and /.
const mul=(a,b)=>fr(a.n*b.n,a.d*b.d), div=(a,b)=>fr(a.n*b.d,a.d*b.n);
// isZ / isOne: is this fraction 0 / 1?  ZERO and ONE are ready-made constants.
const isZ=a=>a.n===0n, isOne=a=>a.n===1n&&a.d===1n, ZERO=fr(0), ONE=fr(1);
// str(a): fraction -> text for display: 3/4, or 3 when the denominator is 1, with a real minus sign.
const str=a=>(a.d===1n?a.n.toString():a.n+'/'+a.d).replace('-','−');
// dec(a): decimal approximation shown under exact answers, e.g. 0.7500.
const dec=a=>{const q=Number(a.n)/Number(a.d);return a.d===1n||!isFinite(q)?'':'≈ '+q.toFixed(4).replace('-','−')};

// NUM: what counts as a number: 3, -2, 1.5, .5, 3/4, 1.5/3 (no exponents, no letters).
const NUM=/^([+-]?\d*\.?\d+)(?:\/(\d*\.?\d+))?$/;
// clean(s): trims spaces, turns the typographic minus into "-", and rejects entries over 40 characters
// (the same limit as the API) by returning "", which NUM then refuses.
const clean=s=>{const t=s.trim().replace('−','-');return t.length>40?'':t}; // same 40-character limit as the API
// parseX(s): typed text -> exact fraction, or null if it is not a valid number.
// Decimals convert exactly: 1.25 becomes 5/4.
function parseX(s){
  const m=clean(s).match(NUM); if(!m) return null;
  const t=x=>{const neg=x[0]==='-',u=x.replace(/^[+-]/,''),k=(u.split('.')[1]||'').length;
    const i=BigInt(u.replace('.',''));return fr(neg?-i:i,10n**BigInt(k))};
  const a=t(m[1]); if(m[2]===undefined) return a;
  const b=t(m[2]); return isZ(b)?null:div(a,b);
}
// parseF(s): same as parseX but returns an ordinary decimal number (used above EXACT_MAX).
function parseF(s){
  const m=clean(s).match(NUM); if(!m) return null;
  const a=Number(m[1]); if(m[2]===undefined) return a;
  const b=Number(m[2]); return b===0?null:a/b;
}

// ---------- matrix helpers ----------
// eye(n): the identity matrix made of fractions.
const eye=n=>Array.from({length:n},(_,i)=>Array.from({length:n},(_,j)=>i===j?ONE:ZERO));
// matmul(P,Q): exact matrix multiplication (row times column, then add up).
const matmul=(P,Q)=>P.map(r=>Q[0].map((_,j)=>r.reduce((s,v,k)=>add(s,mul(v,Q[k][j])),ZERO)));
// same(P,Q): true when two exact matrices are identical. Used to confirm a product equals I.
const same=(P,Q)=>P.every((r,i)=>r.every((v,j)=>v.n===Q[i][j].n&&v.d===Q[i][j].d));

// ---------- built-in solvers (used when the API is offline) ----------
// solveExact(A,B): exact Gauss-Jordan, the JavaScript twin of gauss_jordan_steps in Python.
// Returns the inverse, determinant, both check products, x and every row-operation snapshot.
function solveExact(A,B){
  const n=A.length, I=eye(n), log=[];
  let M=A.map((r,i)=>[...r,...I[i]]), det=ONE;
  const steps=[],pivots=[];let swaps=0;const note=(t,hi)=>{log.push(t);steps.push({t,M:M.map(r=>r.slice()),hi})};steps.push({t:'Write [A | I]',M:M.map(r=>r.slice()),hi:null});
  for(let c=0;c<n;c++){
    // 1) find a pivot: the first row at or below the diagonal with a nonzero entry in this column
    let p=-1; for(let r=c;r<n;r++) if(!isZ(M[r][c])){p=r;break}
    if(p<0){note(`No pivot in column ${c+1}: A is singular.`,null);return{mode:'exact',n,A,B,inv:null,log,det:ZERO,steps}}
    if(p!==c){[M[c],M[p]]=[M[p],M[c]];det=mul(det,fr(-1));swaps++;note(`R${c+1} ↔ R${p+1}`,c)}
    // 2) the pivot also feeds the determinant: product of pivots, sign flips for every swap
    const pv=M[c][c]; det=mul(det,pv); pivots.push(pv);
    if(!isOne(pv)){M[c]=M[c].map(v=>div(v,pv));note(`R${c+1} → R${c+1} ÷ ${str(pv)}`,c)}
    // 3) clear the rest of the column: subtract f x (pivot row) from every other row
    for(let r=0;r<n;r++) if(r!==c&&!isZ(M[r][c])){
      const f=M[r][c]; M[r]=M[r].map((v,j)=>sub(v,mul(f,M[c][j])));
      note(`R${r+1} → R${r+1} − (${str(f)})·R${c+1}`,r);
    }
  }
  const inv=M.map(r=>r.slice(n)), left=matmul(inv,A), right=matmul(A,inv);
  return{mode:'exact',n,A,B,inv,log,det,left,right,ok:same(left,I)&&same(right,I),x:matmul(inv,B.map(v=>[v])),steps,pivots,swaps};
}

/* ---------- double precision, partial pivoting, for n up to 100 ---------- */
// matmulF(P,Q): decimal matrix multiplication (skips zero entries for speed).
function matmulF(P,Q){
  const n=P.length,m=Q[0].length,k=Q.length,R=Array.from({length:n},()=>new Array(m).fill(0));
  for(let i=0;i<n;i++){const Ri=R[i],Pi=P[i];for(let t=0;t<k;t++){const p=Pi[t];if(p===0)continue;const Qt=Q[t];for(let j=0;j<m;j++)Ri[j]+=p*Qt[j]}}
  return R;
}
// dev(M): the largest distance of any entry from the identity matrix. Near 0 means "M is the identity".
const dev=M=>{let e=0;M.forEach((r,i)=>r.forEach((v,j)=>{e=Math.max(e,Math.abs(v-(i===j?1:0)))}));return e};
// norm1(M): the largest column sum of absolute values. Used for the condition number.
const norm1=M=>{let b=0;for(let j=0;j<M.length;j++){let s=0;for(let i=0;i<M.length;i++)s+=Math.abs(M[i][j]);b=Math.max(b,s)}return b};
// solveFloat(A,B): decimal Gauss-Jordan with partial pivoting (twin of float_inverse + float_payload in Python).
// Returns the inverse, both products, x, how far the products are from I (e1, e2), the error of Ax-B (res),
// the condition number (cond) and one snapshot per pivot column.
function solveFloat(A,B){
  const n=A.length; const scale=A[0].map((_,c)=>Math.max(...A.map(r=>Math.abs(r[c])))); // size of each ORIGINAL column
  const M=A.map((r,i)=>{const row=r.concat(new Array(n).fill(0));row[n+i]=1;return row});
  const keep=n<=SNAP_MAX; // above this size a snapshot of every column would hold millions of numbers
  const steps=[{t:'Write [A | I]',M:keep?M.map(r=>r.slice()):null,hi:null}];
  for(let c=0;c<n;c++){
    // partial pivoting: choose the row with the largest |entry| in this column
    let p=c; for(let r=c+1;r<n;r++) if(Math.abs(M[r][c])>Math.abs(M[p][c])) p=r;
    if(Math.abs(M[p][c])<=1e-12*scale[c]) return{steps,mode:'float',n,A,B,inv:null,why:`No usable pivot in column ${c+1}`};
    [M[c],M[p]]=[M[p],M[c]];
    const Mc=M[c],pv=Mc[c]; for(let j=0;j<2*n;j++) Mc[j]/=pv;
    for(let r=0;r<n;r++){if(r===c)continue;const f=M[r][c];if(f===0)continue;const Mr=M[r];for(let j=0;j<2*n;j++)Mr[j]-=f*Mc[j]}
    steps.push({t:`Column ${c+1}: ${p!==c?`swap R${c+1} ↔ R${p+1} (largest entry), `:''}divide R${c+1} by ${fmtF(pv)}, then clear the rest of the column`,M:keep?M.map(r=>r.slice()):null,hi:c});
  }
  const inv=M.map(r=>r.slice(n)),left=matmulF(inv,A),right=matmulF(A,inv),x=matmulF(inv,B.map(v=>[v]));
  let res=0; matmulF(A,x).forEach((r,i)=>{res=Math.max(res,Math.abs(r[0]-B[i]))});
  const e1=dev(left),e2=dev(right);
  return{mode:'float',n,A,B,inv,left,right,x,e1,e2,res,cond:norm1(A)*norm1(inv),ok:Math.max(e1,e2)<1e-6,steps};
}

/* ---------- rendering ---------- */
// ---------- drawing helpers ----------
// subs: the subscript digits used to name the unknowns.
const subs='₀₁₂₃₄₅₆₇₈₉';
// xn(i,n): names unknown number i. Up to 4 unknowns use x, y, z, w (like on the board);
// bigger systems use x1, x2, ... (with subscript digits). n defaults to the current size N.
const xn=(i,n=N)=>n<=4?'xyzw'[i]:'x'+String(i+1).replace(/\d/g,d=>subs[d]);
// fmtF(v): decimal -> text: 4 places, scientific notation for very large or small values, tiny noise shown as 0.
const fmtF=v=>{const a=Math.abs(v);if(a<1e-10)return'0';return((a>=1e5||a<1e-3)?v.toExponential(2):String(+v.toFixed(4))).replace('-','−')};
// mat(M,f): draws a matrix as a bracketed HTML table. f says how to print each entry.
const mat=(M,f)=>`<div class="box"><table class="m" role="presentation">${M.map(r=>'<tr>'+r.map(v=>`<td>${typeof v==='string'?v:f(v)}</td>`).join('')+'</tr>').join('')}</table></div>`;
// lab(s): a small label placed between matrices (A, x, =).
const lab=s=>`<span class="lab">${s}</span>`;
// sci(v): short scientific notation for error sizes, e.g. 2.1e-15.
const sci=v=>v===0?'0':v.toExponential(1).replace('-','−');

// aug(M,f,n,hi): draws one snapshot of [A | I] with a divider after column n and row hi highlighted.
const aug=(M,f,n,hi)=>`<div class="box"><table class="m" role="presentation">${M.map((r,i)=>`<tr${i===hi?' class="hl"':''}>${r.map((v,j)=>`<td${j===n?' class="dv"':''}>${f(v)}</td>`).join('')}</tr>`).join('')}</table></div>`;
// LAZY SECTIONS: heavy parts (step tiles, multiplication work) are only built when you open them,
// so the page stays fast. LZ remembers how to build each one.
let LZ={},lzId=0;
// lazy(summary,build,open,cls,style): returns an empty <details> and remembers how to fill it later.
const lazy=(summary,build,open,cls,style)=>{const id='lz'+(lzId++);LZ[id]=build;
  return `<details data-lz="${id}"${open?' open':''}><summary>${summary}</summary><div class="${cls||''}" id="${id}"${style?` style="${style}"`:''}></div></details>`};
// fillLazy(d): builds the content of a <details> that was just opened.
// Big lists are added 10 items per frame so the page never freezes.
function fillLazy(d){
  const id=d.dataset.lz,b=LZ[id]; if(!b||!d.open) return; delete LZ[id];
  const el=document.getElementById(id), out=b();
  if(!Array.isArray(out)){el.innerHTML=out;return}
  let i=0; const next=()=>{if(!el.isConnected)return;el.insertAdjacentHTML('beforeend',out.slice(i,i+10).join(''));i+=10;if(i<out.length)requestAnimationFrame(next)}; next();
}
// stepsHTML(r,f): the "Full working" section: one tile per row operation (exact)
// or per pivot column (decimal), each showing [A | I] with the changed row highlighted.
function stepsHTML(r,f){
  if(r.steps_omitted) return '<p class="mut">The row-by-row snapshots are hidden because the numbers are very long. Use shorter numbers to see them.</p>';
  const S=r.steps; if(!S||!S.length) return '';
  const fs=v=>typeof v==='string'?v.replace('-','−'):f(v);
  return lazy(`Full working: every ${r.mode==='exact'?'row operation':'pivot column'} on [A | I] (${S.length-1} step${S.length===2?'':'s'})`,
    ()=>[...(S.length>1&&!S[1].M?['<p class="mut">Above 40×40 only the pivot choices are listed: a table for every step would hold millions of numbers.</p>']:[]),...S.map((s,i)=>`<div class="st"><div class="lab"><span class="sn">${i||'▶'}</span> ${esc(s.t)}</div>${s.M?aug(s.M,fs,r.n,s.hi):''}</div>`)],
    r.n<=6,'steps',`--sw:${Math.max(260,r.n*2*56+56)}px`);
}
// eqLines(A,B): writes the equations in words, e.g. 2x1 + x2 - x3 = 8 (Step 1).
const eqLines=(A,B)=>A.map((row,i)=>{
  const t=row.map((v,j)=>({v,j})).filter(o=>!isZ(o.v)).map((o,k)=>{const neg=o.v.n<0n,a=neg?fr(-o.v.n,o.v.d):o.v;return(neg?'− ':(k?'+ ':''))+(isOne(a)?'':str(a))+xn(o.j,A.length)}).join(' ')||'0';
  return `${t} = ${str(B[i])}`});
// dotWork(P,Q,f,lab): exact entry-by-entry arithmetic of P x Q: row i times column j,
// e.g. (4)(2) + (3)(-3) + (-1)(-2) = 1.
const dotWork=(P,Q,f,lab)=>P.map((row,i)=>Q[0].map((_,j)=>`<li>${lab(i,j)} = ${row.map((p,k)=>`(${f(p)})(${f(Q[k][j])})`).join(' + ')} = <b>${str(row.reduce((s,p,k)=>add(s,mul(p,Q[k][j])),ZERO))}</b></li>`).join('')).join('');
// localVer(r): the Step 5 verification computed in the browser (used when the API did not send one).
function localVer(r){
  const ex=r.mode==='exact', n=r.n, I=ex?eye(n):null;
  const rows=r.A.map((row,i)=>{
    if(ex){const l=row.reduce((s,v,j)=>add(s,mul(v,r.x[j][0])),ZERO);return{l:str(l),b:str(r.B[i]),ok:l.n===r.B[i].n&&l.d===r.B[i].d,w:row.map((v,j)=>`(${str(v)})(${str(r.x[j][0])})`).join(' + ')}}
    const l=row.reduce((s,v,j)=>s+v*r.x[j][0],0);return{l:fmtF(l),b:fmtF(r.B[i]),ok:Math.abs(l-r.B[i])<=1e-8*Math.max(1,Math.abs(r.B[i]))}});
  const hits=rows.filter(q=>q.ok).length;
  const okL=ex?same(r.left,I):r.e1<1e-6, okR=ex?same(r.right,I):r.e2<1e-6;
  const okI=ex?same(matmul(r.left,r.x),r.x):matmulF(r.left,r.x).every((q,i)=>Math.abs(q[0]-r.x[i][0])<=1e-6*Math.max(1,Math.abs(r.x[i][0])));
  return{rows,okL,okR,okI};
}
// verifyHTML(r): Step 5. A checklist (A invertible, Ax = B, A^-1 A = I, A A^-1 = I, I x = x)
// plus a table that puts x back into every original equation.
function verifyHTML(r){
  const ex=r.mode==='exact', n=r.n, V=r.ver||localVer(r), rows=V.rows, okL=V.okL, okR=V.okR, okI=V.okI, hits=rows.filter(q=>q.ok).length, hasW=rows.some(q=>q.w);
  const items=[[true,ex?'A is invertible: det(A) ≠ 0':'A is invertible: a pivot was found in every column'],
    [hits===n,`Substituting x into the equations gives Ax = B (${hits} of ${n} equations match)`],
    [okL,'A⁻¹A = Iₙ'],[okR,'AA⁻¹ = Iₙ'],[okI,'Iₙx = x: the identity leaves the solution unchanged']];
  const all=items.every(q=>q[0]);
  return `<section class="card span"><h2><span class="n">5</span>Verification</h2><ul class="chk">${items.map(q=>`<li class="${q[0]?'ok':'bad'}">${q[0]?'✔':'✘'} ${q[1]}</li>`).join('')}</ul>
   <p class="big ${all?'ok':'bad'}">${all?'✔ The solution is valid and the identity matrix stays the same.':'✘ Verification failed. '+(r.cond>1e8?'The matrix is ill-conditioned, so rounding error is too large for decimal arithmetic to confirm the answer.':'Recheck your entries.')}</p>
   <details${n<=10?' open':''}><summary>Equation-by-equation substitution</summary><div class="box"><table class="vt"><tr><th>Row</th>${hasW?'<th>Substitution</th>':''}<th>Σ aᵢⱼxⱼ</th><th>bᵢ</th><th></th></tr>${rows.map((q,i)=>`<tr><td>${i+1}</td>${hasW?`<td class="wk">${q.w}</td>`:''}<td>${q.l}</td><td>${q.b}</td><td class="${q.ok?'ok':'bad'}">${q.ok?'✔':'✘'}</td></tr>`).join('')}</table></div></details></section>`;
}
// dotWorkF(P,Q,lab): the decimal version of dotWork (values shown to 4 decimals).
const dotWorkF=(P,Q,lab)=>P.flatMap((row,i)=>Q[0].map((_,j)=>`<div>${lab(i,j)} = ${row.map((p,k)=>`(${fmtF(p)})(${fmtF(Q[k][j])})`).join(' + ')} = <b>${fmtF(row.reduce((s,p,k)=>s+p*Q[k][j],0))}</b></div>`));
// bm(r,f,M,label): a matrix, or for systems above 24x24 a collapsed section that draws it only when opened, so the page stays fast.
const bm=(r,f,M,label)=>r.n>MAX_N?lazy(`Show ${label} (${r.n}×${r.n})`,()=>mat(M,f),false):mat(M,f);
// checkRows(r,f): Step 3 with every matrix on screen: A^-1 x A = result, and A x A^-1 = result.
const checkRows=(r,f)=>`<div class="row">${lab('A<sup>-1</sup>')}${bm(r,f,r.inv,'A⁻¹')}${lab('×')}${lab('A')}${bm(r,f,r.A,'A')}${lab('=')}${bm(r,f,r.left,'A⁻¹A')}</div>
   <div class="row" style="margin-top:12px">${lab('A')}${bm(r,f,r.A,'A')}${lab('×')}${lab('A<sup>-1</sup>')}${bm(r,f,r.inv,'A⁻¹')}${lab('=')}${bm(r,f,r.right,'AA⁻¹')}</div>`;
// report(r): builds Steps 1 to 5 as HTML for either arithmetic mode and handles singular matrices.
function report(r){
  LZ={};lzId=0;
  const n=r.n, f=r.mode==='exact'?str:fmtF, col=v=>v.map(x=>[x]);
  let h=`<section class="card span"><h2><span class="n">1</span>Matrix equation&nbsp; Ax = B</h2>
   ${r.mode==='exact'?`<p class="eqt">${eqLines(r.A,r.B).join('<br>')}</p>`:''}
   <div class="row">${lab('A')}${bm(r,f,r.A,'A')}${lab('X')}${mat(Array.from({length:n},(_,i)=>[xn(i,n)]),f)}${lab('=')}${lab('B')}${mat(col(r.B),f)}</div></section>`;
  if(!r.inv){
    return h+`<section class="card span"><h2><span class="n">2</span>Inverse A<sup>-1</sup></h2>
     <p class="bad">${r.mode==='exact'?'A is singular (det A = 0), so A<sup>-1</sup> does not exist.':'A is singular, or so ill-conditioned that decimal arithmetic cannot invert it.'}</p>
     <p>${r.why?r.why+'. ':''}The inverse method cannot solve this system. Change a coefficient so the rows are independent, or solve it by row reduction instead.</p>${stepsHTML(r,f)}</section>`;
  }
  h+=`<section class="card span"><h2><span class="n">2</span>Identify the inverse A<sup>-1</sup></h2><div class="row">${lab('A<sup>-1</sup> =')}${bm(r,f,r.inv,'A⁻¹')}</div>`;
  if(r.mode==='exact') h+=`<p class="mut" style="margin:10px 0 0">det(A) = ${r.pivots?`(−1)<sup>${r.swaps}</sup> × ${r.pivots.map(str).join(' × ')} = `:''}${str(r.det)}</p>${stepsHTML(r,f)}`;
  else h+=`<p class="mut" style="margin:10px 0 0">Gauss–Jordan with partial pivoting in double precision, shown to 4 decimals. Condition number ≈ ${sci(r.cond)}.</p>`+
   (r.cond>1e8?`<p class="bad">This matrix is ill-conditioned, so the digits of A<sup>-1</sup> and x may be unreliable.</p>`:'');
  if(r.mode==='float') h+=stepsHTML(r,f);
  h+=`</section>`;
  h+=`<section class="card${n>5?' span':''}"><h2><span class="n">3</span>Check&nbsp; A<sup>-1</sup>A = AA<sup>-1</sup> = I<sub>n</sub></h2>`;
  if(r.mode==='exact') h+=`${checkRows(r,f)}${lazy('Show the multiplication work, entry by entry',()=>`<p class="mut">A<sup>-1</sup>A</p><ul class="wl">${dotWork(r.inv,r.A,f,(i,j)=>`(${i+1},${j+1})`)}</ul><p class="mut">AA<sup>-1</sup></p><ul class="wl">${dotWork(r.A,r.inv,f,(i,j)=>`(${i+1},${j+1})`)}</ul>`,false)}`;
  else{
    h+=`<p class="mut">Largest difference from the identity: A<sup>-1</sup>A is ${sci(r.e1)}, AA<sup>-1</sup> is ${sci(r.e2)}. Anything below 1e−6 counts as the identity.</p>`;
    h+=n<=15?checkRows(r,f):`<div class="row">${lab('A<sup>-1</sup>')}${bm(r,f,r.inv,'A⁻¹')}${lab('and')}${lab('A')}${bm(r,f,r.A,'A')}</div>`;
    if(n>SNAP_MAX) h+='<p class="mut">The entry-by-entry multiplication work is not shown above 40×40 (it would be millions of terms).</p>';
    else h+=lazy('Show the multiplication work, entry by entry (values shown to 4 decimals)',()=>['<p class="mut">A<sup>-1</sup>A</p>',...dotWorkF(r.inv,r.A,(i,j)=>`(${i+1},${j+1})`),'<p class="mut">AA<sup>-1</sup></p>',...dotWorkF(r.A,r.inv,(i,j)=>`(${i+1},${j+1})`)],false,'wl');
  }
  h+=`<p class="${r.ok?'ok':'bad'}">${r.ok?'✔ Both products equal the identity matrix.':'✘ A product is not the identity. '+(r.cond>1e8?'The matrix is ill-conditioned, so decimal rounding error is too large to confirm it.':'Recheck your entries.')}</p></section>`;
  h+=`<section class="card${n>5?' span':''}"><h2><span class="n">4</span>Solve&nbsp; [X] = A<sup>-1</sup>B</h2>
   <div class="row">${lab('A<sup>-1</sup>')}${bm(r,f,r.inv,'A⁻¹')}${lab('B')}${mat(col(r.B),f)}${lab('=')}${mat(r.x,f)}</div>
   <div class="ans${n>10?' dense':''}">${r.x.map((v,i)=>`<div>${xn(i,n)} = ${f(v[0])}${r.mode==='exact'?`<small>${dec(v[0])||'exact'}</small>`:''}</div>`).join('')}</div>`;
  if(r.mode==='exact') h+=`${lazy('Show how each x is computed: x = A<sup>-1</sup>B',()=>`<ul class="wl">${dotWork(r.inv,col(r.B),f,i=>xn(i,n))}</ul>`,n<=4)}`;
  if(r.mode==='float') h+=`<p class="mut">Largest error of Ax − B: ${sci(r.res)}</p>`;
  return h+'</section>'+verifyHTML(r);
}

/* ---------- bridge to the FastAPI server (mc7_api.py) ---------- */
// ---------- bridge to the Python API (mc7_api.py) ----------
// API: server address. Empty (same server) when the page is served by mc7_api.py; otherwise http://127.0.0.1:8000.
const API=location.protocol.startsWith('http')?'':'http://127.0.0.1:8000';
// seq counts requests so a slow, old reply can never overwrite a newer one.
// ctl cancels the previous request. ctm is the typing delay timer (120 ms).
let seq=0,ctl=null,ctm;
// setApi(on): updates the status pill: "Python API connected" or "offline".
const setApi=on=>{const p=$('api');p.dataset.on=String(on);p.textContent=on?'Python API connected':'API offline · using the built-in browser solver'};
// callAPI(): sends A and B to /api/solve. A 422 reply means "the API says the input is invalid"
// (shown to the user). Any other failure means "use the built-in solver".
async function callAPI(){
  if(ctl)ctl.abort(); ctl=new AbortController();
  const r=await fetch(API+'/api/solve',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({A,B,unlimited:UNLOCK}),signal:ctl.signal});
  if(r.status===422){const j=await r.json().catch(()=>({}));const e=new Error(typeof j.detail==='string'?j.detail:'The API could not read this system.');e.api=true;throw e}
  if(!r.ok) throw new Error('API returned '+r.status); /* 404/405/5xx: not our server, so fall back */
  return r.json();
}
// fromAPI(j): turns the server's text answer back into the numbers/fractions that report() understands.
function fromAPI(j){
  const exact=j.mode==='exact', P=exact?parseX:(v=>v), PI=exact?parseX:parseF, cv=M=>M&&M.map(r=>r.map(P));
  const r={...j,A:A.map(r=>r.map(PI)),B:B.map(PI),inv:cv(j.inv),left:cv(j.left),right:cv(j.right),x:cv(j.x),steps:j.steps};
  if(exact){r.det=parseX(j.det);r.pivots=j.pivots&&j.pivots.map(parseX)}
  if(j.ver) r.ver={...j.ver,rows:j.ver.rows.map(q=>({ok:q.ok,l:exact?str(parseX(q.l)):fmtF(q.l),b:exact?str(parseX(q.b)):fmtF(q.b)}))};
  if(exact&&r.ver&&r.x) r.ver.rows.forEach((q,i)=>{q.w=r.A[i].map((v,k)=>`(${str(v)})(${str(r.x[k][0])})`).join(' + ')});
  return r;
}
// randomFrom(n): asks the API for a random system (any size when n is empty).
// Falls back to randomSys if the API is offline.
async function randomFrom(n){
  try{const r=await fetch(API+'/api/random?'+(n?'n='+n+'&':'')+'unlimited='+UNLOCK);if(!r.ok)throw 0;const j=await r.json();load({a:j.a,b:j.b})}
  catch(e){load(randomSys(n||2+Math.floor(Math.random()*((UNLOCK?60:MAX_N)-1))))}
}

/* ---------- input handling ---------- */
// ---------- input handling ----------
// parseText(t): reads a pasted block (one equation per line) and works out n from the number of lines.
function parseText(t){
  const rows=t.split(/\r?\n/).map(l=>l.trim()).filter(Boolean).map(l=>l.split(/[\s,;]+/).filter(Boolean)), m=rows.length;
  if(!m) return{err:'Type or paste at least one equation.'};
  if(m>LIMIT()) return{err:`That is ${m} equations. The maximum is ${LIMIT()}${UNLOCK?'':' (tick "Remove the 24×24 limit" for more)'}.`};
  for(let i=0;i<m;i++) if(rows[i].length!==m+1)
    return{err:`Line ${i+1} has ${rows[i].length} numbers. A ${m}×${m} system needs ${m+1} per line: the coefficients, then the constant.`};
  return{A:rows.map(r=>r.slice(0,m)),B:rows.map(r=>r[m])};
}
// randomSys(n): the browser copy of random_system in Python (same recipe), used when the API is offline.
function randomSys(n){
  const R=(lo,hi)=>lo+Math.floor(Math.random()*(hi-lo+1)),a=[],x=Array.from({length:n},()=>R(-9,9));
  if(n<=EXACT_MAX){ /* A = L·U with unit triangles: det = 1, so A⁻¹ and x stay whole numbers */
    const Z=()=>Array.from({length:n},()=>new Array(n).fill(0)),L=Z(),U=Z();
    for(let i=0;i<n;i++){L[i][i]=U[i][i]=1;for(let j=0;j<i;j++){L[i][j]=R(-1,1);U[j][i]=R(-1,1)}}
    for(let i=0;i<n;i++){a.push([]);for(let j=0;j<n;j++){let s=0;for(let k=0;k<n;k++)s+=L[i][k]*U[k][j];a[i].push(s)}}
  }else for(let i=0;i<n;i++){let s=0;const r=[];for(let j=0;j<n;j++){const v=R(-9,9);r.push(v);if(j!==i)s+=Math.abs(v)}r[i]=s+1+R(0,5);a.push(r)}
  return{a,b:a.map(r=>r.reduce((s,v,j)=>s+v*x[j],0))};
}
// PRESETS: the sample buttons (3x3 sample, 2x2 sample, singular example).
const PRESETS={
  sample:{a:[[2,1,-1],[-3,-1,2],[-2,1,2]],b:[8,-11,-3]},
  two:{a:[[2,1],[5,3]],b:[3,8]},
  sing:{a:[[1,2],[2,4]],b:[3,6]}
};

// STATE: N = size, A = the coefficient texts, B = the constant texts, tmr = text-box delay timer.
let N=3,A=[],B=[],tmr;
// $(id): shortcut for document.getElementById.
const $=id=>document.getElementById(id);
// setSize(n): changes the size and keeps the numbers already typed (new cells start at 0).
function setSize(n){
  const nA=Array.from({length:n},(_,i)=>Array.from({length:n},(_,j)=>(A[i]&&A[i][j]!==undefined)?A[i][j]:'0'));
  const nB=Array.from({length:n},(_,i)=>B[i]!==undefined?B[i]:'0');
  N=n;A=nA;B=nB;build();
}
// load(p): puts a preset or random system into the grid.
function load(p){N=p.a.length;A=p.a.map(r=>r.map(String));B=p.b.map(String);build()}
// syncText(): keeps the optional text box in step with the grid.
function syncText(){$('txt').value=A.map((r,i)=>r.concat(B[i]).join(' ')).join('\n')}
// esc(s): makes typed text safe to place inside HTML (quotes and angle brackets), so it cannot break the page.
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
// gridHTML(): draws the compact input grid used above EQ_STYLE.
const gridHTML=()=>`<table class="cg"><thead><tr>${A.map((_,j)=>`<th>${xn(j)}</th>`).join('')}<th></th><th>B</th></tr></thead><tbody>${A.map((row,i)=>'<tr>'+row.map((v,j)=>`<td><input inputmode="decimal" aria-label="Row ${i+1}, coefficient of ${xn(j)}" data-r="${i}" data-c="${j}" value="${esc(v)}"></td>`).join('')+`<td class="lab">=</td><td><input inputmode="decimal" aria-label="Row ${i+1}, constant" data-r="${i}" data-c="b" value="${esc(B[i])}"></td></tr>`).join('')}</tbody></table>`;
// build(skipText): redraws size buttons, mode label and grid for the current state, then calls compute().
function build(skipText){
  $('size').value=N;
  $('layout').classList.toggle('side',N<=EQ_STYLE);
  $('size').max=LIMIT(); $('unlock').checked=UNLOCK;
  document.querySelectorAll('#chips .big').forEach(b=>{b.hidden=!UNLOCK});
  $('solveBar').hidden=!(N>MAX_N);
  document.querySelectorAll('#chips button').forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.n===N)));
  $('mode').textContent=N<=EXACT_MAX?'Exact fractions':'Decimal, double precision';
  $('form').innerHTML=N>GRID_MAX?`<p class="hint">The grid is available up to ${GRID_MAX}×${GRID_MAX}. For ${N}×${N}, paste the system into the text box below, or press Randomize.</p>`:N>EQ_STYLE?gridHTML():A.map((row,i)=>'<div class="eq">'+row.map((v,j)=>
    `<input inputmode="decimal" aria-label="Row ${i+1}, coefficient of ${xn(j)}" data-r="${i}" data-c="${j}" value="${esc(v)}">${`<span class="lab">${xn(j)}</span>`}${j<N-1?'<span class="op">+</span>':''}`).join('')+
    `<span class="op">=</span><input inputmode="decimal" aria-label="Row ${i+1}, constant" data-r="${i}" data-c="b" value="${esc(B[i])}"></div>`).join('');
  if(N>GRID_MAX) $('textBox').open=true;
  if(!skipText) syncText();
  compute();
}
// compute(): the main routine. Checks every cell, asks the API, falls back to the built-in solver
// if the API is unreachable, and draws the report only if this is still the newest request.
async function compute(force){
  const exact=N<=EXACT_MAX, P=exact?parseX:parseF; let bad=null;
  const pa=A.map((r,i)=>r.map((s,j)=>{const v=P(s);if(v===null&&!bad)bad=`Row ${i+1}, column ${j+1}`;return v}));
  const pb=B.map((s,i)=>{const v=P(s);if(v===null&&!bad)bad=`Row ${i+1}, constant`;return v});
  document.querySelectorAll('#form input').forEach(el=>{
    const v=el.dataset.c==='b'?pb[+el.dataset.r]:pa[+el.dataset.r][+el.dataset.c];el.setAttribute('aria-invalid',String(v===null))});
  $('err').textContent=bad?`${bad} is not a number. Use values like 3, -2, 1.5 or 3/4 (up to 40 characters).`:'';
  if(bad){seq++;if(ctl)ctl.abort();$('out').innerHTML='';return}
  if(N<=MAX_N) $('solveHint').textContent='';
  else if(!force){seq++;if(ctl)ctl.abort();$('out').innerHTML='';$('solveHint').textContent=`${N}×${N} is a large system: press Solve to compute it. This can take a few seconds.`;return}
  else{$('out').innerHTML='<p class="mut">Solving…</p>';$('solveBtn').disabled=true;$('solveHint').textContent=''}
  const mine=++seq;
  try{
    let r;
    try{const j=await callAPI();if(mine!==seq)return;r=fromAPI(j);setApi(true)}
    catch(e){if(e.name==='AbortError')return;if(e.api){$('err').textContent=e.message;$('out').innerHTML='';return}
      setApi(false);r=exact?solveExact(pa,pb):solveFloat(pa,pb)}
    if(mine===seq){$('out').innerHTML=report(r);$('out').querySelectorAll('details[data-lz][open]').forEach(fillLazy)}
  }catch(e){$('err').textContent='Something went wrong: '+e.message}
  finally{$('solveBtn').disabled=false}
}
// fromText(): applies a pasted text block to the grid.
function fromText(){
  const r=parseText($('txt').value);
  if(r.err){$('err').textContent=r.err;$('out').innerHTML='';return}
  N=r.A.length;A=r.A;B=r.B;build(true);
}
// askConfirm(title,bullets,okText): shows the warning popup. Resolves true for the orange button, false for Cancel,
// the Escape key or a click outside the box.
function askConfirm(title,bullets,okText){
  return new Promise(res=>{
    $('mTitle').textContent=title; $('mList').innerHTML=bullets.map(b=>`<li>${b}</li>`).join(''); $('mOk').textContent=okText;
    const m=$('modal'); m.hidden=false; $('mCancel').focus();
    const key=e=>{if(e.key==='Escape')done(false)};
    const done=v=>{m.hidden=true;$('mOk').onclick=$('mCancel').onclick=m.onclick=null;document.removeEventListener('keydown',key);res(v)};
    document.addEventListener('keydown',key);
    $('mOk').onclick=()=>done(true); $('mCancel').onclick=()=>done(false); m.onclick=e=>{if(e.target===m)done(false)};
  });
}
// EVENT WIRING (browser only): size box and buttons, presets, typing in cells (with a short delay),
// pasting a block into the grid, the text box, and opening the lazy sections.
if(typeof document!=='undefined'){
  $('chips').addEventListener('click',e=>{const b=e.target.closest('button');if(b)setSize(+b.dataset.n)});
  $('size').addEventListener('change',()=>{
    let v=parseInt($('size').value,10); if(!(v>=1)) v=N;
    $('sizeNote').textContent=v>LIMIT()?`The maximum is ${LIMIT()}×${LIMIT()}${UNLOCK?'':' (tick "Remove the 24×24 limit" for more)'}.`:''; setSize(Math.min(v,LIMIT()));
  });
  $('presets').addEventListener('click',e=>{const p=e.target.dataset.p;if(!p)return;
    if(p==='randall'){randomFrom()}
    else if(p==='rand'){randomFrom(N<2?3:N)}
    else if(p==='clear'){A=A.map(r=>r.map(()=>'0'));B=B.map(()=>'0');build()}
    else load(PRESETS[p]);$('sizeNote').textContent=''});
  $('form').addEventListener('input',e=>{const el=e.target,i=+el.dataset.r;if(el.dataset.r===undefined)return;
    if(el.dataset.c==='b')B[i]=el.value;else A[i][+el.dataset.c]=el.value;clearTimeout(ctm);ctm=setTimeout(()=>{syncText();compute()},120)});
  $('txt').addEventListener('input',()=>{clearTimeout(tmr);tmr=setTimeout(fromText,N>EXACT_MAX?250:0)});
  $('form').addEventListener('paste',e=>{
    const el=e.target,t=(e.clipboardData||window.clipboardData).getData('text');
    const rows=t.split(/\r?\n/).filter(l=>l.trim()).map(l=>l.trim().split(/[\s,;]+/));
    if(el.dataset.r===undefined||!rows.length||(rows.length<2&&rows[0].length<2))return;
    e.preventDefault();
    const r0=+el.dataset.r,c0=el.dataset.c==='b'?N:+el.dataset.c;
    rows.forEach((row,di)=>row.forEach((v,dj)=>{const i=r0+di,j=c0+dj;if(i<N&&j<N)A[i][j]=v;else if(i<N&&j===N)B[i]=v}));
    build();
  });
  $('solveBtn').addEventListener('click',()=>compute(true));
  // The "Remove the size limit" box: ticking it shows the warning first; unticking asks before cutting a big system down.
  $('unlock').addEventListener('change',async()=>{
    const box=$('unlock');
    if(box.checked){
      const ok=await askConfirm('Remove the size limit?',[
        '<b>The page can slow down or freeze.</b> A 100×100 system has 10,000 numbers and a 200×200 system has 40,000.',
        '<b>Solving takes a few seconds.</b> About 1 second for 100×100 and 2 to 10 seconds for 200×200, depending on your computer. Large systems are solved when you press <b>Solve</b>, not while you type.',
        '<b>Less working is shown.</b> Above 40×40 you see the pivot choices instead of a table for every step, and big matrices stay collapsed until you open them.',
        '<b>Decimals only.</b> Rounding errors can build up, and the page warns you about matrices that are hard to invert.',
        'The largest size is 200×200.'],'Remove the limit');
      if(!ok){box.checked=false;return}
      UNLOCK=true;
    }else{
      if(N>MAX_N&&!(await askConfirm(`Go back to ${MAX_N}×${MAX_N}?`,[`Your ${N}×${N} system will be cut down to ${MAX_N}×${MAX_N}: the extra rows and columns are removed.`],'Cut it down'))){box.checked=true;return}
      UNLOCK=false;
    }
    if(N>LIMIT()) setSize(LIMIT()); else build();
  });
  $('out').addEventListener('toggle',e=>{if(e.target.dataset&&e.target.dataset.lz)fillLazy(e.target)},true);
  load(PRESETS.sample);
}