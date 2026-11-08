// Small dense linear-algebra helpers (symmetric eigen-decomposition, Cholesky)
// used to learn and apply the LDA feature transform.

/** Jacobi eigenvalue algorithm for a symmetric matrix (array of rows). */
export function eigSym(A, { maxSweeps = 100, eps = 1e-12 } = {}) {
  const n = A.length;
  const a = A.map((r) => Float64Array.from(r));
  const V = Array.from({ length: n }, (_, i) => Float64Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    let off = 0;
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += a[i][j] * a[i][j];
    if (off < eps) break;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        if (Math.abs(a[p][q]) < 1e-300) continue;
        const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < n; k++) {
          const akp = a[k][p], akq = a[k][q];
          a[k][p] = c * akp - s * akq;
          a[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = a[p][k], aqk = a[q][k];
          a[p][k] = c * apk - s * aqk;
          a[q][k] = s * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = V[k][p], vkq = V[k][q];
          V[k][p] = c * vkp - s * vkq;
          V[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }
  const values = a.map((r, i) => r[i]);
  const order = values.map((v, i) => i).sort((x, y) => values[y] - values[x]);
  return {
    values: order.map((i) => values[i]),
    vectors: order.map((i) => Float64Array.from(V, (row) => row[i])), // eigenvectors as rows
  };
}

export function cholesky(A) {
  const n = A.length;
  const L = Array.from({ length: n }, () => new Float64Array(n));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = A[i][j];
      for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
      if (i === j) L[i][i] = Math.sqrt(Math.max(s, 1e-10));
      else L[i][j] = s / L[j][j];
    }
  }
  return L;
}

/** Solve L x = b for lower-triangular L. */
function forwardSub(L, b) {
  const n = b.length;
  const x = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = b[i];
    for (let k = 0; k < i; k++) s -= L[i][k] * x[k];
    x[i] = s / L[i][i];
  }
  return x;
}

/** Solve L^T x = b for lower-triangular L. */
function backSubT(L, b) {
  const n = b.length;
  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let s = b[i];
    for (let k = i + 1; k < n; k++) s -= L[k][i] * x[k];
    x[i] = s / L[i][i];
  }
  return x;
}

/**
 * Linear discriminant analysis. frames: Float32Array[]; labels: int[].
 * Returns { mean, W (rows = projection vectors), values }.
 */
export function lda(frames, labels, outDim, { shrink = 1e-3 } = {}) {
  const d = frames[0].length;
  const N = frames.length;
  const mean = new Float64Array(d);
  for (const f of frames) for (let k = 0; k < d; k++) mean[k] += f[k] / N;
  const classes = new Map();
  frames.forEach((f, i) => {
    let c = classes.get(labels[i]);
    if (!c) classes.set(labels[i], (c = { n: 0, sum: new Float64Array(d) }));
    c.n++;
    for (let k = 0; k < d; k++) c.sum[k] += f[k];
  });
  const Sw = Array.from({ length: d }, () => new Float64Array(d));
  const Sb = Array.from({ length: d }, () => new Float64Array(d));
  for (const c of classes.values()) {
    c.mu = c.sum.map((s) => s / c.n);
    for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) Sb[i][j] += (c.n * (c.mu[i] - mean[i]) * (c.mu[j] - mean[j])) / N;
  }
  frames.forEach((f, idx) => {
    const mu = classes.get(labels[idx]).mu;
    for (let i = 0; i < d; i++) {
      const di = f[i] - mu[i];
      for (let j = i; j < d; j++) Sw[i][j] += (di * (f[j] - mu[j])) / N;
    }
  });
  let tr = 0;
  for (let i = 0; i < d; i++) { for (let j = 0; j < i; j++) Sw[i][j] = Sw[j][i]; tr += Sw[i][i]; }
  for (let i = 0; i < d; i++) Sw[i][i] += (shrink * tr) / d;
  // Sb v = λ Sw v  →  (L^-1 Sb L^-T) u = λ u,  v = L^-T u
  const L = cholesky(Sw);
  const Y = Sb.map((row) => forwardSub(L, row)); // rows of L^-1 Sb^T  (Sb symmetric)
  const M = Array.from({ length: d }, () => new Float64Array(d));
  const cols = Array.from({ length: d }, (_, j) => forwardSub(L, Float64Array.from(Y, (r) => r[j])));
  for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) M[i][j] = cols[j][i];
  for (let i = 0; i < d; i++) for (let j = i + 1; j < d; j++) { const s = (M[i][j] + M[j][i]) / 2; M[i][j] = M[j][i] = s; }
  const { values, vectors } = eigSym(M);
  const W = vectors.slice(0, outDim).map((u) => backSubT(L, u));
  return { mean: Array.from(mean), W: W.map((w) => Array.from(w)), values: values.slice(0, outDim) };
}

export function applyLinear(frames, { mean, W }) {
  const d = mean.length, k = W.length;
  return frames.map((f) => {
    const o = new Float32Array(k);
    for (let r = 0; r < k; r++) {
      const w = W[r];
      let s = 0;
      for (let j = 0; j < d; j++) s += w[j] * (f[j] - mean[j]);
      o[r] = s;
    }
    return o;
  });
}
