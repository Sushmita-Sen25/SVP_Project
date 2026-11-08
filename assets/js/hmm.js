// Left-to-right (Bakis) continuous-density HMM with diagonal-covariance
// Gaussian emissions. Each state may stay, advance one state, or skip one.
// Training: flat start → Viterbi (segmental k-means) → Baum–Welch EM.
// Everything runs in the log domain. No DOM dependencies (works in Node too).

const LOG2PI = Math.log(2 * Math.PI);
const NEG_INF = -Infinity;

function lse(a, b) {
  if (a === NEG_INF) return b;
  if (b === NEG_INF) return a;
  return a > b ? a + Math.log1p(Math.exp(b - a)) : b + Math.log1p(Math.exp(a - b));
}

export class LeftRightHMM {
  constructor(N, dim) {
    this.N = N;
    this.dim = dim;
    this.means = Array.from({ length: N }, () => new Float64Array(dim));
    this.vars = Array.from({ length: N }, () => new Float64Array(dim).fill(1));
    this.logTrans = Array.from({ length: N }, (_, s) => {
      const t = new Float64Array(3).fill(NEG_INF);
      const valid = Math.min(3, N - s);
      for (let k = 0; k < valid; k++) t[k] = Math.log(k === 0 ? 0.6 : k === 1 ? 0.35 : 0.05);
      return t;
    });
    this.normalizeTransitions();
    this.updateCache();
  }

  normalizeTransitions() {
    for (const t of this.logTrans) {
      let z = NEG_INF;
      for (let k = 0; k < 3; k++) z = lse(z, t[k]);
      for (let k = 0; k < 3; k++) if (t[k] !== NEG_INF) t[k] -= z;
    }
  }

  updateCache() {
    this.gconst = new Float64Array(this.N);
    this.invVar = this.vars.map((v) => Float64Array.from(v, (x) => 1 / x));
    for (let s = 0; s < this.N; s++) {
      let l = 0;
      for (let k = 0; k < this.dim; k++) l += Math.log(this.vars[s][k]);
      this.gconst[s] = -0.5 * (this.dim * LOG2PI + l);
    }
  }

  logEmission(x, s) {
    const mu = this.means[s];
    const iv = this.invVar[s];
    let q = 0;
    for (let k = 0; k < this.dim; k++) {
      const d = x[k] - mu[k];
      q += d * d * iv[k];
    }
    return this.gconst[s] - 0.5 * q;
  }

  emissionMatrix(obs) {
    const T = obs.length, N = this.N;
    const B = new Float64Array(T * N);
    for (let t = 0; t < T; t++) for (let s = 0; s < N; s++) B[t * N + s] = this.logEmission(obs[t], s);
    return B;
  }

  /** log P(O | λ) by the forward algorithm. */
  forward(obs, { endAnyState = false, B = null } = {}) {
    const T = obs.length, N = this.N;
    B = B || this.emissionMatrix(obs);
    const alpha = new Float64Array(T * N).fill(NEG_INF);
    alpha[0] = B[0];
    for (let t = 1; t < T; t++) {
      const p = (t - 1) * N, c = t * N;
      for (let j = 0; j < N; j++) {
        let v = NEG_INF;
        for (let k = 0; k < 3 && j - k >= 0; k++) {
          const a = alpha[p + j - k];
          if (a !== NEG_INF) v = lse(v, a + this.logTrans[j - k][k]);
        }
        alpha[c + j] = v === NEG_INF ? NEG_INF : v + B[c + j];
      }
    }
    let ll = NEG_INF;
    const last = (T - 1) * N;
    if (endAnyState) for (let j = 0; j < N; j++) ll = lse(ll, alpha[last + j]);
    else ll = alpha[last + N - 1];
    return { logLik: ll, alpha, B };
  }

  backward(B, T) {
    const N = this.N;
    const beta = new Float64Array(T * N).fill(NEG_INF);
    beta[(T - 1) * N + N - 1] = 0;
    for (let t = T - 2; t >= 0; t--) {
      const c = t * N, nx = (t + 1) * N;
      for (let i = 0; i < N; i++) {
        let v = NEG_INF;
        for (let k = 0; k < 3 && i + k < N; k++) {
          const b = beta[nx + i + k];
          if (b !== NEG_INF) v = lse(v, this.logTrans[i][k] + B[nx + i + k] + b);
        }
        beta[c + i] = v;
      }
    }
    return beta;
  }

  /** Most likely state sequence. */
  viterbi(obs, { endAnyState = false, B = null } = {}) {
    const T = obs.length, N = this.N;
    B = B || this.emissionMatrix(obs);
    const delta = new Float64Array(T * N).fill(NEG_INF);
    const psi = new Int32Array(T * N);
    delta[0] = B[0];
    for (let t = 1; t < T; t++) {
      const p = (t - 1) * N, c = t * N;
      for (let j = 0; j < N; j++) {
        let best = NEG_INF, arg = j;
        for (let k = 0; k < 3 && j - k >= 0; k++) {
          const v = delta[p + j - k] + this.logTrans[j - k][k];
          if (v > best) { best = v; arg = j - k; }
        }
        delta[c + j] = best === NEG_INF ? NEG_INF : best + B[c + j];
        psi[c + j] = arg;
      }
    }
    const last = (T - 1) * N;
    let end = N - 1;
    if (endAnyState) for (let j = 0; j < N; j++) if (delta[last + j] > delta[last + end]) end = j;
    const logLik = delta[last + end];
    const path = new Int32Array(T);
    path[T - 1] = end;
    for (let t = T - 1; t > 0; t--) path[t - 1] = psi[t * N + path[t]];
    return { logLik, path };
  }

  toJSON() {
    const r = (x) => Number(x.toPrecision(5));
    return {
      N: this.N,
      dim: this.dim,
      means: this.means.map((m) => Array.from(m, r)),
      vars: this.vars.map((v) => Array.from(v, r)),
      logTrans: this.logTrans.map((t) => Array.from(t, (x) => (x === NEG_INF ? null : r(x)))),
    };
  }

  static fromJSON(j) {
    const h = new LeftRightHMM(j.N, j.dim);
    h.means = j.means.map((m) => Float64Array.from(m));
    h.vars = j.vars.map((v) => Float64Array.from(v));
    h.logTrans = j.logTrans.map((t) => Float64Array.from(t, (x) => (x === null ? NEG_INF : x)));
    h.updateCache();
    return h;
  }
}

function globalVariance(seqs, dim) {
  const mean = new Float64Array(dim), sq = new Float64Array(dim);
  let n = 0;
  for (const s of seqs) for (const x of s) {
    n++;
    for (let k = 0; k < dim; k++) { mean[k] += x[k]; sq[k] += x[k] * x[k]; }
  }
  return Float64Array.from(mean, (m, k) => Math.max(sq[k] / n - (m / n) ** 2, 1e-6));
}

function reestimateFromAlignment(model, seqs, paths, floor) {
  const { N, dim } = model;
  const cnt = new Float64Array(N);
  const sx = Array.from({ length: N }, () => new Float64Array(dim));
  const sxx = Array.from({ length: N }, () => new Float64Array(dim));
  const trans = Array.from({ length: N }, () => new Float64Array(3));
  seqs.forEach((seq, n) => {
    const p = paths[n];
    for (let t = 0; t < seq.length; t++) {
      const s = p[t];
      cnt[s]++;
      const x = seq[t];
      for (let k = 0; k < dim; k++) { sx[s][k] += x[k]; sxx[s][k] += x[k] * x[k]; }
      if (t > 0) trans[p[t - 1]][Math.min(2, Math.max(0, s - p[t - 1]))]++;
    }
  });
  applyStats(model, cnt, sx, sxx, trans, floor);
}

function applyStats(model, cnt, sx, sxx, trans, floor) {
  const { N, dim } = model;
  for (let s = 0; s < N; s++) {
    if (cnt[s] < 1e-3) continue; // keep previous parameters for an empty state
    for (let k = 0; k < dim; k++) {
      const m = sx[s][k] / cnt[s];
      model.means[s][k] = m;
      model.vars[s][k] = Math.max(sxx[s][k] / cnt[s] - m * m, floor[k]);
    }
    const valid = Math.min(3, N - s);
    let tot = 0;
    for (let k = 0; k < valid; k++) tot += trans[s][k] + 0.1;
    for (let k = 0; k < 3; k++) model.logTrans[s][k] = k < valid ? Math.log((trans[s][k] + 0.1) / tot) : NEG_INF;
  }
  model.updateCache();
}

/**
 * Train a left-to-right HMM on a list of observation sequences.
 * @returns {{model: LeftRightHMM, history: {iter:number, stage:string, avgLogLik:number}[]}}
 */
export function trainHMM(seqs, { nStates, viterbiIters = 4, emIters = 6, varFloorRatio = 0.05, onIter } = {}) {
  const dim = seqs[0][0].length;
  const avgT = seqs.reduce((a, s) => a + s.length, 0) / seqs.length;
  const N = nStates || Math.max(12, Math.min(60, Math.round(avgT / 20)));
  const model = new LeftRightHMM(N, dim);
  const gv = globalVariance(seqs, dim);
  const floor = Float64Array.from(gv, (v) => v * varFloorRatio);
  const history = [];
  const totalFrames = seqs.reduce((a, s) => a + s.length, 0);

  // flat start: uniform segmentation
  const flat = seqs.map((seq) => Int32Array.from({ length: seq.length }, (_, t) => Math.min(N - 1, Math.floor((t * N) / seq.length))));
  reestimateFromAlignment(model, seqs, flat, floor);

  for (let it = 0; it < viterbiIters; it++) {
    let ll = 0;
    const paths = seqs.map((seq) => {
      const v = model.viterbi(seq);
      ll += v.logLik;
      return v.path;
    });
    reestimateFromAlignment(model, seqs, paths, floor);
    const h = { iter: history.length + 1, stage: 'viterbi', avgLogLik: ll / totalFrames };
    history.push(h);
    onIter && onIter(h);
  }

  for (let it = 0; it < emIters; it++) {
    const cnt = new Float64Array(N);
    const sx = Array.from({ length: N }, () => new Float64Array(dim));
    const sxx = Array.from({ length: N }, () => new Float64Array(dim));
    const trans = Array.from({ length: N }, () => new Float64Array(3));
    let ll = 0;
    for (const seq of seqs) {
      const T = seq.length;
      const { logLik, alpha, B } = model.forward(seq);
      if (!Number.isFinite(logLik)) continue;
      ll += logLik;
      const beta = model.backward(B, T);
      for (let t = 0; t < T; t++) {
        const c = t * N;
        const x = seq[t];
        for (let s = 0; s < N; s++) {
          const lg = alpha[c + s] + beta[c + s] - logLik;
          if (lg < -30) continue;
          const g = Math.exp(lg);
          cnt[s] += g;
          for (let k = 0; k < dim; k++) { sx[s][k] += g * x[k]; sxx[s][k] += g * x[k] * x[k]; }
        }
        if (t < T - 1) {
          const nx = (t + 1) * N;
          for (let s = 0; s < N; s++) {
            const a = alpha[c + s];
            if (a === NEG_INF) continue;
            for (let k = 0; k < 3 && s + k < N; k++) {
              const lx = a + model.logTrans[s][k] + B[nx + s + k] + beta[nx + s + k] - logLik;
              if (lx > -30) trans[s][k] += Math.exp(lx);
            }
          }
        }
      }
    }
    applyStats(model, cnt, sx, sxx, trans, floor);
    const h = { iter: history.length + 1, stage: 'baum-welch', avgLogLik: ll / totalFrames };
    history.push(h);
    onIter && onIter(h);
  }
  return { model, history };
}

/** Score an utterance against every model; best first. */
export function recognize(models, obs, { endAnyState = false } = {}) {
  const T = obs.length;
  const scores = Object.entries(models).map(([id, m]) => {
    const { logLik } = m.forward(obs, { endAnyState });
    return { id, logLik, perFrame: logLik / T };
  });
  scores.sort((a, b) => b.logLik - a.logLik);
  // relative posterior with a temperature so the numbers stay readable
  const best = scores[0].perFrame;
  const z = scores.reduce((a, s) => a + Math.exp((s.perFrame - best) * 3), 0);
  for (const s of scores) s.posterior = Math.exp((s.perFrame - best) * 3) / z;
  return scores;
}
