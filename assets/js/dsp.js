// Core speech DSP: FFT, STFT, mel filterbank, MFCC, deltas, CMVN, spectral
// descriptors, YIN pitch and silence trimming. Pure functions with no DOM
// dependencies so the same code runs in the browser and in Node (training).

export const SR = 16000;

export const DEFAULTS = Object.freeze({
  sr: SR,
  frameLen: 400, // 25 ms
  hop: 160, // 10 ms
  nfft: 512,
  nMels: 26,
  nCeps: 13,
  preEmph: 0.97,
  lifter: 22,
  fmin: 20,
  fmax: 8000,
});

// ---------------------------------------------------------------- FFT ----
const twiddleCache = new Map();
function twiddles(n) {
  let t = twiddleCache.get(n);
  if (!t) {
    const cos = new Float64Array(n / 2);
    const sin = new Float64Array(n / 2);
    for (let i = 0; i < n / 2; i++) {
      cos[i] = Math.cos((2 * Math.PI * i) / n);
      sin[i] = -Math.sin((2 * Math.PI * i) / n);
    }
    t = { cos, sin };
    twiddleCache.set(n, t);
  }
  return t;
}

/** In-place iterative radix-2 FFT. re/im are Float64Arrays of power-of-two length. */
export function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      let t = re[i]; re[i] = re[j]; re[j] = t;
      t = im[i]; im[i] = im[j]; im[j] = t;
    }
  }
  const { cos, sin } = twiddles(n);
  for (let size = 2; size <= n; size <<= 1) {
    const half = size >> 1;
    const step = n / size;
    for (let start = 0; start < n; start += size) {
      for (let k = 0; k < half; k++) {
        const wr = cos[k * step];
        const wi = sin[k * step];
        const a = start + k;
        const b = a + half;
        const xr = re[b] * wr - im[b] * wi;
        const xi = re[b] * wi + im[b] * wr;
        re[b] = re[a] - xr; im[b] = im[a] - xi;
        re[a] += xr; im[a] += xi;
      }
    }
  }
}

const windowCache = new Map();
export function hamming(n) {
  let w = windowCache.get(n);
  if (!w) {
    w = new Float64Array(n);
    for (let i = 0; i < n; i++) w[i] = 0.54 - 0.46 * Math.cos((2 * Math.PI * i) / (n - 1));
    windowCache.set(n, w);
  }
  return w;
}

export function numFrames(len, frameLen, hop) {
  return len < frameLen ? 1 : 1 + Math.floor((len - frameLen) / hop);
}

export function preEmphasis(x, a = 0.97) {
  const y = new Float32Array(x.length);
  y[0] = x[0] || 0;
  for (let i = 1; i < x.length; i++) y[i] = x[i] - a * x[i - 1];
  return y;
}

/** Power spectrogram: array of Float32Array(nfft/2+1), one per frame. */
export function powerSpectrogram(signal, opts = {}) {
  const o = { ...DEFAULTS, ...opts };
  const x = o.preEmph ? preEmphasis(signal, o.preEmph) : signal;
  const T = numFrames(x.length, o.frameLen, o.hop);
  const win = hamming(o.frameLen);
  const bins = o.nfft / 2 + 1;
  const re = new Float64Array(o.nfft);
  const im = new Float64Array(o.nfft);
  const out = new Array(T);
  for (let t = 0; t < T; t++) {
    re.fill(0); im.fill(0);
    const off = t * o.hop;
    for (let i = 0; i < o.frameLen; i++) re[i] = (x[off + i] || 0) * win[i];
    fft(re, im);
    const p = new Float32Array(bins);
    for (let k = 0; k < bins; k++) p[k] = (re[k] * re[k] + im[k] * im[k]) / o.nfft;
    out[t] = p;
  }
  return out;
}

// ------------------------------------------------------------- Mel/MFCC ---
const hz2mel = (f) => 2595 * Math.log10(1 + f / 700);
const mel2hz = (m) => 700 * (10 ** (m / 2595) - 1);

const melCache = new Map();
export function melFilterbank(nMels, nfft, sr, fmin, fmax) {
  const key = `${nMels}|${nfft}|${sr}|${fmin}|${fmax}`;
  let fb = melCache.get(key);
  if (fb) return fb;
  const bins = nfft / 2 + 1;
  const mMin = hz2mel(fmin);
  const mMax = hz2mel(Math.min(fmax, sr / 2));
  const pts = [];
  for (let i = 0; i < nMels + 2; i++) pts.push(mel2hz(mMin + ((mMax - mMin) * i) / (nMels + 1)));
  const binHz = sr / nfft;
  fb = [];
  for (let m = 0; m < nMels; m++) {
    const [l, c, r] = [pts[m], pts[m + 1], pts[m + 2]];
    const start = Math.max(0, Math.floor(l / binHz));
    const end = Math.min(bins - 1, Math.ceil(r / binHz));
    const w = new Float32Array(end - start + 1);
    for (let k = start; k <= end; k++) {
      const f = k * binHz;
      let v = 0;
      if (f > l && f <= c) v = (f - l) / (c - l);
      else if (f > c && f < r) v = (r - f) / (r - c);
      w[k - start] = v;
    }
    fb.push({ start, w, center: c });
  }
  melCache.set(key, fb);
  return fb;
}

export function melEnergies(spec, o) {
  const fb = melFilterbank(o.nMels, o.nfft, o.sr, o.fmin, o.fmax);
  return spec.map((p) => {
    const e = new Float32Array(o.nMels);
    for (let m = 0; m < fb.length; m++) {
      const { start, w } = fb[m];
      let s = 0;
      for (let k = 0; k < w.length; k++) s += w[k] * p[start + k];
      e[m] = Math.log(Math.max(s, 1e-10));
    }
    return e;
  });
}

const dctCache = new Map();
function dctMatrix(nCeps, nMels) {
  const key = `${nCeps}|${nMels}`;
  let d = dctCache.get(key);
  if (!d) {
    d = [];
    for (let k = 0; k < nCeps; k++) {
      const row = new Float64Array(nMels);
      const s = k === 0 ? Math.sqrt(1 / nMels) : Math.sqrt(2 / nMels);
      for (let m = 0; m < nMels; m++) row[m] = s * Math.cos((Math.PI * k * (m + 0.5)) / nMels);
      d.push(row);
    }
    dctCache.set(key, d);
  }
  return d;
}

/**
 * MFCC analysis. Returns { mfcc, logMel, spec, logE } where mfcc is an array of
 * Float32Array(nCeps) (c0..c12, liftered).
 */
export function mfcc(signal, opts = {}) {
  const o = { ...DEFAULTS, ...opts };
  const spec = powerSpectrogram(signal, o);
  const logMel = melEnergies(spec, o);
  const D = dctMatrix(o.nCeps, o.nMels);
  const lift = new Float64Array(o.nCeps);
  for (let k = 0; k < o.nCeps; k++) lift[k] = o.lifter ? 1 + (o.lifter / 2) * Math.sin((Math.PI * k) / o.lifter) : 1;
  const feats = logMel.map((lm) => {
    const c = new Float32Array(o.nCeps);
    for (let k = 0; k < o.nCeps; k++) {
      let s = 0;
      const row = D[k];
      for (let m = 0; m < o.nMels; m++) s += row[m] * lm[m];
      c[k] = s * lift[k];
    }
    return c;
  });
  const logE = spec.map((p) => {
    let s = 0;
    for (let k = 0; k < p.length; k++) s += p[k];
    return Math.log(Math.max(s, 1e-10));
  });
  return { mfcc: feats, logMel, spec, logE };
}

/** Regression deltas (HTK style) with edge replication. */
export function deltas(feats, N = 2) {
  const T = feats.length;
  if (!T) return [];
  const dim = feats[0].length;
  let denom = 0;
  for (let n = 1; n <= N; n++) denom += 2 * n * n;
  const out = new Array(T);
  for (let t = 0; t < T; t++) {
    const d = new Float32Array(dim);
    for (let n = 1; n <= N; n++) {
      const a = feats[Math.min(T - 1, t + n)];
      const b = feats[Math.max(0, t - n)];
      for (let k = 0; k < dim; k++) d[k] += n * (a[k] - b[k]);
    }
    for (let k = 0; k < dim; k++) d[k] /= denom;
    out[t] = d;
  }
  return out;
}

export function concatFeatures(...streams) {
  const T = streams[0].length;
  const out = new Array(T);
  for (let t = 0; t < T; t++) {
    const parts = streams.map((s) => s[t]);
    const dim = parts.reduce((a, p) => a + p.length, 0);
    const v = new Float32Array(dim);
    let off = 0;
    for (const p of parts) { v.set(p, off); off += p.length; }
    out[t] = v;
  }
  return out;
}

/** Cepstral mean and variance normalisation per utterance. */
export function cmvn(feats, { variance = true } = {}) {
  const T = feats.length;
  if (!T) return [];
  const dim = feats[0].length;
  const mean = new Float64Array(dim);
  const v = new Float64Array(dim);
  for (const f of feats) for (let k = 0; k < dim; k++) mean[k] += f[k];
  for (let k = 0; k < dim; k++) mean[k] /= T;
  for (const f of feats) for (let k = 0; k < dim; k++) v[k] += (f[k] - mean[k]) ** 2;
  const sd = Array.from(v, (x) => Math.sqrt(x / T) || 1);
  return feats.map((f) => {
    const o = new Float32Array(dim);
    for (let k = 0; k < dim; k++) o[k] = variance ? (f[k] - mean[k]) / sd[k] : f[k] - mean[k];
    return o;
  });
}

// ------------------------------------------------------- Frame features ---
export function frameEnergyDb(signal, frameLen = 400, hop = 160) {
  const T = numFrames(signal.length, frameLen, hop);
  const out = new Float32Array(T);
  for (let t = 0; t < T; t++) {
    let s = 0;
    const off = t * hop;
    for (let i = 0; i < frameLen; i++) {
      const v = signal[off + i] || 0;
      s += v * v;
    }
    out[t] = 10 * Math.log10(s / frameLen + 1e-12);
  }
  return out;
}

export function zeroCrossingRate(signal, frameLen = 400, hop = 160) {
  const T = numFrames(signal.length, frameLen, hop);
  const out = new Float32Array(T);
  for (let t = 0; t < T; t++) {
    let z = 0;
    const off = t * hop;
    for (let i = 1; i < frameLen; i++) {
      const a = signal[off + i - 1] || 0;
      const b = signal[off + i] || 0;
      if ((a >= 0) !== (b >= 0)) z++;
    }
    out[t] = z / (frameLen - 1);
  }
  return out;
}

/** Spectral centroid / bandwidth / 85% roll-off (Hz) and flatness per frame. */
export function spectralDescriptors(spec, sr = SR, nfft = 512) {
  const binHz = sr / nfft;
  return spec.map((p) => {
    let sum = 0, wsum = 0, logSum = 0;
    for (let k = 1; k < p.length; k++) {
      sum += p[k];
      wsum += p[k] * k * binHz;
      logSum += Math.log(p[k] + 1e-12);
    }
    const centroid = sum > 0 ? wsum / sum : 0;
    let bw = 0;
    for (let k = 1; k < p.length; k++) bw += p[k] * (k * binHz - centroid) ** 2;
    bw = sum > 0 ? Math.sqrt(bw / sum) : 0;
    let acc = 0, rolloff = 0;
    for (let k = 1; k < p.length; k++) {
      acc += p[k];
      if (acc >= 0.85 * sum) { rolloff = k * binHz; break; }
    }
    const n = p.length - 1;
    const flatness = sum > 0 ? Math.exp(logSum / n) / (sum / n) : 1;
    return { centroid, bandwidth: bw, rolloff, flatness };
  });
}

/** YIN fundamental-frequency tracker. Returns Float32Array of f0 (0 = unvoiced). */
export function pitchYin(signal, { sr = SR, hop = 160, win = 400, fmin = 70, fmax = 450, threshold = 0.2, fallback = 0.35 } = {}) {
  const tauMin = Math.floor(sr / fmax);
  const tauMax = Math.ceil(sr / fmin);
  const T = numFrames(signal.length, win, hop);
  const f0 = new Float32Array(T);
  const d = new Float64Array(tauMax + 2);
  for (let t = 0; t < T; t++) {
    const off = t * hop;
    if (off + win + tauMax >= signal.length) break;
    d.fill(0);
    for (let tau = 1; tau <= tauMax + 1; tau++) {
      let s = 0;
      for (let j = 0; j < win; j++) {
        const diff = signal[off + j] - signal[off + j + tau];
        s += diff * diff;
      }
      d[tau] = s;
    }
    // cumulative mean normalised difference
    let run = 0;
    let best = -1;
    const cm = new Float64Array(tauMax + 2);
    cm[0] = 1;
    for (let tau = 1; tau <= tauMax + 1; tau++) {
      run += d[tau];
      cm[tau] = run > 0 ? (d[tau] * tau) / run : 1;
    }
    let gmin = tauMin;
    for (let tau = tauMin; tau <= tauMax; tau++) {
      if (cm[tau] < cm[gmin]) gmin = tau;
      if (cm[tau] < threshold) {
        while (tau + 1 <= tauMax && cm[tau + 1] < cm[tau]) tau++;
        best = tau;
        break;
      }
    }
    // no dip below the threshold: accept the global minimum if it is still clearly periodic
    if (best < 0 && cm[gmin] < fallback) best = gmin;
    if (best > 0) {
      const a = cm[best - 1], b = cm[best], c = cm[best + 1];
      const denom = a - 2 * b + c;
      const shift = denom !== 0 ? (0.5 * (a - c)) / denom : 0;
      f0[t] = sr / (best + Math.max(-1, Math.min(1, shift)));
    }
  }
  return f0;
}

/** Leading/trailing silence detection on frame energy. Returns sample range. */
export function trimSilence(signal, { sr = SR, frameLen = 400, hop = 160, dbBelowPeak = 35, padMs = 40 } = {}) {
  const e = frameEnergyDb(signal, frameLen, hop);
  let peak = -Infinity;
  for (const v of e) if (v > peak) peak = v;
  const sorted = Array.from(e).sort((a, b) => a - b);
  const floor = sorted[Math.floor(sorted.length * 0.1)] ?? peak;
  const thr = Math.max(peak - dbBelowPeak, floor + 6);
  let s = 0;
  while (s < e.length && e[s] < thr) s++;
  let en = e.length - 1;
  while (en > s && e[en] < thr) en--;
  const pad = Math.round((padMs / 1000) * sr);
  const start = Math.max(0, s * hop - pad);
  const end = Math.min(signal.length, en * hop + frameLen + pad);
  if (end - start < frameLen * 2) return { start: 0, end: signal.length };
  return { start, end };
}

export function trim(signal, opts) {
  const { start, end } = trimSilence(signal, opts);
  return signal.subarray(start, end);
}

/** Peak-normalise to 0.95 full scale. */
export function normalizePeak(signal) {
  let m = 0;
  for (let i = 0; i < signal.length; i++) m = Math.max(m, Math.abs(signal[i]));
  if (m < 1e-6) return signal;
  const g = 0.95 / m;
  const y = new Float32Array(signal.length);
  for (let i = 0; i < signal.length; i++) y[i] = signal[i] * g;
  return y;
}

/**
 * Standard recognition front end: trim → MFCC(13) → Δ → ΔΔ → CMVN (39 dims).
 */
export function recognitionFeatures(signal, opts = {}) {
  const x = opts.skipTrim ? signal : trim(signal, opts);
  const { mfcc: c } = mfcc(x, opts);
  const d1 = deltas(c, 2);
  const d2 = deltas(d1, 2);
  return cmvn(concatFeatures(c, d1, d2));
}

/**
 * Heuristic frame labelling for pronunciation analysis:
 *   'S' silence, 'V' vowel-like (voiced, sonorant), 'C' consonant-like.
 */
export function labelFrames({ energyDb, zcr, f0, flatness }) {
  let peak = -Infinity;
  for (const v of energyDb) if (v > peak) peak = v;
  const T = energyDb.length;
  const labels = new Array(T);
  for (let t = 0; t < T; t++) {
    const e = energyDb[t];
    const flat = flatness[t] ?? 0;
    if (e < peak - 35) labels[t] = 'S';
    else if ((f0[t] || 0) > 0 && e > peak - 28 && zcr[t] < 0.2 && flat < 0.3) labels[t] = 'V';
    // loud, low-ZCR, peaky-spectrum frames are sonorant even if the pitch tracker dropped out
    else if (e > peak - 15 && zcr[t] < 0.15 && flat < 0.08) labels[t] = 'V';
    else labels[t] = 'C';
  }
  // smooth isolated labels (median-like, 3 frames)
  for (let t = 1; t < T - 1; t++) {
    if (labels[t - 1] === labels[t + 1] && labels[t] !== labels[t - 1]) labels[t] = labels[t - 1];
  }
  return labels;
}

export function segmentsFromLabels(labels, hopSec = 0.01) {
  const segs = [];
  let start = 0;
  for (let t = 1; t <= labels.length; t++) {
    if (t === labels.length || labels[t] !== labels[start]) {
      segs.push({ label: labels[start], start, end: t, t0: start * hopSec, t1: t * hopSec });
      start = t;
    }
  }
  return segs;
}

/** Simple linear-interpolation resampler (used only as a fallback). */
export function resampleLinear(x, from, to) {
  if (from === to) return x;
  const ratio = from / to;
  const n = Math.floor(x.length / ratio);
  const y = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const p = i * ratio;
    const i0 = Math.floor(p);
    const f = p - i0;
    y[i] = (x[i0] || 0) * (1 - f) + (x[i0 + 1] || 0) * f;
  }
  return y;
}

export function mean(arr) {
  let s = 0;
  for (const v of arr) s += v;
  return arr.length ? s / arr.length : 0;
}
