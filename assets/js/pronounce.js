// Pronunciation assessment: compares a student utterance with a reference
// utterance using MFCC + DTW, then breaks the difference down into vowel,
// consonant, duration, spectral, pitch and per-coefficient components.
// Pure computation (no DOM) so it can be calibrated from Node.

import {
  SR, mfcc, deltas, cmvn, concatFeatures, frameEnergyDb, zeroCrossingRate,
  spectralDescriptors, pitchYin, labelFrames, segmentsFromLabels, trim, normalizePeak, mean,
} from './dsp.js';
import { dtw, costPerReferenceFrame, referenceToTestMap } from './dtw.js';
import { applyLinear } from './linalg.js';

const HOP_SEC = 0.01;

// Logistic mapping from a distance to a 0–100 similarity score.
// d50 = distance that maps to 50 %, s = spread. Calibrated on the TTS corpus
// with scripts/calibrate_pronunciation.mjs (correct vs. mispronounced words
// spoken by speakers other than the reference speaker).
// Values below are for the LDA front end (data/lda.json): a correct word from
// a different speaker has a median DTW distance of ~5.8, which maps to ~84 %.
// Vowel/consonant curves are scaled so their medians map to the same ~84 %.
export const CALIBRATION = {
  dtw: { d50: 6.8, s: 0.6 },
  vowel: { d50: 8.07, s: 0.88 },
  consonant: { d50: 8.76, s: 1.19 },
  duration: { tol: 0.35 },
  spectral: { d50: 6.0, s: 1.6 },
};

const logistic = (d, { d50, s }) => 100 / (1 + Math.exp((d - d50) / s));

/**
 * @param {Float32Array} raw mono 16 kHz signal
 * @param {{sr?: number, lda?: {mean:number[], W:number[][]}}} opts
 *   lda: speaker-robust projection of MFCC+Δ+ΔΔ (data/lda.json). Without it the
 *   comparison uses plain CMVN MFCCs, which are far more speaker dependent.
 */
export function analyzeUtterance(raw, { sr = SR, lda = null } = {}) {
  const x = trim(normalizePeak(raw), { sr });
  const { mfcc: c, spec, logMel } = mfcc(x, { sr });
  const d1 = deltas(c, 2);
  let feat;
  if (lda) {
    feat = applyLinear(cmvn(concatFeatures(c, d1, deltas(d1, 2))), lda);
  } else {
    // c1..c12 and the deltas of c0..c12: c0 (loudness) is left out of the
    // static part so recording level does not dominate the distance.
    feat = cmvn(concatFeatures(c.map((v) => v.subarray(1)), d1));
  }
  const energyDb = frameEnergyDb(x);
  const zcr = zeroCrossingRate(x);
  const f0 = pitchYin(x, { sr });
  const desc = spectralDescriptors(spec, sr);
  const T = Math.min(c.length, energyDb.length, zcr.length, f0.length);
  const labels = labelFrames({
    energyDb: energyDb.subarray(0, T),
    zcr: zcr.subarray(0, T),
    f0: f0.subarray(0, T),
    flatness: desc.slice(0, T).map((d) => d.flatness),
  });
  return {
    signal: x, sr, duration: x.length / sr, mfcc: c.slice(0, T), feat: feat.slice(0, T), spec: spec.slice(0, T),
    logMel: logMel.slice(0, T), energyDb, zcr, f0, desc: desc.slice(0, T), labels, segments: segmentsFromLabels(labels, HOP_SEC),
  };
}

function smooth(arr, w = 5) {
  const out = new Float32Array(arr.length);
  const h = Math.floor(w / 2);
  for (let i = 0; i < arr.length; i++) {
    let s = 0, n = 0;
    for (let k = -h; k <= h; k++) {
      const v = arr[i + k];
      if (v !== undefined) { s += v; n++; }
    }
    out[i] = s / n;
  }
  return out;
}

function pearson(a, b) {
  const n = a.length;
  if (n < 3) return null;
  const ma = mean(a), mb = mean(b);
  let num = 0, da = 0, db = 0;
  for (let i = 0; i < n; i++) {
    num += (a[i] - ma) * (b[i] - mb);
    da += (a[i] - ma) ** 2;
    db += (b[i] - mb) ** 2;
  }
  return da && db ? num / Math.sqrt(da * db) : null;
}

const count = (labels, l) => labels.reduce((a, x) => a + (x === l), 0);

/**
 * Multi-template error identification: compare the student utterance with
 * every pronunciation form (correct + typical errors) spoken by several
 * speakers and average the DTW distance per form. Because every form is
 * represented by the same set of speakers, speaker mismatch cancels out.
 * @param templates {formId: analysis[]}
 */
export function identifyForm(test, templates, { tau = 0.12 } = {}) {
  const rows = Object.entries(templates).map(([form, list]) => {
    const ds = list.map((t) => dtw(t.feat, test.feat, { band: 0.35 }).distance);
    return { form, dist: mean(ds), n: ds.length };
  });
  const best = Math.min(...rows.map((r) => r.dist));
  const z = rows.reduce((a, r) => a + Math.exp(-(r.dist - best) / tau), 0);
  for (const r of rows) r.prob = Math.exp(-(r.dist - best) / tau) / z;
  return rows.sort((a, b) => a.dist - b.dist);
}

export function comparePronunciation(ref, test, { calibration = CALIBRATION } = {}) {
  const CAL = { ...CALIBRATION, ...calibration };
  const res = dtw(ref.feat, test.feat, { band: 0.35 });
  const perRef = costPerReferenceFrame(res);
  const map = referenceToTestMap(res);
  const { path } = res;

  // ---- segmental (vowel / consonant) costs
  const vCost = [], cCost = [];
  ref.labels.forEach((l, i) => { if (l === 'V') vCost.push(perRef[i]); else if (l === 'C') cCost.push(perRef[i]); });
  const vowelDist = vCost.length ? mean(vCost) : res.distance;
  const consDist = cCost.length ? mean(cCost) : res.distance;

  // ---- durations
  const durRatio = test.duration / ref.duration;
  const refV = count(ref.labels, 'V'), testV = count(test.labels, 'V');
  const refC = count(ref.labels, 'C'), testC = count(test.labels, 'C');
  const vowelShareRef = refV / Math.max(1, refV + refC);
  const vowelShareTest = testV / Math.max(1, testV + testC);

  // ---- MFCC coefficient differences along the path (mean-normalised MFCC)
  const cmn = (c) => cmvn(c, { variance: false });
  const rc = cmn(ref.mfcc), tc = cmn(test.mfcc);
  const nC = rc[0].length;
  const coefDiff = new Float64Array(nC);
  for (const [i, j] of path) for (let k = 0; k < nC; k++) coefDiff[k] += Math.abs(rc[i][k] - tc[j][k]);
  for (let k = 0; k < nC; k++) coefDiff[k] /= path.length;

  // ---- spectral envelope: long-term average log-mel spectrum (shape only)
  const ltas = (lm) => {
    const m = new Float64Array(lm[0].length);
    for (const f of lm) for (let k = 0; k < m.length; k++) m[k] += f[k];
    const avg = Array.from(m, (v) => v / lm.length);
    const g = mean(avg);
    return avg.map((v) => (10 / Math.LN10) * (v - g)); // dB, level removed
  };
  const ltasRef = ltas(ref.logMel), ltasTest = ltas(test.logMel);
  const ltasDiff = ltasRef.map((v, k) => ltasTest[k] - v);
  const spectralDist = Math.sqrt(mean(ltasDiff.map((d) => d * d)));

  // aligned spectral centroid in fricative-like reference frames
  const fricRef = [], fricTest = [];
  ref.labels.forEach((l, i) => {
    if (l === 'C' && ref.zcr[i] > 0.25) {
      fricRef.push(ref.desc[i].centroid);
      fricTest.push(test.desc[Math.round(map[i])]?.centroid ?? 0);
    }
  });
  const fricCentroidDiff = fricRef.length >= 3 ? mean(fricTest) - mean(fricRef) : null;

  // aligned pitch contour (semitones, voiced in both)
  const pa = [], pb = [];
  for (const [i, j] of path) {
    const a = ref.f0[i], b = test.f0[j];
    if (a > 0 && b > 0) { pa.push(12 * Math.log2(a)); pb.push(12 * Math.log2(b)); }
  }
  const pitchCorr = pearson(pa, pb);

  // tempo: local slope of the warping path (test frames per reference frame)
  const tempo = smooth(Float32Array.from(map, (v, i) => (i ? v - map[i - 1] : 1)), 15);

  // ---- problem regions on the reference time axis
  // Frames whose smoothed cost is clearly above this word's own average are
  // "hot"; contiguous hot runs (≥ 40 ms) are ranked by severity.
  const sm = smooth(perRef, 7);
  const speech = Array.from(sm).filter((_, i) => ref.labels[i] !== 'S');
  const mu = mean(speech);
  const sd = Math.sqrt(mean(speech.map((v) => (v - mu) ** 2)));
  const thr = mu + 0.5 * sd;
  const makeRegion = (a, b) => {
    const labs = ref.labels.slice(a, b);
    return {
      start: a, end: b, t0: a * HOP_SEC, t1: b * HOP_SEC,
      kind: count(labs, 'V') >= count(labs, 'C') ? 'vowel' : 'consonant',
      severity: mean(Array.from(sm.subarray(a, b))),
      testT0: map[a] * HOP_SEC, testT1: map[b - 1] * HOP_SEC,
    };
  };
  let regions = [];
  let s = -1;
  for (let i = 0; i <= sm.length; i++) {
    const hot = i < sm.length && sm[i] > thr && ref.labels[i] !== 'S';
    if (hot && s < 0) s = i;
    if (!hot && s >= 0) {
      if (i - s >= 4) regions.push(makeRegion(s, i));
      s = -1;
    }
  }
  if (!regions.length && sm.length > 8) {
    // fall back to the single worst 80 ms window
    let best = 0, bestV = -Infinity;
    for (let i = 0; i + 8 <= sm.length; i++) {
      const v = mean(Array.from(sm.subarray(i, i + 8)));
      if (v > bestV) { bestV = v; best = i; }
    }
    regions = [makeRegion(best, best + 8)];
  }
  regions.sort((a, b) => b.severity * (b.end - b.start) ** 0.5 - a.severity * (a.end - a.start) ** 0.5);

  // ---- scores
  const mfccScore = logistic(res.distance, CAL.dtw);
  const vowelScore = logistic(vowelDist, CAL.vowel);
  const consonantScore = logistic(consDist, CAL.consonant);
  const durationScore = 100 * Math.exp(-Math.abs(Math.log(durRatio)) / CAL.duration.tol);
  const spectralScore = logistic(spectralDist, CAL.spectral);
  const overall = 0.5 * mfccScore + 0.15 * vowelScore + 0.15 * consonantScore + 0.1 * durationScore + 0.1 * spectralScore;

  return {
    overall, scores: { mfcc: mfccScore, vowel: vowelScore, consonant: consonantScore, duration: durationScore, spectral: spectralScore },
    distances: { dtw: res.distance, vowel: vowelDist, consonant: consDist, spectral: spectralDist },
    dtw: res, perRef, map, tempo, coefDiff: Array.from(coefDiff), ltasRef, ltasTest, ltasDiff,
    durRatio, vowelShareRef, vowelShareTest, fricCentroidDiff, pitchCorr, regions,
    feedback: feedback({ durRatio, vowelShareRef, vowelShareTest, vowelScore, consonantScore, fricCentroidDiff, pitchCorr, regions, overall }),
  };
}

function feedback(r) {
  const out = [];
  const pct = (x) => `${Math.round(Math.abs(x - 1) * 100)}%`;
  if (r.overall >= 80) out.push({ level: 'good', text: 'Close match to the reference overall.' });
  if (r.durRatio > 1.2) out.push({ level: 'warn', text: `Spoken ${pct(r.durRatio)} slower than the reference.` });
  else if (r.durRatio < 0.83) out.push({ level: 'warn', text: `Spoken ${pct(r.durRatio)} faster than the reference.` });
  const share = r.vowelShareTest / Math.max(1e-6, r.vowelShareRef);
  if (share > 1.25) out.push({ level: 'warn', text: 'Vowels take up more of the word than in the reference: check for lengthened vowels (a→ā, i→ī) or extra vowels breaking a cluster.' });
  else if (share < 0.8) out.push({ level: 'warn', text: 'Vowels take up less of the word than in the reference: check for shortened long vowels (ā→a, ī→i).' });
  if (r.vowelScore + 8 < r.consonantScore) out.push({ level: 'bad', text: 'The largest differences are in vowel regions (vowel quality or length).' });
  else if (r.consonantScore + 8 < r.vowelScore) out.push({ level: 'bad', text: 'The largest differences are in consonant regions (place of articulation, aspiration or clusters).' });
  if (r.fricCentroidDiff !== null) {
    if (r.fricCentroidDiff > 600) out.push({ level: 'bad', text: `Hissing sounds are ~${Math.round(r.fricCentroidDiff)} Hz brighter than the reference. A dental s may be replacing a palatal ś or retroflex ṣ.` });
    else if (r.fricCentroidDiff < -600) out.push({ level: 'bad', text: `Hissing sounds are ~${Math.round(-r.fricCentroidDiff)} Hz darker than the reference. A ś/ṣ may be replacing a dental s, or the sound is under-articulated.` });
  }
  if (r.pitchCorr !== null && r.pitchCorr < 0.3) out.push({ level: 'info', text: 'Pitch contour differs from the reference (intonation or accent).' });
  if (r.regions.length) {
    const top = r.regions[0];
    out.push({ level: 'info', text: `Biggest mismatch: ${top.t0.toFixed(2)}–${top.t1.toFixed(2)} s in the reference (${top.kind} region).` });
  }
  return out;
}
