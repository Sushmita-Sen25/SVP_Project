// Video analysis primitives that work on sampled frames (RGBA pixel arrays):
// colour histograms, shot-boundary detection, keyframe selection, motion
// energy, background subtraction + connected components, a multi-object
// tracker and summary planning. No DOM access.

// ------------------------------------------------------------ colour ---
export const H_BINS = 16, S_BINS = 4, V_BINS = 4;
export const HIST_SIZE = H_BINS * S_BINS * V_BINS;

/** Normalised HSV histogram (16×4×4 bins) of an RGBA buffer. */
export function hsvHistogram(rgba) {
  const hist = new Float32Array(HIST_SIZE);
  const n = rgba.length / 4;
  for (let i = 0; i < rgba.length; i += 4) {
    const r = rgba[i] / 255, g = rgba[i + 1] / 255, b = rgba[i + 2] / 255;
    const max = Math.max(r, g, b), min = Math.min(r, g, b);
    const d = max - min;
    let h = 0;
    if (d > 1e-6) {
      if (max === r) h = ((g - b) / d + 6) % 6;
      else if (max === g) h = (b - r) / d + 2;
      else h = (r - g) / d + 4;
    }
    const s = max > 0 ? d / max : 0;
    const hb = Math.min(H_BINS - 1, Math.floor((h / 6) * H_BINS));
    const sb = Math.min(S_BINS - 1, Math.floor(s * S_BINS));
    const vb = Math.min(V_BINS - 1, Math.floor(max * V_BINS));
    hist[(hb * S_BINS + sb) * V_BINS + vb]++;
  }
  for (let k = 0; k < HIST_SIZE; k++) hist[k] /= n;
  return hist;
}

/** Bhattacharyya (Hellinger) distance between two normalised histograms, 0..1. */
export function histDistance(a, b) {
  let s = 0;
  for (let k = 0; k < a.length; k++) s += Math.sqrt(a[k] * b[k]);
  return Math.sqrt(Math.max(0, 1 - s));
}

export function toGray(rgba) {
  const g = new Uint8Array(rgba.length / 4);
  for (let i = 0, j = 0; i < rgba.length; i += 4, j++) g[j] = (rgba[i] * 77 + rgba[i + 1] * 150 + rgba[i + 2] * 29) >> 8;
  return g;
}

/** Variance of the Laplacian: a standard focus / sharpness measure. */
export function sharpness(gray, w, h) {
  let s = 0, s2 = 0, n = 0;
  for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
    const i = y * w + x;
    const l = 4 * gray[i] - gray[i - 1] - gray[i + 1] - gray[i - w] - gray[i + w];
    s += l; s2 += l * l; n++;
  }
  return n ? s2 / n - (s / n) ** 2 : 0;
}

export function meanAbsDiff(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]);
  return s / a.length / 255;
}

// ------------------------------------------------------------ shots ---
/**
 * Shot-boundary detection on consecutive-sample dissimilarities.
 * Combined distance D = 0.75·histogram + 0.25·(2·pixel difference).
 *  - abrupt cut: D_i is a local maximum above max(minCut, μ + k·σ) where μ, σ
 *    come from a sliding window that excludes i (adaptive threshold);
 *  - gradual transition (twin comparison): a run of moderately high D whose
 *    end points differ by more than the cut threshold.
 */
export function detectBoundaries(samples, { k = 3, window = 8, minCut = 0.28, minSceneSec = 1.5 } = {}) {
  const n = samples.length;
  const D = new Float32Array(n);
  for (let i = 1; i < n; i++) {
    const h = histDistance(samples[i - 1].hist, samples[i].hist);
    const p = Math.min(1, 2 * meanAbsDiff(samples[i - 1].gray, samples[i].gray));
    D[i] = 0.75 * h + 0.25 * p;
  }
  const thr = new Float32Array(n);
  const lowThr = new Float32Array(n);
  for (let i = 1; i < n; i++) {
    let s = 0, s2 = 0, c = 0;
    for (let j = Math.max(1, i - window); j <= Math.min(n - 1, i + window); j++) {
      if (Math.abs(j - i) <= 1) continue;
      s += D[j]; s2 += D[j] * D[j]; c++;
    }
    const mu = c ? s / c : 0;
    const sd = c ? Math.sqrt(Math.max(0, s2 / c - mu * mu)) : 0;
    thr[i] = Math.max(minCut, mu + k * sd);
    lowThr[i] = Math.max(minCut * 0.45, mu + 1.5 * sd);
  }
  const raw = [];
  for (let i = 1; i < n; i++) {
    if (D[i] > thr[i] && D[i] >= (D[i - 1] || 0) && D[i] >= (D[i + 1] || 0)) {
      raw.push({ index: i, time: (samples[i - 1].t + samples[i].t) / 2, type: 'cut', strength: D[i] / thr[i] });
    }
  }
  // twin comparison for gradual transitions
  let i = 1;
  while (i < n) {
    if (D[i] > lowThr[i] && !raw.some((b) => Math.abs(b.index - i) <= 1)) {
      let j = i;
      while (j + 1 < n && D[j + 1] > lowThr[i] * 0.7 && !raw.some((b) => b.index === j + 1)) j++;
      if (j > i) {
        const span = histDistance(samples[i - 1].hist, samples[j].hist);
        if (span > Math.max(minCut, thr[i] * 0.9)) {
          const mid = Math.round((i + j) / 2);
          raw.push({ index: mid, time: (samples[i - 1].t + samples[j].t) / 2, type: 'gradual', strength: span / thr[i], span: [samples[i - 1].t, samples[j].t] });
        }
      }
      i = j + 1;
    } else i++;
  }
  raw.sort((a, b) => a.time - b.time);
  // enforce a minimum scene length (keep the stronger boundary)
  const out = [];
  for (const b of raw) {
    const last = out[out.length - 1];
    const prevT = last ? last.time : samples[0]?.t ?? 0;
    if (b.time - prevT < minSceneSec) {
      if (last && b.strength > last.strength) out[out.length - 1] = b;
      continue;
    }
    out.push(b);
  }
  return { boundaries: out, D, thr };
}

export function scenesFromBoundaries(boundaries, samples, duration) {
  const scenes = [];
  let start = 0, t0 = 0;
  for (const b of boundaries) {
    scenes.push({ start, end: b.index, t0, t1: b.time });
    start = b.index; t0 = b.time;
  }
  scenes.push({ start, end: samples.length, t0, t1: duration });
  return scenes.filter((s) => s.end > s.start).map((s, i) => ({ ...s, id: i + 1 }));
}

// ------------------------------------------------------------ keyframes ---
function meanHist(samples, a, b) {
  const m = new Float32Array(HIST_SIZE);
  for (let i = a; i < b; i++) for (let k = 0; k < HIST_SIZE; k++) m[k] += samples[i].hist[k];
  for (let k = 0; k < HIST_SIZE; k++) m[k] /= Math.max(1, b - a);
  return m;
}

/**
 * One keyframe per scene (the medoid: the sample closest to the scene's mean
 * histogram, with a small bonus for sharpness), plus extra keyframes inside
 * long scenes whenever the content drifts far from the last keyframe.
 */
export function selectKeyframes(samples, scenes, { novelty = 0.3, minGapSec = 8 } = {}) {
  const maxSharp = Math.max(1, ...samples.map((s) => s.sharp));
  const keyframes = [];
  for (const sc of scenes) {
    const mh = meanHist(samples, sc.start, sc.end);
    sc.meanHist = mh;
    let best = sc.start, bestScore = Infinity;
    // skip the first/last sample of a scene (often mid-transition) when possible
    const a = sc.end - sc.start > 3 ? sc.start + 1 : sc.start;
    const b = sc.end - sc.start > 3 ? sc.end - 1 : sc.end;
    for (let i = a; i < b; i++) {
      const sc2 = histDistance(samples[i].hist, mh) - 0.08 * (samples[i].sharp / maxSharp);
      if (sc2 < bestScore) { bestScore = sc2; best = i; }
    }
    const picked = [best];
    let last = best;
    for (let i = a; i < b; i++) {
      if (samples[i].t - samples[last].t < minGapSec) continue;
      if (histDistance(samples[i].hist, samples[last].hist) > novelty) { picked.push(i); last = i; }
    }
    picked.sort((x, y) => x - y);
    sc.keyframes = picked;
    for (const idx of picked) keyframes.push({ index: idx, t: samples[idx].t, scene: sc.id, primary: idx === best });
  }
  return keyframes;
}

// ------------------------------------------------------------ motion ---
/** Motion energy per sample and an accumulated motion (energy) map. Pairs across a boundary are skipped. */
export function motionAnalysis(samples, boundaries, w, h) {
  const cut = new Set(boundaries.map((b) => b.index));
  const energy = new Float32Array(samples.length);
  const map = new Float32Array(w * h);
  let pairs = 0;
  for (let i = 1; i < samples.length; i++) {
    if (cut.has(i)) continue;
    const a = samples[i - 1].gray, b = samples[i].gray;
    let s = 0;
    for (let p = 0; p < a.length; p++) {
      const d = Math.abs(a[p] - b[p]);
      s += d;
      if (d > 12) map[p] += d;
    }
    energy[i] = s / a.length / 255;
    pairs++;
  }
  let mx = 0;
  for (const v of map) if (v > mx) mx = v;
  if (mx > 0) for (let p = 0; p < map.length; p++) map[p] /= mx;
  return { energy, map, pairs };
}

// ------------------------------------------------------- camera motion ---
/**
 * Global translation between two grayscale frames by exhaustive block
 * matching on a 2× downsampled image. Returns the shift (in pixels of the
 * full analysis frame) and how much better it explains the change than "no
 * motion" (gain > 1.3 with a non-zero shift ⇒ the camera is panning/tilting).
 */
export function globalShift(a, b, w, h, maxShift = 6) {
  const W = w >> 1, H = h >> 1;
  const da = new Float32Array(W * H), db = new Float32Array(W * H);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const i = 2 * y * w + 2 * x;
    da[y * W + x] = (a[i] + a[i + 1] + a[i + w] + a[i + w + 1]) / 4;
    db[y * W + x] = (b[i] + b[i + 1] + b[i + w] + b[i + w + 1]) / 4;
  }
  const err = (dx, dy) => {
    let s = 0, n = 0;
    for (let y = maxShift; y < H - maxShift; y++) for (let x = maxShift; x < W - maxShift; x++) {
      s += Math.abs(da[y * W + x] - db[(y + dy) * W + x + dx]);
      n++;
    }
    return s / n;
  };
  const e0 = err(0, 0);
  let best = { dx: 0, dy: 0, e: e0 };
  for (let dy = -maxShift; dy <= maxShift; dy++) for (let dx = -maxShift; dx <= maxShift; dx++) {
    if (!dx && !dy) continue;
    const e = err(dx, dy);
    if (e < best.e) best = { dx, dy, e };
  }
  const gain = e0 / Math.max(0.5, best.e);
  const moving = (best.dx || best.dy) && gain > 1.3 && e0 > 2;
  return { dx: best.dx * 2, dy: best.dy * 2, gain, moving: !!moving };
}

// ------------------------------------------------------- blob detection ---
function median(arr) {
  const s = Float32Array.from(arr).sort();
  return s[s.length >> 1];
}

/** Per-scene background (temporal median of up to 25 samples). */
export function sceneBackground(samples, sc, w, h) {
  const idx = [];
  const n = sc.end - sc.start;
  const step = Math.max(1, Math.floor(n / 25));
  for (let i = sc.start; i < sc.end; i += step) idx.push(i);
  const bg = new Uint8Array(w * h);
  const col = new Float32Array(idx.length);
  for (let p = 0; p < w * h; p++) {
    for (let k = 0; k < idx.length; k++) col[k] = samples[idx[k]].gray[p];
    bg[p] = median(col);
  }
  return bg;
}

/** Foreground boxes from background subtraction + dilation + connected components. */
export function detectBlobs(gray, bg, w, h, { thresh = 28, minAreaFrac = 0.0015, maxFgFrac = 0.4 } = {}) {
  const fg = new Uint8Array(w * h);
  let count = 0;
  for (let p = 0; p < fg.length; p++) if (Math.abs(gray[p] - bg[p]) > thresh) { fg[p] = 1; count++; }
  if (count / fg.length > maxFgFrac) return { boxes: [], cameraMotion: true };
  // 3×3 dilation (twice) to merge limbs into one blob
  let m = fg;
  for (let it = 0; it < 2; it++) {
    const d = new Uint8Array(w * h);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      if (!m[y * w + x]) continue;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const yy = y + dy, xx = x + dx;
        if (yy >= 0 && yy < h && xx >= 0 && xx < w) d[yy * w + xx] = 1;
      }
    }
    m = d;
  }
  // connected components (iterative flood fill, 8-connectivity)
  const lab = new Int32Array(w * h);
  const boxes = [];
  const stack = [];
  let id = 0;
  for (let p = 0; p < m.length; p++) {
    if (!m[p] || lab[p]) continue;
    id++;
    let x0 = w, y0 = h, x1 = 0, y1 = 0, area = 0;
    stack.push(p);
    lab[p] = id;
    while (stack.length) {
      const q = stack.pop();
      const x = q % w, y = (q / w) | 0;
      area++;
      if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const yy = y + dy, xx = x + dx;
        if (yy < 0 || yy >= h || xx < 0 || xx >= w) continue;
        const r = yy * w + xx;
        if (m[r] && !lab[r]) { lab[r] = id; stack.push(r); }
      }
    }
    if (area >= minAreaFrac * w * h) boxes.push({ x: x0 / w, y: y0 / h, w: (x1 - x0 + 1) / w, h: (y1 - y0 + 1) / h, score: 1, label: 'object' });
  }
  return { boxes, cameraMotion: false };
}

// ------------------------------------------------------------ tracking ---
const center = (b) => [b.x + b.w / 2, b.y + b.h / 2];
function iou(a, b) {
  const x0 = Math.max(a.x, b.x), y0 = Math.max(a.y, b.y);
  const x1 = Math.min(a.x + a.w, b.x + b.w), y1 = Math.min(a.y + a.h, b.y + b.h);
  const inter = Math.max(0, x1 - x0) * Math.max(0, y1 - y0);
  return inter / (a.w * a.h + b.w * b.h - inter || 1);
}

/**
 * Greedy multi-object tracker: constant-velocity prediction, matching cost =
 * centre distance (normalised) minus an IoU bonus, gated; tracks end after
 * `maxMissed` consecutive misses.
 */
export class Tracker {
  constructor({ gate = 0.2, maxMissed = 2 } = {}) {
    this.gate = gate; this.maxMissed = maxMissed;
    this.active = []; this.done = []; this.nextId = 1;
  }
  update(dets, t) {
    const pairs = [];
    this.active.forEach((tr, ti) => {
      const last = tr.points[tr.points.length - 1];
      const dt = t - last.t;
      const pred = { ...last.box, x: last.box.x + tr.vx * dt, y: last.box.y + tr.vy * dt };
      const [px, py] = center(pred);
      const [lx, ly] = center(last.box);
      dets.forEach((d, di) => {
        if (tr.label !== d.label) return;
        const [dx, dy] = center(d);
        // nearest of the predicted and the last position, so bounces and turns still match
        const dist = Math.min(Math.hypot(px - dx, py - dy), 1.15 * Math.hypot(lx - dx, ly - dy));
        const sizeRatio = Math.max(d.w * d.h, pred.w * pred.h) / Math.max(1e-6, Math.min(d.w * d.h, pred.w * pred.h));
        if (dist > this.gate * Math.max(1, Math.sqrt(dt * 2)) || sizeRatio > 4) return;
        pairs.push({ ti, di, cost: dist - 0.1 * iou(pred, d) });
      });
    });
    pairs.sort((a, b) => a.cost - b.cost);
    const usedT = new Set(), usedD = new Set();
    for (const p of pairs) {
      if (usedT.has(p.ti) || usedD.has(p.di)) continue;
      usedT.add(p.ti); usedD.add(p.di);
      const tr = this.active[p.ti];
      const last = tr.points[tr.points.length - 1];
      const d = dets[p.di];
      const dt = Math.max(1e-3, t - last.t);
      const [lx, ly] = center(last.box), [cx, cy] = center(d);
      tr.vx = 0.5 * tr.vx + 0.5 * (cx - lx) / dt;
      tr.vy = 0.5 * tr.vy + 0.5 * (cy - ly) / dt;
      tr.points.push({ t, box: d });
      d.trackRef = tr;
      tr.missed = 0;
    }
    this.active.forEach((tr, ti) => { if (!usedT.has(ti)) tr.missed++; });
    this.done.push(...this.active.filter((tr) => tr.missed > this.maxMissed));
    this.active = this.active.filter((tr) => tr.missed <= this.maxMissed);
    dets.forEach((d, di) => {
      if (usedD.has(di)) return;
      const tr = { id: this.nextId++, label: d.label, points: [{ t, box: d }], vx: 0, vy: 0, missed: 0 };
      d.trackRef = tr;
      this.active.push(tr);
    });
  }
  flush() {
    this.done.push(...this.active);
    this.active = [];
  }
  tracks(minHits = 3, minDuration = 0) {
    return this.done.filter((tr) => tr.points.length >= minHits && tr.points[tr.points.length - 1].t - tr.points[0].t >= minDuration).map((tr) => {
      const p = tr.points;
      let path = 0;
      for (let i = 1; i < p.length; i++) {
        const [ax, ay] = center(p[i - 1].box), [bx, by] = center(p[i].box);
        path += Math.hypot(bx - ax, by - ay);
      }
      const [sx, sy] = center(p[0].box), [ex, ey] = center(p[p.length - 1].box);
      const dur = p[p.length - 1].t - p[0].t;
      return { id: tr.id, ref: tr, label: tr.label, points: p, t0: p[0].t, t1: p[p.length - 1].t, duration: dur, path, speed: dur > 0 ? path / dur : 0, dx: ex - sx, dy: ey - sy };
    });
  }
}

export function describeDirection(dx, dy) {
  if (Math.hypot(dx, dy) < 0.05) return 'mostly stationary';
  const horiz = Math.abs(dx) > 0.03 ? (dx > 0 ? 'left → right' : 'right → left') : '';
  const vert = Math.abs(dy) > 0.06 ? (dy > 0 ? 'downwards' : 'upwards') : '';
  return [horiz, vert].filter(Boolean).join(', ') || 'small movement';
}

// ------------------------------------------------------------ summary ---
/** Score scenes and choose summary segments up to `targetSec`. */
export function planSummary(scenes, samples, energy, { targetSec = 30, globalHist }) {
  const maxDur = Math.max(...scenes.map((s) => Math.log1p(s.t1 - s.t0)));
  scenes.forEach((s) => {
    let m = 0, c = 0, det = 0;
    for (let i = s.start; i < s.end; i++) { m += energy[i]; c++; det += samples[i].dets?.length || 0; }
    s.motion = c ? m / c : 0;
    s.detections = det;
    s.distinct = histDistance(s.meanHist, globalHist);
  });
  const maxM = Math.max(1e-6, ...scenes.map((s) => s.motion));
  const maxDist = Math.max(1e-6, ...scenes.map((s) => s.distinct));
  const maxDet = Math.max(1, ...scenes.map((s) => s.detections));
  scenes.forEach((s) => {
    s.importance = 0.3 * (Math.log1p(s.t1 - s.t0) / maxDur) + 0.3 * (s.motion / maxM) + 0.25 * (s.distinct / maxDist) + 0.15 * (s.detections / maxDet);
  });
  const ranked = [...scenes].sort((a, b) => b.importance - a.importance);
  const perSeg = Math.max(2, Math.min(6, targetSec / Math.max(1, Math.min(scenes.length, 8))));
  const segs = [];
  let total = 0;
  for (const s of ranked) {
    if (total >= targetSec) break;
    const len = Math.min(perSeg, s.t1 - s.t0, targetSec - total);
    if (len < 1) continue;
    // centre on the most active sample of the scene
    let bi = s.start, bm = -1;
    for (let i = s.start; i < s.end; i++) if (energy[i] > bm) { bm = energy[i]; bi = i; }
    let start = Math.max(s.t0, Math.min(samples[bi].t - len / 2, s.t1 - len));
    segs.push({ scene: s.id, start, end: start + len, importance: s.importance });
    total += len;
  }
  return segs.sort((a, b) => a.start - b.start);
}

export function globalMeanHist(samples) {
  return meanHist(samples, 0, samples.length);
}

/** Score detected boundaries against ground truth with a time tolerance. */
export function scoreBoundaries(detected, truth, tol = 1.5) {
  const used = new Set();
  let tp = 0;
  for (const g of truth) {
    let best = -1, bd = Infinity;
    detected.forEach((d, i) => {
      const dd = Math.abs(d.time - g.time);
      if (!used.has(i) && dd <= tol && dd < bd) { bd = dd; best = i; }
    });
    if (best >= 0) { used.add(best); tp++; }
  }
  const precision = detected.length ? tp / detected.length : 0;
  const recall = truth.length ? tp / truth.length : 0;
  return { tp, precision, recall, f1: precision + recall ? (2 * precision * recall) / (precision + recall) : 0 };
}
