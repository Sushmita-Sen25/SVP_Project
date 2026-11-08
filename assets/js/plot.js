// Canvas plotting helpers (waveform, spectrogram, heatmaps, DTW matrix, line
// and bar charts). Colours come from CSS custom properties so plots follow
// the light/dark theme; callers redraw on theme change.
import { cssVar } from './ui.js';

export function prepCanvas(canvas, height) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.max(50, canvas.clientWidth || canvas.parentElement?.clientWidth || 600);
  const h = height ?? (canvas.clientHeight || 160);
  canvas.style.height = `${h}px`;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = cssVar('--plot-bg');
  ctx.fillRect(0, 0, w, h);
  ctx.font = '11px Inter, system-ui, sans-serif';
  return { ctx, w, h };
}

// Perceptual "inferno"-like colormap (dark → purple → orange → pale yellow)
const STOPS = [
  [0, 0, 4], [31, 12, 72], [85, 15, 109], [136, 34, 106], [186, 54, 85],
  [227, 89, 51], [249, 140, 10], [249, 201, 50], [252, 255, 164],
];
export function colormap(t) {
  t = Math.max(0, Math.min(1, t)) * (STOPS.length - 1);
  const i = Math.min(STOPS.length - 2, Math.floor(t));
  const f = t - i;
  const a = STOPS[i], b = STOPS[i + 1];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}
// Diverging blue ↔ neutral ↔ orange for signed data
export function diverging(t) {
  t = Math.max(-1, Math.min(1, t));
  const neg = [51, 70, 168], mid = [245, 242, 235], pos = [194, 65, 12];
  const [a, b, f] = t < 0 ? [mid, neg, -t] : [mid, pos, t];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

function imageFrom(width, height, fill) {
  const c = document.createElement('canvas');
  c.width = width; c.height = height;
  const g = c.getContext('2d');
  const img = g.createImageData(width, height);
  fill(img.data);
  g.putImageData(img, 0, 0);
  return c;
}

const LABEL_COLORS = { V: '--accent', C: '--accent-2', S: '--plot-grid' };

/** Vowel/consonant/silence label strip. */
function drawLabelStrip(ctx, labels, x0, w, y, hh) {
  const n = labels.length;
  for (let i = 0; i < n; i++) {
    ctx.fillStyle = cssVar(LABEL_COLORS[labels[i]] || '--plot-grid');
    ctx.fillRect(x0 + (i / n) * w, y, w / n + 0.6, hh);
  }
}

function drawRegions(ctx, regions, totalSec, x0, w, y0, h) {
  if (!regions?.length) return;
  ctx.save();
  for (const r of regions) {
    const a = x0 + (r.t0 / totalSec) * w;
    const b = x0 + (r.t1 / totalSec) * w;
    ctx.fillStyle = 'rgba(220, 38, 38, 0.16)';
    ctx.fillRect(a, y0, b - a, h);
    ctx.strokeStyle = 'rgba(220, 38, 38, 0.85)';
    ctx.lineWidth = 1.5;
    ctx.strokeRect(a + 0.5, y0 + 0.5, b - a - 1, h - 1);
  }
  ctx.restore();
}

export function drawWaveform(canvas, signal, { height = 110, sr = 16000, color, regions, labels } = {}) {
  const { ctx, w, h } = prepCanvas(canvas, height);
  const strip = labels ? 6 : 0;
  const mid = (h - strip) / 2 + strip;
  const amp = (h - strip) / 2 - 6;
  ctx.strokeStyle = cssVar('--plot-grid');
  ctx.beginPath(); ctx.moveTo(0, mid); ctx.lineTo(w, mid); ctx.stroke();
  const n = signal.length;
  const step = n / w;
  ctx.fillStyle = color || cssVar('--accent');
  for (let x = 0; x < w; x++) {
    let lo = 0, hi = 0;
    const a = Math.floor(x * step), b = Math.min(n, Math.floor((x + 1) * step) + 1);
    for (let i = a; i < b; i++) { const v = signal[i]; if (v < lo) lo = v; if (v > hi) hi = v; }
    ctx.fillRect(x, mid - hi * amp, 1, Math.max(1, (hi - lo) * amp));
  }
  if (labels) drawLabelStrip(ctx, labels, 0, w, 0, strip);
  drawRegions(ctx, regions, n / sr, 0, w, strip, h - strip);
  ctx.fillStyle = cssVar('--plot-ink');
  ctx.fillText(`${(n / sr).toFixed(2)} s`, w - 44, h - 6);
}

export function drawSpectrogram(canvas, spec, { height = 190, sr = 16000, maxHz = 8000, dbRange = 75, regions, labels, hopSec = 0.01 } = {}) {
  const { ctx, w, h } = prepCanvas(canvas, height);
  const T = spec.length;
  if (!T) return;
  const bins = Math.min(spec[0].length, Math.round((maxHz / (sr / 2)) * (spec[0].length - 1)) + 1);
  let peak = -Infinity;
  const db = spec.map((p) => {
    const r = new Float32Array(bins);
    for (let k = 0; k < bins; k++) { r[k] = 10 * Math.log10(p[k] + 1e-12); if (r[k] > peak) peak = r[k]; }
    return r;
  });
  const img = imageFrom(T, bins, (d) => {
    for (let t = 0; t < T; t++) for (let k = 0; k < bins; k++) {
      const v = (db[t][k] - (peak - dbRange)) / dbRange;
      const [r, g, b] = colormap(v);
      const o = ((bins - 1 - k) * T + t) * 4;
      d[o] = r; d[o + 1] = g; d[o + 2] = b; d[o + 3] = 255;
    }
  });
  const strip = labels ? 7 : 0;
  const axis = 34;
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(img, axis, strip, w - axis, h - strip);
  if (labels) drawLabelStrip(ctx, labels, axis, w - axis, 0, strip);
  drawRegions(ctx, regions, T * hopSec, axis, w - axis, strip, h - strip);
  ctx.fillStyle = cssVar('--plot-ink');
  for (const f of [0, 2000, 4000, 6000, 8000]) {
    if (f > maxHz) continue;
    const y = strip + (h - strip) * (1 - f / maxHz);
    ctx.fillText(f ? `${f / 1000}k` : '0', 4, Math.min(h - 3, Math.max(strip + 10, y + 4)));
  }
}

/** Heatmap of a T × D matrix (e.g. MFCCs). Column-wise z-scoring optional. */
export function drawMatrix(canvas, rows, { height = 120, signed = true, rowLabel = 'coef' } = {}) {
  const { ctx, w, h } = prepCanvas(canvas, height);
  const T = rows.length;
  if (!T) return;
  const D = rows[0].length;
  let m = 0;
  for (const r of rows) for (let k = 0; k < D; k++) m = Math.max(m, Math.abs(r[k]));
  m = m || 1;
  const img = imageFrom(T, D, (d) => {
    for (let t = 0; t < T; t++) for (let k = 0; k < D; k++) {
      const v = rows[t][k] / m;
      const [r, g, b] = signed ? diverging(v * 1.6) : colormap(v);
      const o = ((D - 1 - k) * T + t) * 4;
      d[o] = r; d[o + 1] = g; d[o + 2] = b; d[o + 3] = 255;
    }
  });
  const axis = 34;
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(img, axis, 0, w - axis, h);
  ctx.fillStyle = cssVar('--plot-ink');
  ctx.fillText(`${rowLabel} 0`, 2, h - 4);
  ctx.fillText(String(D - 1), 2, 11);
}

/** DTW local-cost matrix (reference on y, test on x) with warping path. */
export function drawDtw(canvas, res, { height = 280 } = {}) {
  const { ctx, w, h } = prepCanvas(canvas, height);
  const { C, n, m, path } = res;
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < C.length; i++) { const v = C[i]; if (v > 0) { if (v < lo) lo = v; if (v > hi) hi = v; } }
  const img = imageFrom(m, n, (d) => {
    for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) {
      const c = C[i * m + j];
      const v = c > 0 ? 1 - (c - lo) / (hi - lo || 1) : 0;
      const [r, g, b] = colormap(v * 0.92);
      const o = ((n - 1 - i) * m + j) * 4;
      d[o] = r; d[o + 1] = g; d[o + 2] = b; d[o + 3] = 255;
    }
  });
  const ax = 30, ay = 18;
  const pw = w - ax, ph = h - ay;
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(img, ax, 0, pw, ph);
  ctx.strokeStyle = '#38e0c8';
  ctx.lineWidth = 2;
  ctx.beginPath();
  path.forEach(([i, j], k) => {
    const x = ax + ((j + 0.5) / m) * pw;
    const y = ((n - 1 - i + 0.5) / n) * ph;
    k ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  });
  ctx.stroke();
  ctx.strokeStyle = 'rgba(255,255,255,0.35)';
  ctx.setLineDash([4, 4]);
  ctx.beginPath(); ctx.moveTo(ax, ph); ctx.lineTo(w, 0); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = cssVar('--plot-ink');
  ctx.fillText('student →', w - 64, h - 4);
  ctx.save(); ctx.translate(11, ph / 2 + 30); ctx.rotate(-Math.PI / 2); ctx.fillText('reference →', 0, 0); ctx.restore();
}

/**
 * Multi-series line chart.
 * series: [{ data: number[]|Float32Array, x?: number[], color, label, width, fill }]
 */
export function drawLines(canvas, series, { height = 160, yMin, yMax, xMax, xMin = 0, xFmt = (x) => x.toFixed(1), yFmt = (y) => y.toFixed(1), vlines = [], bands = [], onX } = {}) {
  const { ctx, w, h } = prepCanvas(canvas, height);
  const L = 40, R = 8, Tp = 8, B = 20;
  const pw = w - L - R, ph = h - Tp - B;
  let lo = yMin, hi = yMax;
  if (lo === undefined || hi === undefined) {
    let a = Infinity, b = -Infinity;
    for (const s of series) for (const v of s.data) if (Number.isFinite(v)) { if (v < a) a = v; if (v > b) b = v; }
    if (lo === undefined) lo = a;
    if (hi === undefined) hi = b;
    if (hi === lo) { hi += 1; lo -= 1; }
  }
  const maxX = xMax ?? Math.max(...series.map((s) => (s.x ? s.x[s.x.length - 1] : s.data.length - 1)));
  const X = (x) => L + ((x - xMin) / (maxX - xMin || 1)) * pw;
  const Y = (y) => Tp + (1 - (y - lo) / (hi - lo)) * ph;
  ctx.strokeStyle = cssVar('--plot-grid');
  ctx.fillStyle = cssVar('--plot-ink');
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = lo + ((hi - lo) * i) / 4;
    ctx.beginPath(); ctx.moveTo(L, Y(y)); ctx.lineTo(w - R, Y(y)); ctx.stroke();
    ctx.fillText(yFmt(y), 2, Y(y) + 4);
  }
  for (let i = 0; i <= 5; i++) {
    const x = xMin + ((maxX - xMin) * i) / 5;
    ctx.fillText(xFmt(x), Math.min(w - 36, X(x) - 8), h - 5);
  }
  for (const b of bands) {
    ctx.fillStyle = b.color || 'rgba(194,65,12,0.12)';
    ctx.fillRect(X(b.x0), Tp, Math.max(1, X(b.x1) - X(b.x0)), ph);
  }
  for (const v of vlines) {
    ctx.strokeStyle = v.color || cssVar('--accent-2');
    ctx.lineWidth = v.width || 1;
    ctx.setLineDash(v.dash || []);
    ctx.beginPath(); ctx.moveTo(X(v.x), Tp); ctx.lineTo(X(v.x), Tp + ph); ctx.stroke();
    ctx.setLineDash([]);
  }
  for (const s of series) {
    ctx.strokeStyle = s.color || cssVar('--accent');
    ctx.lineWidth = s.width || 1.6;
    ctx.beginPath();
    let pen = false;
    const n = s.data.length;
    const stride = Math.max(1, Math.floor(n / (pw * 2)));
    for (let i = 0; i < n; i += stride) {
      const v = s.data[i];
      if (!Number.isFinite(v) || (s.skipZero && v === 0)) { pen = false; continue; }
      const x = X(s.x ? s.x[i] : i), y = Y(Math.max(lo, Math.min(hi, v)));
      pen ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
      pen = true;
    }
    ctx.stroke();
    if (s.fill) {
      ctx.lineTo(X(s.x ? s.x[n - 1] : n - 1), Y(lo));
      ctx.lineTo(X(s.x ? s.x[0] : 0), Y(lo));
      ctx.closePath();
      ctx.globalAlpha = 0.14;
      ctx.fillStyle = s.color || cssVar('--accent');
      ctx.fill();
      ctx.globalAlpha = 1;
    }
  }
  if (onX) {
    canvas.onclick = (e) => {
      const r = canvas.getBoundingClientRect();
      const x = xMin + ((e.clientX - r.left - L) / pw) * (maxX - xMin);
      if (x >= xMin && x <= maxX) onX(x);
    };
    canvas.style.cursor = 'pointer';
  }
  return { X, Y };
}

export function drawBars(canvas, values, { height = 150, labels = [], color, negColor, yFmt = (v) => v.toFixed(1), signed = false } = {}) {
  const { ctx, w, h } = prepCanvas(canvas, height);
  const L = 36, B = 18, Tp = 8;
  const pw = w - L - 6, ph = h - B - Tp;
  const max = Math.max(1e-9, ...values.map((v) => Math.abs(v)));
  const n = values.length;
  const bw = pw / n;
  const zero = signed ? Tp + ph / 2 : Tp + ph;
  const scale = signed ? ph / 2 / max : ph / max;
  ctx.strokeStyle = cssVar('--plot-grid');
  ctx.beginPath(); ctx.moveTo(L, zero); ctx.lineTo(w - 6, zero); ctx.stroke();
  values.forEach((v, i) => {
    ctx.fillStyle = v >= 0 ? color || cssVar('--accent') : negColor || cssVar('--accent-2');
    const bh = v * scale;
    ctx.fillRect(L + i * bw + bw * 0.15, v >= 0 ? zero - bh : zero, bw * 0.7, Math.abs(bh));
  });
  ctx.fillStyle = cssVar('--plot-ink');
  const every = Math.ceil(n / Math.max(1, Math.floor(pw / 26)));
  labels.forEach((l, i) => { if (i % every === 0) ctx.fillText(l, L + i * bw + bw * 0.15, h - 4); });
  ctx.fillText(yFmt(signed ? max : max), 2, Tp + 8);
  if (signed) ctx.fillText(yFmt(-max), 2, Tp + ph);
}

/** Small confusion-matrix heat table rendered on canvas. */
export function drawConfusion(canvas, matrix, labels, { height } = {}) {
  const n = labels.length;
  const size = height || Math.min(420, 36 + n * 30);
  const { ctx, w, h } = prepCanvas(canvas, size);
  const L = 46, Tp = 6, B = 30;
  const cell = Math.min((w - L - 6) / n, (h - Tp - B) / n);
  const max = Math.max(1, ...matrix.flat());
  ctx.textAlign = 'center';
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    const v = matrix[i][j];
    const t = v / max;
    ctx.fillStyle = i === j ? `rgba(21,128,61,${0.15 + 0.75 * t})` : v ? `rgba(185,28,28,${0.2 + 0.7 * t})` : cssVar('--surface-2');
    ctx.fillRect(L + j * cell + 1, Tp + i * cell + 1, cell - 2, cell - 2);
    if (v) {
      ctx.fillStyle = t > 0.5 ? '#fff' : cssVar('--plot-ink');
      ctx.fillText(String(v), L + j * cell + cell / 2, Tp + i * cell + cell / 2 + 4);
    }
  }
  ctx.fillStyle = cssVar('--plot-ink');
  labels.forEach((l, i) => {
    ctx.fillText(l, L + i * cell + cell / 2, Tp + n * cell + 14);
    ctx.textAlign = 'right';
    ctx.fillText(l, L - 6, Tp + i * cell + cell / 2 + 4);
    ctx.textAlign = 'center';
  });
  ctx.fillText('predicted →', L + (n * cell) / 2, Tp + n * cell + 27);
  ctx.textAlign = 'left';
}
