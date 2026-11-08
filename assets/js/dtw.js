// Dynamic Time Warping with the symmetric-2 step pattern and an optional
// Sakoe–Chiba band. Returns the local cost matrix and the optimal path so the
// UI can show *where* two utterances differ, not only how much.

export function euclidean(a, b, from = 0, to = a.length) {
  let s = 0;
  for (let k = from; k < to; k++) {
    const d = a[k] - b[k];
    s += d * d;
  }
  return Math.sqrt(s);
}

/**
 * @param {Float32Array[]} A reference sequence (rows = frames)
 * @param {Float32Array[]} B test sequence
 * @param {{band?: number, from?: number, to?: number}} opts band = fraction of length
 */
export function dtw(A, B, { band = 0.25, from = 0, to } = {}) {
  const n = A.length;
  const m = B.length;
  const dimTo = to ?? A[0].length;
  const C = new Float32Array(n * m);
  const D = new Float64Array(n * m).fill(Infinity);
  const w = band == null ? Infinity : Math.max(Math.ceil(band * Math.max(n, m)), Math.abs(n - m) + 2);
  const inBand = (i, j) => Math.abs((i * (m - 1)) / Math.max(1, n - 1) - j) <= w;

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < m; j++) {
      if (!inBand(i, j)) continue;
      const c = euclidean(A[i], B[j], from, dimTo);
      C[i * m + j] = c;
      if (i === 0 && j === 0) { D[0] = 2 * c; continue; }
      let best = Infinity;
      if (i > 0 && j > 0) best = D[(i - 1) * m + j - 1] + 2 * c;
      if (i > 0) best = Math.min(best, D[(i - 1) * m + j] + c);
      if (j > 0) best = Math.min(best, D[i * m + j - 1] + c);
      D[i * m + j] = best;
    }
  }

  // backtrack
  const path = [];
  let i = n - 1, j = m - 1;
  path.push([i, j]);
  while (i > 0 || j > 0) {
    if (i === 0) j--;
    else if (j === 0) i--;
    else {
      const diag = D[(i - 1) * m + j - 1];
      const up = D[(i - 1) * m + j];
      const left = D[i * m + j - 1];
      if (diag <= up && diag <= left) { i--; j--; }
      else if (up <= left) i--;
      else j--;
    }
    path.push([i, j]);
  }
  path.reverse();
  const distance = D[n * m - 1] / (n + m);
  return { distance, path, C, n, m };
}

/** Average local cost along the path for every reference frame. */
export function costPerReferenceFrame(result) {
  const { path, C, n, m } = result;
  const sum = new Float64Array(n);
  const cnt = new Float64Array(n);
  for (const [i, j] of path) { sum[i] += C[i * m + j]; cnt[i]++; }
  return Float32Array.from(sum, (s, i) => (cnt[i] ? s / cnt[i] : 0));
}

/** Map each reference frame to the mean matching test frame index. */
export function referenceToTestMap(result) {
  const { path, n } = result;
  const sum = new Float64Array(n);
  const cnt = new Float64Array(n);
  for (const [i, j] of path) { sum[i] += j; cnt[i]++; }
  return Float32Array.from(sum, (s, i) => (cnt[i] ? s / cnt[i] : 0));
}
