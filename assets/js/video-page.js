import { $, $$, el, toast, fmtTime, onThemeChange, cssVar, yieldUI, download, setPressed } from './ui.js';
import * as V from './video-core.js';
import { drawLines, colormap, prepCanvas } from './plot.js';

const video = $('#video');
const overlay = $('#overlay');
const state = {
  name: '', isDemo: false, truth: null, a: null, running: false, cancel: false,
  summary: { playing: false, idx: 0 }, kfSort: 'time', tab: 'frames', model: null, modelKind: null,
};
const TRACK_COLORS = ['#f97316', '#3b82f6', '#22c55e', '#e11d48', '#a855f7', '#eab308', '#06b6d4', '#f43f5e', '#84cc16', '#6366f1'];
const trackColor = (id) => TRACK_COLORS[(id - 1) % TRACK_COLORS.length];

// ------------------------------------------------------------ loading ---
function setSource(url, name, isDemo) {
  stopSummary();
  state.name = name;
  state.isDemo = isDemo;
  state.truth = null;
  state.a = null;
  $('#results').classList.add('hidden');
  clearOverlay();
  video.src = url;
  $('#src-info').textContent = 'Loading…';
  video.addEventListener('loadedmetadata', () => {
    $('#src-info').textContent = `${name} · ${fmtTime(video.duration)} · ${video.videoWidth}×${video.videoHeight}`;
    $('#run').disabled = false;
    if (isDemo) $('#tracker').value = 'blob';
  }, { once: true });
  video.addEventListener('error', () => {
    $('#src-info').textContent = 'This browser cannot play that file. Try MP4 (H.264) or WebM.';
    $('#run').disabled = true;
  }, { once: true });
  if (isDemo) fetch('media/demo_truth.json').then((r) => r.json()).then((t) => (state.truth = t)).catch(() => {});
}

function bindSource() {
  const drop = $('#drop');
  const input = $('#file');
  const pick = (f) => {
    if (!f) return;
    if (!f.type.startsWith('video/') && !/\.(mp4|webm|mov|mkv|m4v|ogv)$/i.test(f.name)) return toast('Please choose a video file.');
    setSource(URL.createObjectURL(f), f.name, false);
  };
  drop.addEventListener('click', () => input.click());
  drop.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') input.click(); });
  input.addEventListener('change', () => pick(input.files[0]));
  drop.addEventListener('dragover', (e) => { e.preventDefault(); drop.classList.add('over'); });
  drop.addEventListener('dragleave', () => drop.classList.remove('over'));
  drop.addEventListener('drop', (e) => { e.preventDefault(); drop.classList.remove('over'); pick(e.dataTransfer.files[0]); });
  $('#demo').addEventListener('click', async () => {
    // Load the demo as a blob: object URLs are always seekable, even from
    // servers that do not support HTTP range requests.
    $('#src-info').textContent = 'Downloading demo video…';
    try {
      const blob = await (await fetch('media/demo.mp4')).blob();
      setSource(URL.createObjectURL(blob), 'demo.mp4', true);
    } catch (e) {
      $('#src-info').textContent = `Could not load the demo: ${e.message}`;
    }
  });
  $('#sens').addEventListener('input', (e) => ($('#sens-v').textContent = e.target.value));
}

function loadScript(src) {
  return new Promise((res, rej) => {
    if ([...document.scripts].some((s) => s.src === src)) return res();
    const s = el('script', { src, crossorigin: 'anonymous' });
    s.onload = res;
    s.onerror = () => rej(new Error(`failed to load ${src}`));
    document.head.append(s);
  });
}

async function loadDetector() {
  if (state.model) return state.model;
  await loadScript('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js');
  await loadScript('https://cdn.jsdelivr.net/npm/@tensorflow-models/coco-ssd@2.2.3/dist/coco-ssd.min.js');
  state.model = await window.cocoSsd.load({ base: 'lite_mobilenet_v2' });
  return state.model;
}

function seekTo(t) {
  return new Promise((resolve) => {
    let done = false;
    const finish = () => {
      if (done) return;
      done = true;
      video.removeEventListener('seeked', finish);
      clearTimeout(timer);
      resolve();
    };
    const timer = setTimeout(finish, 5000);
    video.addEventListener('seeked', finish);
    video.currentTime = t;
  });
}

function autoRate(duration) {
  if (duration <= 180) return 4;
  if (duration <= 600) return 2;
  if (duration <= 2400) return 1;
  return 0.5;
}

// ------------------------------------------------------------ analysis ---
async function analyse() {
  if (state.running) { state.cancel = true; return; }
  stopSummary();
  const dur = video.duration;
  if (!Number.isFinite(dur) || dur <= 0) return toast('Load a video first.');
  const seekEnd = video.seekable.length ? video.seekable.end(video.seekable.length - 1) : 0;
  if (seekEnd < dur - 1) return toast('This video source is not seekable, so frames cannot be sampled. Choose a local file instead.');
  state.running = true;
  state.cancel = false;
  const runBtn = $('#run');
  runBtn.textContent = 'Cancel';
  runBtn.classList.remove('primary');
  const prog = $('#prog');
  prog.classList.remove('hidden');
  const bar = prog.firstElementChild;
  const status = $('#run-status');
  const t0 = performance.now();
  const wasMuted = video.muted;
  video.pause();
  video.muted = true;

  try {
    const rateSel = $('#rate').value;
    let rate = rateSel === 'auto' ? autoRate(dur) : Number(rateSel);
    let n = Math.max(2, Math.floor(dur * rate));
    if (n > 3600) { n = 3600; rate = n / dur; }
    const dt = dur / n;
    const vw = video.videoWidth, vh = video.videoHeight;
    const aw = 160, ah = Math.max(8, Math.min(200, Math.round((160 * vh) / vw)));
    const cA = el('canvas', { width: aw, height: ah });
    const gA = cA.getContext('2d', { willReadFrequently: true });

    let mode = $('#tracker').value;
    let detCanvas, gD, model;
    if (mode.startsWith('coco')) {
      status.textContent = 'Loading the COCO-SSD detector (first time only)…';
      try {
        model = await loadDetector();
        const dw = 400, dh = Math.round((400 * vh) / vw);
        detCanvas = el('canvas', { width: dw, height: dh });
        gD = detCanvas.getContext('2d');
      } catch (e) {
        toast('Could not load the detector. Falling back to moving-blob tracking.');
        mode = 'blob';
      }
    }

    // ---- pass 1: sample frames
    const samples = [];
    const tLoop = performance.now();
    for (let i = 0; i < n; i++) {
      if (state.cancel) throw new Error('cancelled');
      const t = Math.min(dur - 0.04, i * dt + Math.min(0.04, dt / 4));
      await seekTo(t);
      gA.drawImage(video, 0, 0, aw, ah);
      const px = gA.getImageData(0, 0, aw, ah).data;
      const gray = V.toGray(px);
      const s = { t, hist: V.hsvHistogram(px), gray, sharp: V.sharpness(gray, aw, ah), dets: [] };
      if (model) {
        gD.drawImage(video, 0, 0, detCanvas.width, detCanvas.height);
        const preds = await model.detect(detCanvas, 20, 0.45);
        s.dets = preds
          .filter((p) => mode === 'coco-all' || p.class === 'person')
          .map((p) => ({ x: p.bbox[0] / detCanvas.width, y: p.bbox[1] / detCanvas.height, w: p.bbox[2] / detCanvas.width, h: p.bbox[3] / detCanvas.height, score: p.score, label: p.class }));
      }
      samples.push(s);
      if (i % 4 === 0 || i === n - 1) {
        const el2 = (performance.now() - tLoop) / 1000;
        const eta = (el2 / (i + 1)) * (n - i - 1);
        bar.style.width = `${((i + 1) / n) * 85}%`;
        status.textContent = `Sampling frame ${i + 1} / ${n} at ${fmtTime(t)} · about ${fmtTime(eta)} left`;
        await yieldUI();
      }
    }

    // ---- shot boundaries, scenes, motion
    status.textContent = 'Detecting scenes and computing motion…';
    await yieldUI();
    const k = 4.5 - 0.5 * Number($('#sens').value);
    const { boundaries, D, thr } = V.detectBoundaries(samples, { k, minSceneSec: Math.max(1.5, 2.5 * dt) });
    const scenes = V.scenesFromBoundaries(boundaries, samples, dur);
    const motion = V.motionAnalysis(samples, boundaries, aw, ah);
    // camera motion (global translation) between consecutive samples of a scene
    const cutSet = new Set(boundaries.map((b) => b.index));
    for (let i = 1; i < samples.length; i++) {
      if (cutSet.has(i)) continue;
      samples[i].cam = V.globalShift(samples[i - 1].gray, samples[i].gray, aw, ah);
      if (i % 50 === 0) await yieldUI();
    }
    for (const sc of scenes) {
      // a pan/tilt moves the whole frame consistently in one direction;
      // large moving objects can fool single pairs but not the consensus
      const shifts = [];
      let c = 0;
      for (let i = sc.start + 1; i < sc.end; i++) if (samples[i].cam) { c++; if (samples[i].cam.moving) shifts.push(samples[i].cam); }
      const agree = (key) => {
        const pos = shifts.filter((m) => m[key] > 0).length, neg = shifts.filter((m) => m[key] < 0).length;
        return Math.max(pos, neg) / Math.max(1, shifts.length);
      };
      sc.cameraMoving = c > 2 && shifts.length / c > 0.5 && Math.max(agree('dx'), agree('dy')) > 0.85;
    }
    const meanGray = new Float32Array(aw * ah);
    for (const s of samples) for (let p = 0; p < meanGray.length; p++) meanGray[p] += s.gray[p] / samples.length;

    // ---- detections for blob mode
    if (mode === 'blob') {
      status.textContent = 'Background subtraction per scene…';
      await yieldUI();
      for (const sc of scenes) {
        if (sc.cameraMoving) continue; // a static background model is meaningless while the camera pans
        const bg = V.sceneBackground(samples, sc, aw, ah);
        for (let i = sc.start; i < sc.end; i++) {
          const r = V.detectBlobs(samples[i].gray, bg, aw, ah);
          samples[i].dets = r.boxes;
          samples[i].cameraMotion = r.cameraMotion;
        }
      }
    }

    // ---- tracking (reset at every scene)
    let tracks = [];
    if (mode !== 'none') {
      for (const sc of scenes) {
        const tr = new V.Tracker({ gate: 0.2, maxMissed: 2 });
        for (let i = sc.start; i < sc.end; i++) tr.update(samples[i].dets, samples[i].t);
        tr.flush();
        tracks.push(...tr.tracks(3, 1.0).map((x) => ({ ...x, scene: sc.id })));
      }
      tracks.sort((a, b) => a.t0 - b.t0);
      tracks.forEach((t, i) => { t.displayId = i + 1; t.ref.displayId = i + 1; });
    }

    // ---- key frames, importance, summary
    const keyframes = V.selectKeyframes(samples, scenes, { minGapSec: Math.max(8, dur / 60) });
    const globalHist = V.globalMeanHist(samples);
    const targetSec = Math.min(Number($('#sumlen').value), dur * 0.5);
    const segments = V.planSummary(scenes, samples, motion.energy, { targetSec, globalHist });
    for (const kf of keyframes) kf.importance = scenes.find((s) => s.id === kf.scene).importance;

    // ---- pass 2: thumbnails for key frames
    const tw = 320, th = Math.round((320 * vh) / vw);
    const cT = el('canvas', { width: tw, height: th });
    const gT = cT.getContext('2d');
    const shown = keyframes.slice(0, 150);
    for (let i = 0; i < shown.length; i++) {
      if (state.cancel) throw new Error('cancelled');
      await seekTo(shown[i].t);
      gT.drawImage(video, 0, 0, tw, th);
      shown[i].img = cT.toDataURL('image/jpeg', 0.82);
      bar.style.width = `${85 + ((i + 1) / shown.length) * 15}%`;
      status.textContent = `Capturing key frame ${i + 1} / ${shown.length}`;
      if (i % 3 === 0) await yieldUI();
    }

    // free the heavy per-sample buffers
    for (const s of samples) { delete s.gray; delete s.hist; delete s.cam; }
    for (const sc of scenes) delete sc.meanHist;

    state.a = {
      dur, rate, dt, n, aw, ah, vw, vh, mode, samples, boundaries, D, thr, scenes, motion, meanGray, tracks, keyframes: shown, segments,
      ms: performance.now() - t0,
    };
    bar.style.width = '100%';
    status.textContent = `Done: ${n} frames in ${((performance.now() - t0) / 1000).toFixed(1)} s.`;
    await seekTo(0);
    renderAll();
  } catch (e) {
    if (e.message === 'cancelled') status.textContent = 'Analysis cancelled.';
    else { console.error(e); status.textContent = `Analysis failed: ${e.message}`; }
  } finally {
    state.running = false;
    video.muted = wasMuted;
    runBtn.textContent = 'Analyse video';
    runBtn.classList.add('primary');
  }
}

// ------------------------------------------------------------ rendering ---
function stat(k, v, d) {
  return el('div', { class: 'stat' }, el('div', { class: 'k', text: k }), el('div', { class: 'v', text: v }), el('div', { class: 'd', text: d || '' }));
}

function renderAll() {
  const a = state.a;
  $('#results').classList.remove('hidden');
  fillTrackSceneSelect();
  const st = $('#stats');
  st.innerHTML = '';
  const cuts = a.boundaries.filter((b) => b.type === 'cut').length;
  const people = a.tracks.filter((t) => t.label === 'person').length;
  st.append(
    stat('Frames analysed', String(a.n), `${a.rate.toFixed(2)} fps over ${fmtTime(a.dur)}`),
    stat('Scenes', String(a.scenes.length), `${cuts} cuts · ${a.boundaries.length - cuts} gradual`),
    stat('Tracked objects', String(a.tracks.length), a.mode === 'none' ? 'tracking off' : `${people} people · ${a.mode.startsWith('coco') ? 'COCO-SSD' : 'blobs'}`),
    stat('Processing time', `${(a.ms / 1000).toFixed(0)} s`, `${((a.dur * 1000) / a.ms).toFixed(1)}× real time`),
  );
  const ts = $('#truth-score');
  ts.innerHTML = '';
  if (state.truth) {
    const sc = V.scoreBoundaries(a.boundaries, state.truth.boundaries, Math.max(1.5, 2 * a.dt));
    ts.append(el('span', { class: 'tag indigo', text: 'Demo ground truth' }), ' ',
      `Scene boundaries: ${sc.tp}/${state.truth.boundaries.length} found · precision ${(sc.precision * 100).toFixed(0)}% · recall ${(sc.recall * 100).toFixed(0)}% · F1 ${(sc.f1 * 100).toFixed(0)}%`);
  }
  renderTab();
}

function renderTab() {
  if (!state.a) return;
  ({ frames: renderFrames, scenes: renderScenes, motion: renderMotion, tracks: renderTracks, summary: renderSummary })[state.tab]();
}

function renderFrames() {
  const a = state.a;
  const list = [...a.keyframes];
  if (state.kfSort === 'importance') list.sort((x, y) => y.importance - x.importance);
  const rank = [...a.keyframes].sort((x, y) => y.importance - x.importance);
  const box = $('#thumbs');
  box.innerHTML = '';
  for (const kf of list) {
    const r = rank.indexOf(kf) + 1;
    box.append(el('div', { class: 'thumb', onclick: () => jump(kf.t) },
      el('img', { src: kf.img, alt: `Key frame at ${fmtTime(kf.t)}`, loading: 'lazy' }),
      el('span', { class: 'badge', text: `#${r}` }),
      el('div', { class: 'cap' }, el('span', { class: 'mono', text: fmtTime(kf.t) }), el('span', { class: 'muted', text: `scene ${kf.scene} · ${(kf.importance * 100).toFixed(0)}%` }))));
  }
}

function renderScenes() {
  const a = state.a;
  // timeline
  const c = $('#c-timeline');
  const { ctx, w, h } = prepCanvas(c, 64);
  const X = (t) => (t / a.dur) * w;
  const maxImp = Math.max(...a.scenes.map((s) => s.importance));
  a.scenes.forEach((s, i) => {
    const [r, g, b] = colormap(0.25 + 0.7 * (s.importance / maxImp));
    ctx.fillStyle = `rgb(${r},${g},${b})`;
    ctx.fillRect(X(s.t0), 14, Math.max(1, X(s.t1) - X(s.t0) - 1), 34);
    if (X(s.t1) - X(s.t0) > 18) {
      ctx.fillStyle = s.importance / maxImp > 0.6 ? '#111' : '#fff';
      ctx.font = '11px JetBrains Mono, monospace';
      ctx.fillText(String(s.id), X(s.t0) + 4, 35);
    }
  });
  if (state.truth) {
    ctx.fillStyle = cssVar('--accent-2');
    for (const b of state.truth.boundaries) {
      ctx.beginPath(); ctx.moveTo(X(b.time) - 5, 2); ctx.lineTo(X(b.time) + 5, 2); ctx.lineTo(X(b.time), 11); ctx.fill();
    }
    $('#truth-legend').textContent = '▼ ground-truth boundary';
  } else $('#truth-legend').textContent = '';
  ctx.fillStyle = cssVar('--plot-ink');
  ctx.font = '11px Inter, sans-serif';
  ctx.fillText('0:00', 2, h - 3);
  ctx.fillText(fmtTime(a.dur), w - 36, h - 3);
  c.onclick = (e) => jump((e.offsetX / c.clientWidth) * a.dur);
  c.style.cursor = 'pointer';

  const times = a.samples.map((s) => s.t);
  drawLines($('#c-diff'), [
    { data: Array.from(a.D), x: times, color: cssVar('--accent'), width: 1.4 },
    { data: Array.from(a.thr), x: times, color: cssVar('--muted'), width: 1 },
  ], {
    height: 170, yMin: 0, yMax: Math.max(0.6, ...a.D), xMax: a.dur, xFmt: fmtTime, yFmt: (y) => y.toFixed(2),
    vlines: a.boundaries.map((b) => ({ x: b.time, color: b.type === 'cut' ? cssVar('--accent-2') : cssVar('--good'), dash: b.type === 'cut' ? [] : [4, 3] })),
    onX: jump,
  });

  const t = $('#scene-table');
  t.innerHTML = '';
  t.append(el('tr', {}, ...['#', 'Start', 'End', 'Length', 'Starts with', 'Camera', 'Motion', 'Importance'].map((x, i) => el('th', { class: i >= 6 ? 'num' : '', text: x }))));
  a.scenes.forEach((s, i) => {
    const b = a.boundaries[i - 1];
    t.append(el('tr', { style: 'cursor:pointer', onclick: () => jump(s.t0 + 0.05) },
      el('td', { class: 'mono', text: String(s.id) }), el('td', { class: 'mono', text: fmtTime(s.t0) }), el('td', { class: 'mono', text: fmtTime(s.t1) }),
      el('td', { class: 'mono', text: `${(s.t1 - s.t0).toFixed(1)} s` }),
      el('td', {}, i === 0 ? 'start of video' : el('span', { class: `tag ${b.type === 'cut' ? 'indigo' : 'good'}`, text: b.type === 'cut' ? 'hard cut' : 'gradual' })),
      el('td', { text: s.cameraMoving ? 'moving' : 'static' }),
      el('td', { class: 'num', text: (s.motion * 100).toFixed(1) }), el('td', { class: 'num', text: `${(s.importance * 100).toFixed(0)}%` })));
  });
}

/** Draw the mean frame (grayscale) as a backdrop. */
function drawBackdrop(ctx, w, h, dim = 0.55) {
  const a = state.a;
  const img = new ImageData(a.aw, a.ah);
  for (let p = 0; p < a.meanGray.length; p++) {
    const v = a.meanGray[p] * dim;
    img.data[p * 4] = v; img.data[p * 4 + 1] = v; img.data[p * 4 + 2] = v; img.data[p * 4 + 3] = 255;
  }
  const tmp = el('canvas', { width: a.aw, height: a.ah });
  tmp.getContext('2d').putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(tmp, 0, 0, w, h);
}

function heatOver(ctx, w, h, field, fw, fh) {
  const img = new ImageData(fw, fh);
  for (let p = 0; p < field.length; p++) {
    const v = Math.pow(field[p], 0.6);
    const [r, g, b] = colormap(v);
    img.data[p * 4] = r; img.data[p * 4 + 1] = g; img.data[p * 4 + 2] = b; img.data[p * 4 + 3] = Math.round(235 * Math.min(1, v * 1.4));
  }
  const tmp = el('canvas', { width: fw, height: fh });
  tmp.getContext('2d').putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(tmp, 0, 0, w, h);
}

function renderMotion() {
  const a = state.a;
  const c = $('#c-motionmap');
  const wPx = c.clientWidth || 400;
  const { ctx, w, h } = prepCanvas(c, Math.round((wPx * a.ah) / a.aw));
  drawBackdrop(ctx, w, h);
  heatOver(ctx, w, h, a.motion.map, a.aw, a.ah);

  const times = a.samples.map((s) => s.t);
  const e = Array.from(a.motion.energy, (v) => v * 100);
  drawLines($('#c-motion'), [{ data: e, x: times, color: cssVar('--accent'), fill: true }], {
    height: 170, yMin: 0, xMax: a.dur, xFmt: fmtTime, yFmt: (y) => y.toFixed(1),
    vlines: a.boundaries.map((b) => ({ x: b.time, color: cssVar('--plot-grid') })), onX: jump,
  });
  // top moments (one per scene at most)
  const best = a.scenes.map((s) => {
    let bi = s.start, bm = -1;
    for (let i = s.start; i < s.end; i++) if (a.motion.energy[i] > bm) { bm = a.motion.energy[i]; bi = i; }
    return { t: a.samples[bi].t, m: bm, scene: s.id };
  }).sort((x, y) => y.m - x.m).slice(0, 5);
  const box = $('#motion-top');
  box.innerHTML = '';
  best.forEach((b) => box.append(el('div', { class: 'row-between', style: 'padding:5px 0;border-bottom:1px solid var(--line);cursor:pointer', onclick: () => jump(b.t) },
    el('span', {}, el('span', { class: 'mono', text: fmtTime(b.t) }), `  scene ${b.scene}`), el('span', { class: 'mono muted', text: (b.m * 100).toFixed(1) }))));
}

const imgCache = new Map();
function loadImg(src) {
  if (!imgCache.has(src)) imgCache.set(src, new Promise((r) => { const i = new Image(); i.onload = () => r(i); i.onerror = () => r(null); i.src = src; }));
  return imgCache.get(src);
}

function fillTrackSceneSelect() {
  const a = state.a;
  const sel = $('#track-scene');
  const prev = sel.value;
  sel.innerHTML = '';
  sel.append(el('option', { value: 'all', text: `All scenes (${a.tracks.length} tracks)` }));
  for (const sc of a.scenes) {
    const n = a.tracks.filter((t) => t.scene === sc.id).length;
    if (n) sel.append(el('option', { value: String(sc.id), text: `Scene ${sc.id} (${n} track${n > 1 ? 's' : ''})` }));
  }
  sel.value = [...sel.options].some((o) => o.value === prev) ? prev : 'all';
}

async function renderTracks() {
  const a = state.a;
  if (!$('#track-scene').options.length) fillTrackSceneSelect();
  const pick = $('#track-scene').value;
  const tracks = pick === 'all' ? a.tracks : a.tracks.filter((t) => String(t.scene) === pick);
  const kf = pick === 'all' ? null : a.keyframes.find((k) => String(k.scene) === pick && k.primary) || a.keyframes.find((k) => String(k.scene) === pick);
  const bgImg = kf?.img ? await loadImg(kf.img) : null;
  const t = $('#track-table');
  t.innerHTML = '';
  const c1 = $('#c-traj'), c2 = $('#c-density');
  const wPx = c1.clientWidth || 400;
  const H = Math.round((wPx * a.ah) / a.aw);
  const { ctx, w, h } = prepCanvas(c1, H);
  const d2 = prepCanvas(c2, H);
  for (const g of [ctx, d2.ctx]) {
    if (bgImg) { g.globalAlpha = 0.55; g.drawImage(bgImg, 0, 0, w, h); g.globalAlpha = 1; } else drawBackdrop(g, w, h, 0.5);
  }
  if (!tracks.length) {
    ctx.fillStyle = '#fff';
    ctx.fillText(a.mode === 'none' ? 'Tracking was switched off.' : 'No objects were tracked for 3+ samples.', 12, 22);
    t.append(el('tr', {}, el('td', { text: a.mode.startsWith('coco') ? 'No tracks. Try "Moving blobs" for synthetic or non-human content.' : 'No tracks found.' })));
    return;
  }
  // density field of all track centres
  const fw = 64, fh = Math.max(8, Math.round((64 * a.ah) / a.aw));
  const field = new Float32Array(fw * fh);
  for (const tr of tracks) for (const p of tr.points) {
    const cx = (p.box.x + p.box.w / 2) * fw, cy = (p.box.y + p.box.h / 2) * fh;
    for (let y = 0; y < fh; y++) for (let x = 0; x < fw; x++) field[y * fw + x] += Math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / 8);
  }
  const mx = Math.max(...field);
  for (let i = 0; i < field.length; i++) field[i] /= mx || 1;
  heatOver(d2.ctx, d2.w, d2.h, field, fw, fh);

  for (const tr of tracks) {
    const col = trackColor(tr.displayId);
    ctx.strokeStyle = col;
    ctx.fillStyle = col;
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    tr.points.forEach((p, i) => {
      const x = (p.box.x + p.box.w / 2) * w, y = (p.box.y + p.box.h / 2) * h;
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.stroke();
    const p0 = tr.points[0].box, p1 = tr.points[tr.points.length - 1].box;
    ctx.beginPath(); ctx.arc((p0.x + p0.w / 2) * w, (p0.y + p0.h / 2) * h, 4, 0, 7); ctx.fill();
    const ex = (p1.x + p1.w / 2) * w, ey = (p1.y + p1.h / 2) * h;
    ctx.beginPath(); ctx.moveTo(ex, ey - 6); ctx.lineTo(ex - 5, ey + 4); ctx.lineTo(ex + 5, ey + 4); ctx.fill();
    ctx.font = '11px JetBrains Mono, monospace';
    ctx.fillText(`#${tr.displayId}`, ex + 6, ey - 4);
  }
  t.append(el('tr', {}, ...['ID', 'Class', 'Scene', 'Visible', 'Duration', 'Path', 'Speed', 'Direction'].map((x, i) => el('th', { class: i >= 4 && i <= 6 ? 'num' : '', text: x }))));
  for (const tr of tracks) {
    const px = tr.path * a.vw;
    t.append(el('tr', { style: 'cursor:pointer', onclick: () => jump(tr.t0) },
      el('td', {}, el('span', { class: 'mono', style: `color:${trackColor(tr.displayId)};font-weight:700`, text: `#${tr.displayId}` })),
      el('td', { text: tr.label }), el('td', { class: 'mono', text: String(tr.scene) }),
      el('td', { class: 'mono', text: `${fmtTime(tr.t0)}–${fmtTime(tr.t1)}` }),
      el('td', { class: 'num', text: `${tr.duration.toFixed(1)} s` }), el('td', { class: 'num', text: `${Math.round(px)} px` }),
      el('td', { class: 'num', text: `${Math.round(tr.duration ? px / tr.duration : 0)} px/s` }),
      el('td', { text: V.describeDirection(tr.dx, tr.dy) })));
  }
}

function renderSummary() {
  const a = state.a;
  const total = a.segments.reduce((s, x) => s + x.end - x.start, 0);
  $('#sum-info').textContent = `${a.segments.length} clips · ${total.toFixed(1)} s from ${fmtTime(a.dur)} (${((total / a.dur) * 100).toFixed(1)}%)`;
  const c = $('#c-sum');
  const { ctx, w } = prepCanvas(c, 46);
  const X = (t) => (t / a.dur) * w;
  ctx.fillStyle = cssVar('--surface-2');
  ctx.fillRect(0, 10, w, 22);
  a.boundaries.forEach((b) => { ctx.fillStyle = cssVar('--plot-grid'); ctx.fillRect(X(b.time), 10, 1, 22); });
  ctx.fillStyle = cssVar('--accent');
  a.segments.forEach((s) => ctx.fillRect(X(s.start), 10, Math.max(2, X(s.end) - X(s.start)), 22));
  const t = $('#sum-table');
  t.innerHTML = '';
  t.append(el('tr', {}, ...['Clip', 'Scene', 'From', 'To', 'Length', 'Scene importance'].map((x, i) => el('th', { class: i >= 4 ? 'num' : '', text: x }))));
  a.segments.forEach((s, i) => t.append(el('tr', { style: 'cursor:pointer', onclick: () => jump(s.start) },
    el('td', { class: 'mono', text: String(i + 1) }), el('td', { class: 'mono', text: String(s.scene) }), el('td', { class: 'mono', text: fmtTime(s.start) }),
    el('td', { class: 'mono', text: fmtTime(s.end) }), el('td', { class: 'num', text: `${(s.end - s.start).toFixed(1)} s` }), el('td', { class: 'num', text: `${(s.importance * 100).toFixed(0)}%` }))));
}

// ------------------------------------------------------------ playback ---
function jump(t) {
  stopSummary();
  video.currentTime = Math.max(0, Math.min(video.duration - 0.05, t));
  video.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

function clearOverlay() {
  const ctx = overlay.getContext('2d');
  ctx.clearRect(0, 0, overlay.width, overlay.height);
}

/** Video content rectangle inside the element (object-fit: contain). */
function contentRect() {
  const W = video.clientWidth, H = video.clientHeight;
  const ar = video.videoWidth / video.videoHeight || 16 / 9;
  let w = W, h = W / ar;
  if (h > H) { h = H; w = H * ar; }
  return { x: (W - w) / 2, y: (H - h) / 2, w, h };
}

function drawOverlay() {
  const a = state.a;
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const W = overlay.clientWidth, H = overlay.clientHeight;
  if (overlay.width !== Math.round(W * dpr)) { overlay.width = Math.round(W * dpr); overlay.height = Math.round(H * dpr); }
  const ctx = overlay.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  const t = video.currentTime;
  if (state.summary.playing) {
    const seg = a.segments[state.summary.idx];
    if (seg) {
      ctx.fillStyle = 'rgba(0,0,0,0.6)';
      ctx.fillRect(10, 10, 210, 26);
      ctx.fillStyle = '#fff';
      ctx.font = '600 13px Inter, sans-serif';
      ctx.fillText(`Summary ${state.summary.idx + 1}/${a.segments.length} · scene ${seg.scene}`, 18, 28);
    }
  }
  const sc = a.scenes.find((s) => t >= s.t0 && t < s.t1);
  $('#now-scene').textContent = sc ? `Scene ${sc.id} of ${a.scenes.length}` : '';
  if (!$('#show-boxes').checked || a.mode === 'none') return;
  // nearest sample
  let lo = 0, hi = a.samples.length - 1;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (a.samples[mid].t < t) lo = mid + 1; else hi = mid; }
  const cand = [a.samples[lo], a.samples[lo - 1]].filter(Boolean).sort((x, y) => Math.abs(x.t - t) - Math.abs(y.t - t))[0];
  if (!cand || Math.abs(cand.t - t) > a.dt * 0.75) return;
  const r = contentRect();
  ctx.lineWidth = 2;
  ctx.font = '600 12px Inter, sans-serif';
  for (const d of cand.dets || []) {
    const id = d.trackRef?.displayId;
    const col = id ? trackColor(id) : 'rgba(255,255,255,0.7)';
    const x = r.x + d.x * r.w, y = r.y + d.y * r.h;
    ctx.strokeStyle = col;
    ctx.strokeRect(x, y, d.w * r.w, d.h * r.h);
    const tag = `${id ? `#${id} ` : ''}${d.label}${d.score < 1 ? ` ${(d.score * 100).toFixed(0)}%` : ''}`;
    const tw = ctx.measureText(tag).width + 8;
    ctx.fillStyle = col;
    ctx.fillRect(x, y - 18, tw, 18);
    ctx.fillStyle = '#000';
    ctx.fillText(tag, x + 4, y - 5);
  }
}

function overlayLoop() {
  if (state.a) drawOverlay();
  if (state.summary.playing) checkSummary();
  if (video.requestVideoFrameCallback && !video.paused) video.requestVideoFrameCallback(overlayLoop);
}
video.addEventListener('play', () => overlayLoop());
video.addEventListener('seeked', () => state.a && !state.running && drawOverlay());
video.addEventListener('timeupdate', () => { if (!video.requestVideoFrameCallback) overlayLoop(); });
$('#show-boxes').addEventListener('change', () => state.a && drawOverlay());

async function startSummary() {
  const a = state.a;
  if (!a?.segments.length) return;
  if (state.summary.playing) return stopSummary();
  state.summary = { playing: true, idx: 0 };
  $('#play-sum').textContent = '■ Stop summary';
  video.scrollIntoView({ behavior: 'smooth', block: 'center' });
  await seekTo(a.segments[0].start);
  video.play();
}
function checkSummary() {
  const a = state.a;
  const seg = a.segments[state.summary.idx];
  if (!seg) return stopSummary();
  if (video.currentTime >= seg.end - 0.03) {
    state.summary.idx++;
    const next = a.segments[state.summary.idx];
    if (!next) { video.pause(); return stopSummary(); }
    video.currentTime = next.start;
  }
}
function stopSummary() {
  if (!state.summary.playing) return;
  state.summary.playing = false;
  $('#play-sum').textContent = '▶ Play summary';
  video.pause();
}

async function exportSummary() {
  const a = state.a;
  if (!a?.segments.length) return;
  if (!window.MediaRecorder || !HTMLCanvasElement.prototype.captureStream) return toast('Recording is not supported in this browser.');
  stopSummary();
  const btn = $('#export-sum');
  btn.disabled = true;
  const scale = Math.min(1, 1280 / a.vw);
  const c = el('canvas', { width: Math.round(a.vw * scale), height: Math.round(a.vh * scale) });
  const g = c.getContext('2d');
  const stream = c.captureStream(30);
  try {
    const vs = video.captureStream ? video.captureStream() : video.mozCaptureStream?.();
    vs?.getAudioTracks().forEach((tr) => stream.addTrack(tr));
  } catch { /* audio is optional */ }
  const mime = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm'].find((m) => MediaRecorder.isTypeSupported(m)) || '';
  const rec = new MediaRecorder(stream, mime ? { mimeType: mime, videoBitsPerSecond: 4e6 } : undefined);
  const chunks = [];
  rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
  const stopped = new Promise((r) => (rec.onstop = r));
  rec.start(250);
  try {
    for (let i = 0; i < a.segments.length; i++) {
      const s = a.segments[i];
      btn.textContent = `Recording clip ${i + 1}/${a.segments.length}…`;
      await seekTo(s.start);
      await video.play();
      while (video.currentTime < s.end && !video.paused) {
        g.drawImage(video, 0, 0, c.width, c.height);
        g.fillStyle = 'rgba(0,0,0,0.55)';
        g.fillRect(12, 12, 230 * scale + 60, 30);
        g.fillStyle = '#fff';
        g.font = `600 ${Math.round(16 * Math.max(0.8, scale))}px Inter, sans-serif`;
        g.fillText(`Scene ${s.scene} · ${fmtTime(s.start)}`, 22, 33);
        await new Promise((r) => setTimeout(r, 1000 / 30));
      }
      video.pause();
    }
  } finally {
    rec.stop();
    await stopped;
    btn.disabled = false;
    btn.textContent = '⬇ Export summary (.webm)';
  }
  download(`${state.name.replace(/\.[^.]+$/, '')}-summary.webm`, new Blob(chunks, { type: 'video/webm' }));
}

function exportJson() {
  const a = state.a;
  const out = {
    video: state.name, duration: a.dur, framesAnalysed: a.n, sampleRate: a.rate,
    scenes: a.scenes.map((s) => ({ id: s.id, start: s.t0, end: s.t1, motion: s.motion, importance: s.importance })),
    boundaries: a.boundaries.map((b) => ({ time: b.time, type: b.type, strength: b.strength })),
    keyframes: a.keyframes.map((k) => ({ time: k.t, scene: k.scene, importance: k.importance })),
    tracks: a.tracks.map((t) => ({ id: t.displayId, label: t.label, scene: t.scene, start: t.t0, end: t.t1, pathPx: t.path * a.vw, direction: V.describeDirection(t.dx, t.dy), points: t.points.map((p) => ({ t: p.t, box: [p.box.x, p.box.y, p.box.w, p.box.h].map((v) => +v.toFixed(4)) })) })),
    summary: a.segments,
  };
  download(`${state.name.replace(/\.[^.]+$/, '')}-analysis.json`, JSON.stringify(out, null, 1), 'application/json');
}

// ------------------------------------------------------------ init ---
function bindTabs() {
  const buttons = $$('#tabs button');
  buttons.forEach((b) => b.addEventListener('click', () => {
    buttons.forEach((x) => x.setAttribute('aria-selected', String(x === b)));
    state.tab = b.dataset.tab;
    $$('[data-tabpane]').forEach((p) => p.classList.toggle('hidden', p.dataset.tabpane !== state.tab));
    renderTab();
  }));
  const sortBtns = $$('#kf-sort button');
  sortBtns.forEach((b) => b.addEventListener('click', () => { setPressed(sortBtns, b); state.kfSort = b.dataset.sort; renderFrames(); }));
}

bindSource();
bindTabs();
$('#run').addEventListener('click', analyse);
$('#play-sum').addEventListener('click', startSummary);
$('#export-sum').addEventListener('click', exportSummary);
$('#dl-json').addEventListener('click', exportJson);
$('#track-scene').addEventListener('change', () => state.a && renderTracks());
let resizeTimer;
window.addEventListener('resize', () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(() => { renderTab(); state.a && drawOverlay(); }, 150); });
onThemeChange(renderTab);
