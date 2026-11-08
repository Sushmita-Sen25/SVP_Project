import { $, $$, el, toast, setPressed, onThemeChange, cssVar, yieldUI } from './ui.js';
import { fetchAudio, decodeFile, MicRecorder, playSignal } from './audio.js';
import { recognitionFeatures, powerSpectrogram, trim } from './dsp.js';
import { LeftRightHMM, trainHMM, recognize } from './hmm.js';
import { drawSpectrogram, drawLines, drawConfusion } from './plot.js';

const vid = (id) => 'v' + id.replace('.', '_');
const state = {
  meta: null, models: {}, history: {}, mode: 'clip', input: null, truth: null, result: null, evalData: null,
};

const shortVoice = (v) => v.replace(/Neural$/, '').replace(/^(..)-(..)-/, '$1-$2 ');

// ------------------------------------------------------------ models ---
async function loadPretrained() {
  const j = await (await fetch('data/gita_hmm.json')).json();
  state.models = Object.fromEntries(Object.entries(j.models).map(([k, m]) => [k, LeftRightHMM.fromJSON(m)]));
  state.history = j.history;
  const Ns = Object.values(state.models).map((m) => m.N);
  $('#model-info').textContent = `Pretrained: ${Ns.length} verse HMMs, ${Math.min(...Ns)}–${Math.max(...Ns)} states each, ${j.features.dim}-dim features, trained on ${j.training.utterances} recitations.`;
  drawTraining();
}

async function retrain() {
  const btn = $('#retrain');
  btn.disabled = true;
  const prog = $('#train-prog');
  prog.classList.remove('hidden');
  const bar = prog.firstElementChild;
  const status = $('#train-status');
  try {
    const { verses, train_voices: voices, train_configs: cfgs } = state.meta;
    const jobs = verses.flatMap((v) => voices.flatMap((voice) => cfgs.map((c) => ({ verse: v.id, url: `audio/gita/train/${vid(v.id)}/${voice}__${c.id}.mp3` }))));
    const feats = {};
    let done = 0;
    status.textContent = `Downloading and extracting features from ${jobs.length} recitations…`;
    const t0 = performance.now();
    // decode with limited concurrency
    const queue = [...jobs];
    await Promise.all(Array.from({ length: 6 }, async () => {
      while (queue.length) {
        const j = queue.shift();
        const sig = await fetchAudio(j.url);
        (feats[j.verse] ||= []).push(recognitionFeatures(sig));
        done++;
        bar.style.width = `${(done / jobs.length) * 50}%`;
        if (done % 6 === 0) await yieldUI();
      }
    }));
    const models = {}, history = {};
    for (let i = 0; i < verses.length; i++) {
      const v = verses[i];
      status.textContent = `Training HMM for verse ${v.id} (${feats[v.id].length} recitations)…`;
      await yieldUI();
      const r = trainHMM(feats[v.id]);
      models[v.id] = r.model;
      history[v.id] = r.history;
      bar.style.width = `${50 + ((i + 1) / verses.length) * 50}%`;
    }
    state.models = models;
    state.history = history;
    const secs = ((performance.now() - t0) / 1000).toFixed(1);
    status.textContent = `Done in ${secs} s. The freshly trained models are now in use.`;
    $('#model-info').textContent = `Browser-trained: ${verses.length} HMMs from ${jobs.length} recitations (${secs} s).`;
    drawTraining();
    if (state.input) recognizeNow();
  } catch (e) {
    status.textContent = `Training failed: ${e.message}`;
  } finally {
    btn.disabled = false;
  }
}

// ------------------------------------------------------------ input ---
function fillSelectors() {
  const vs = $('#clip-verse');
  for (const v of state.meta.verses) vs.append(el('option', { value: v.id, text: `${v.id} · ${v.iast[0].split(' ').slice(0, 3).join(' ')}…` }));
  vs.value = '9.26';
  const vo = $('#clip-voice');
  for (const t of state.meta.test_sets) vo.append(el('option', { value: t.voice, text: `${shortVoice(t.voice)} · ${t.note}` }));
}

async function loadClip() {
  const verse = $('#clip-verse').value;
  const voice = $('#clip-voice').value;
  const url = `audio/gita/test/${vid(verse)}/${voice}.mp3`;
  $('#in-info').textContent = 'Loading clip…';
  const sig = await fetchAudio(url);
  setInput(sig, verse, `Test clip: verse ${verse}, ${shortVoice(voice)}`);
}

function setInput(sig, truth, label) {
  state.input = sig;
  state.truth = truth;
  $('#play').disabled = false;
  $('#in-info').textContent = `${label} · ${(sig.length / 16000).toFixed(1)} s`;
}

function bindInput() {
  const buttons = $$('#src-seg button');
  buttons.forEach((b) => b.addEventListener('click', () => {
    setPressed(buttons, b);
    state.mode = b.dataset.mode;
    $$('[data-pane]').forEach((p) => p.classList.toggle('hidden', p.dataset.pane !== state.mode));
    if (state.mode === 'clip') loadClip().then(recognizeNow);
  }));
  $('#clip-verse').addEventListener('change', () => loadClip().then(recognizeNow));
  $('#clip-voice').addEventListener('change', () => loadClip().then(recognizeNow));
  $('#partial').addEventListener('change', () => state.input && recognizeNow());
  $('#go').addEventListener('click', async () => {
    if (state.mode === 'clip' && !state.input) await loadClip();
    if (!state.input) return toast('Record or upload a recitation first.');
    recognizeNow();
  });
  $('#play').addEventListener('click', () => state.input && playSignal(state.input));

  const rec = new MicRecorder();
  const btn = $('#rec-btn');
  btn.addEventListener('click', async () => {
    if (!MicRecorder.supported()) return toast('Recording is not supported in this browser.');
    try {
      if (!rec.recording) {
        await rec.start();
        btn.classList.add('recording');
        btn.textContent = '■ Stop and recognise';
      } else {
        btn.classList.remove('recording');
        btn.textContent = '● Record recitation';
        const sig = await rec.stop();
        if (sig.length < 16000) return toast('Recording too short. Recite at least one line.');
        setInput(sig, null, 'Your recording');
        recognizeNow();
      }
    } catch (e) {
      btn.classList.remove('recording');
      btn.textContent = '● Record recitation';
      toast(`Microphone error: ${e.message}`);
    }
  });
  $('#file-in').addEventListener('change', async (e) => {
    const f = e.target.files[0];
    if (!f) return;
    try {
      setInput(await decodeFile(f), null, f.name);
      recognizeNow();
    } catch (err) { toast(`Could not decode ${f.name}: ${err.message}`); }
  });
}

// ------------------------------------------------------------ recognise ---
function recognizeNow() {
  const sig = state.input;
  const t0 = performance.now();
  const obs = recognitionFeatures(sig);
  const endAnyState = $('#partial').checked;
  const scores = recognize(state.models, obs, { endAnyState });
  const best = scores[0];
  const vit = state.models[best.id].viterbi(obs, { endAnyState });
  state.result = { obs, scores, vit, ms: performance.now() - t0, trimmed: trim(sig) };
  renderResult();
}

function renderResult() {
  const { scores, ms } = state.result;
  const best = scores[0];
  const verse = state.meta.verses.find((v) => v.id === best.id);
  $('#pred').textContent = `Verse ${best.id}`;
  const truth = $('#truth');
  truth.innerHTML = '';
  if (state.truth) {
    const ok = state.truth === best.id;
    truth.append(el('span', { class: `tag ${ok ? 'good' : 'bad'}`, text: ok ? 'correct' : `wrong · true ${state.truth}` }));
  }
  truth.append(el('div', { class: 'small muted', style: 'margin-top:6px', text: `margin ${(best.perFrame - scores[1].perFrame).toFixed(2)} nats/frame · ${state.result.obs.length} frames · ${ms.toFixed(0)} ms` }));

  const pv = $('#pred-verse');
  pv.innerHTML = '';
  pv.append(
    el('div', { class: 'verse-text' }, ...verse.deva.flatMap((l, i) => (i ? [el('br'), l] : [l]))),
    el('div', { class: 'verse-iast', html: verse.iast.join('<br>') }),
    el('p', { class: 'small', style: 'margin:8px 0 0', text: verse.meaning }),
  );

  const rk = $('#ranking');
  rk.innerHTML = '';
  const worst = scores[scores.length - 1].perFrame;
  scores.forEach((s, i) => {
    const w = ((s.perFrame - worst) / (best.perFrame - worst || 1)) * 100;
    rk.append(el('div', { class: `rank-row ${i === 0 ? 'best' : ''}` },
      el('span', { class: 'id', text: s.id }),
      el('div', { class: 'bar-track' }, el('div', { class: 'bar-fill', style: `width:${Math.max(2, w).toFixed(1)}%;background:${i === 0 ? cssVar('--accent') : cssVar('--muted')}` })),
      el('span', { class: 'v', text: `${s.perFrame.toFixed(2)} · ${(s.posterior * 100).toFixed(0)}%` })));
  });
  drawAlignment();
}

function drawAlignment() {
  if (!state.result) return;
  const { vit, trimmed } = state.result;
  const spec = powerSpectrogram(trimmed);
  const canvas = $('#c-align');
  drawSpectrogram(canvas, spec, { height: 200 });
  const ctx = canvas.getContext('2d');
  const w = canvas.clientWidth, h = 200, axis = 34;
  const T = vit.path.length;
  const x = (t) => axis + (t / T) * (w - axis);
  ctx.save();
  let start = 0;
  for (let t = 1; t <= T; t++) {
    if (t === T || vit.path[t] !== vit.path[start]) {
      const s = vit.path[start];
      ctx.fillStyle = s % 2 ? 'rgba(255,255,255,0.10)' : 'rgba(0,0,0,0.0)';
      ctx.fillRect(x(start), 0, x(t) - x(start), h);
      ctx.strokeStyle = 'rgba(255,255,255,0.55)';
      ctx.beginPath(); ctx.moveTo(x(start) + 0.5, 0); ctx.lineTo(x(start) + 0.5, h); ctx.stroke();
      if ((x(t) - x(start)) > 14) {
        ctx.fillStyle = '#fff';
        ctx.font = '10px JetBrains Mono, monospace';
        ctx.fillText(String(s), x(start) + 3, 12);
      }
      start = t;
    }
  }
  ctx.restore();
  const secs = Array.from(vit.path, (_, i) => i * 0.01);
  drawLines($('#c-states'), [{ data: Array.from(vit.path), x: secs, color: cssVar('--accent'), width: 2 }], {
    height: 140, yMin: 0, yMax: state.models[state.result.scores[0].id].N - 1, xFmt: (v) => v.toFixed(1), yFmt: (v) => v.toFixed(0),
  });
}

function drawTraining() {
  const ids = Object.keys(state.history);
  if (!ids.length) return;
  const palette = ['--accent', '--accent-2', '--good', '--warn', '--bad'];
  const series = ids.map((id, i) => ({
    data: state.history[id].map((h) => h.avgLogLik),
    x: state.history[id].map((h) => h.iter),
    color: cssVar(palette[i % palette.length]),
    width: i < 5 ? 1.8 : 1.1,
  }));
  const nIter = state.history[ids[0]].length;
  drawLines($('#c-train'), series, {
    height: 190, xMin: 1, xMax: nIter, xFmt: (v) => v.toFixed(0), yFmt: (v) => v.toFixed(1),
    vlines: [{ x: 4.5, color: cssVar('--muted'), dash: [4, 4] }],
  });
}

// ------------------------------------------------------------ evaluation ---
async function renderEval() {
  const e = await (await fetch('data/gita_eval.json')).json();
  state.evalData = e;
  const ho = e.heldOut;
  const unseen = ho.results.filter((r) => r.voice.startsWith('ne-'));
  const stat = (k, v, d) => el('div', { class: 'stat' }, el('div', { class: 'k', text: k }), el('div', { class: 'v', text: v }), el('div', { class: 'd', text: d }));
  const pct = (x) => `${(x * 100).toFixed(1)}%`;
  $('#eval-stats').append(
    stat('Held-out accuracy', pct(ho.accuracy), `${ho.results.length} test recitations`),
    stat('Unseen speakers', pct(unseen.filter((r) => r.correct).length / unseen.length), `${unseen.length} clips, 2 new voices`),
    stat('Leave-one-speaker-out', pct(e.loso.accuracy), `${e.loso.folds.length} folds`),
    stat('At 20 dB SNR', pct(ho.noise.find((n) => n.snr === 20).accuracy), 'white noise added'),
  );
  drawEvalPlots();
  const t = $('#partial-table');
  t.append(el('tr', {}, el('th', { text: 'Portion recited' }), el('th', { class: 'num', text: 'Must end in last state' }), el('th', { class: 'num', text: 'May end anywhere' })));
  for (const p of ho.partial) t.append(el('tr', {}, el('td', { text: `first ${p.fraction * 100}%` }), el('td', { class: 'num', text: pct(p.strictEnd) }), el('td', { class: 'num', text: pct(p.anyEnd) })));
}

function drawEvalPlots() {
  const e = state.evalData;
  if (!e) return;
  drawConfusion($('#c-conf'), e.loso.confusion, e.loso.labels);
  const n = e.heldOut.noise;
  const xs = n.map((p) => p.snr);
  drawLines($('#c-noise'), [{ data: n.map((p) => p.accuracy * 100), x: xs, color: cssVar('--accent'), width: 2.4 }, { data: xs.map(() => 10), x: xs, color: cssVar('--muted'), width: 1 }], {
    height: 170, yMin: 0, yMax: 100, xMin: 0, xMax: 30, xFmt: (v) => `${v.toFixed(0)} dB`, yFmt: (v) => v.toFixed(0),
  });
}

function renderLibrary() {
  const box = $('#library');
  for (const v of state.meta.verses) {
    const card = el('div', { class: 'card flat', style: 'padding:14px 16px;cursor:pointer' },
      el('div', { class: 'row-between' }, el('span', { class: 'tag accent', text: `Verse ${v.id}` }), el('span', { class: 'small muted', text: 'load test clip →' })),
      el('div', { class: 'verse-text', style: 'font-size:1.12rem;margin-top:6px' }, v.deva[0], el('br'), v.deva[1]),
      el('div', { class: 'verse-iast small' }, v.iast.join(' / ')),
    );
    card.addEventListener('click', () => {
      setPressed($$('#src-seg button'), $('#src-seg [data-mode="clip"]'));
      state.mode = 'clip';
      $$('[data-pane]').forEach((p) => p.classList.toggle('hidden', p.dataset.pane !== 'clip'));
      $('#clip-verse').value = v.id;
      loadClip().then(recognizeNow);
      window.scrollTo({ top: 0, behavior: 'smooth' });
    });
    box.append(card);
  }
}

let resizeTimer;
const redraw = () => { drawAlignment(); drawTraining(); drawEvalPlots(); if (state.result) renderResult(); };
window.addEventListener('resize', () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(redraw, 150); });
onThemeChange(redraw);

(async function init() {
  try {
    state.meta = await (await fetch('data/verses.json')).json();
    fillSelectors();
    bindInput();
    renderLibrary();
    $('#retrain').addEventListener('click', retrain);
    await loadPretrained();
    renderEval();
    await loadClip();
    recognizeNow();
  } catch (e) {
    console.error(e);
    $('#in-info').textContent = `Failed to initialise: ${e.message}`;
  }
})();
