import { $, $$, el, toast, setPressed, onThemeChange, scoreColor, cssVar } from './ui.js';
import { fetchAudio, decodeFile, MicRecorder, playSignal } from './audio.js';
import { analyzeUtterance, comparePronunciation, identifyForm } from './pronounce.js';
import { drawWaveform, drawSpectrogram, drawMatrix, drawDtw, drawLines, drawBars } from './plot.js';
import { cmvn } from './dsp.js';

const state = {
  meta: null, lda: null, word: null,
  ref: { mode: 'builtin', voice: null, signal: null, analysis: null, label: '' },
  stu: { mode: 'builtin', voice: null, form: 'v3', signal: null, analysis: null, label: '' },
  result: null, token: 0, tab: 'signals',
};
const analysisCache = new Map();
const wordUrl = (w, voice, form) => `audio/words/${w}/${voice}__${form}.mp3`;

async function analysisFor(url) {
  if (!analysisCache.has(url)) {
    analysisCache.set(url, fetchAudio(url).then((sig) => ({ signal: sig, analysis: analyzeUtterance(sig, { lda: state.lda }) })));
  }
  return analysisCache.get(url);
}

function setStatus(msg) { $('#status').textContent = msg || ''; }

// ------------------------------------------------------------ setup UI ---
function buildWordChips() {
  const box = $('#word-chips');
  box.innerHTML = '';
  for (const w of state.meta.words) {
    const b = el('button', { class: 'word-chip', 'aria-pressed': String(w.id === state.word.id), onclick: () => selectWord(w.id) },
      el('span', { class: 'deva', text: w.deva }), el('span', { class: 'iast', text: w.iast }));
    box.append(b);
  }
}

function fillVoices() {
  for (const id of ['#ref-voice', '#stu-voice']) {
    const s = $(id);
    s.innerHTML = '';
    for (const v of state.meta.voices) s.append(el('option', { value: v.id, text: v.label }));
  }
  $('#ref-voice').value = state.meta.teacher;
  $('#stu-voice').value = 'hi-IN-MadhurNeural';
  state.ref.voice = state.meta.teacher;
  state.stu.voice = 'hi-IN-MadhurNeural';
}

function fillForms() {
  const s = $('#stu-form');
  const prev = state.stu.form;
  s.innerHTML = '';
  for (const f of state.word.forms) {
    s.append(el('option', { value: f.id, text: f.id === 'ref' ? `${f.label} (correct)` : `${f.label} (${f.error})` }));
  }
  s.value = state.word.forms.some((f) => f.id === prev) ? prev : 'v3';
  state.stu.form = s.value;
}

function selectWord(id) {
  state.word = state.meta.words.find((w) => w.id === id);
  $$('#word-chips .word-chip').forEach((b, i) => b.setAttribute('aria-pressed', String(state.meta.words[i].id === id)));
  $('#word-deva').textContent = state.word.deva;
  $('#word-iast').textContent = state.word.iast + ' · ' + state.word.meaning;
  fillForms();
  if (state.ref.mode === 'builtin') state.ref.signal = null;
  if (state.stu.mode === 'builtin') state.stu.signal = null;
  run();
}

function bindSource(key) {
  const src = state[key];
  const seg = $(`.seg[data-src="${key}"]`);
  const buttons = $$('button', seg);
  buttons.forEach((b) => b.addEventListener('click', () => {
    setPressed(buttons, b);
    src.mode = b.dataset.mode;
    for (const m of ['builtin', 'record', 'upload']) $(`[data-pane="${key}-${m}"]`).classList.toggle('hidden', m !== src.mode);
    if (src.mode === 'builtin') { src.signal = null; run(); }
  }));

  $(`#${key}-voice`).addEventListener('change', (e) => { src.voice = e.target.value; src.signal = null; run(); });
  if (key === 'stu') $('#stu-form').addEventListener('change', (e) => { src.form = e.target.value; src.signal = null; run(); });

  const rec = new MicRecorder();
  const recBtn = $(`#${key}-rec`);
  const idle = recBtn.textContent;
  recBtn.addEventListener('click', async () => {
    if (!MicRecorder.supported()) return toast('Recording is not supported in this browser.');
    try {
      if (!rec.recording) {
        await rec.start();
        recBtn.classList.add('recording');
        recBtn.textContent = '■ Stop recording';
      } else {
        recBtn.classList.remove('recording');
        recBtn.textContent = idle;
        const sig = await rec.stop();
        if (!sig || sig.length < 3200) return toast('Recording too short. Try again.');
        src.signal = sig;
        src.analysis = analyzeUtterance(sig, { lda: state.lda });
        src.label = 'your recording';
        run();
      }
    } catch (e) {
      recBtn.classList.remove('recording');
      recBtn.textContent = idle;
      toast(`Microphone error: ${e.message}`);
    }
  });

  $(`#${key}-file`).addEventListener('change', async (e) => {
    const f = e.target.files[0];
    if (!f) return;
    try {
      const sig = await decodeFile(f);
      src.signal = sig;
      src.analysis = analyzeUtterance(sig, { lda: state.lda });
      src.label = f.name;
      run();
    } catch (err) {
      toast(`Could not decode ${f.name}: ${err.message}`);
    }
  });

  $(`#${key}-play`).addEventListener('click', () => src.analysis && playSignal(src.analysis.signal));
}

// ------------------------------------------------------------ pipeline ---
async function resolve(key) {
  const src = state[key];
  if (src.mode === 'builtin') {
    const form = key === 'ref' ? 'ref' : src.form;
    const { signal, analysis } = await analysisFor(wordUrl(state.word.id, src.voice, form));
    src.signal = signal;
    src.analysis = analysis;
    const v = state.meta.voices.find((x) => x.id === src.voice);
    const f = state.word.forms.find((x) => x.id === form);
    src.label = `${f.label} · ${v.label}`;
  }
  return src.analysis;
}

async function run() {
  const token = ++state.token;
  if ((state.ref.mode !== 'builtin' && !state.ref.analysis) || (state.stu.mode !== 'builtin' && !state.stu.analysis)) {
    setStatus(state.ref.mode !== 'builtin' && !state.ref.analysis ? 'Record or upload the reference.' : 'Record or upload the student audio.');
    if (state.stu.mode === 'builtin' || state.stu.analysis) updateInfo();
    return;
  }
  setStatus('Analysing…');
  try {
    const [ref, stu] = await Promise.all([resolve('ref'), resolve('stu')]);
    if (token !== state.token) return;
    updateInfo();
    const result = comparePronunciation(ref, stu);
    state.result = result;
    renderScores(result);
    renderTab();
    setStatus(`DTW distance ${result.distances.dtw.toFixed(2)} · ${result.dtw.path.length}-step warping path · ${ref.feat.length}×${stu.feat.length} frames`);
    runIdentification(token);
  } catch (e) {
    console.error(e);
    setStatus(`Error: ${e.message}`);
  }
}

function updateInfo() {
  for (const k of ['ref', 'stu']) {
    const a = state[k].analysis;
    $(`#${k}-info`).textContent = a ? `${state[k].label} · ${a.duration.toFixed(2)} s` : '–';
    $(`#${k}-play`).disabled = !a;
  }
  const s = state.stu;
  if (s.mode === 'builtin') {
    const f = state.word.forms.find((x) => x.id === s.form);
    $('#stu-said').innerHTML = '';
    $('#stu-said').append(el('span', { class: 'deva', style: 'font-size:1.25rem', text: f.deva }), ' ', el('span', { text: f.label }), ' ',
      el('span', { class: `tag ${f.id === 'ref' ? 'good' : 'bad'}`, text: f.id === 'ref' ? 'correct' : 'mispronounced' }));
  } else {
    $('#stu-said').textContent = s.label || '–';
  }
}

// ------------------------------------------------------------ render ---
function renderScores(r) {
  const v = Math.round(r.overall);
  $('#score-num').textContent = `${v}%`;
  const arc = $('#gauge-arc');
  arc.setAttribute('stroke-dasharray', `${(v / 100) * 377} 377`);
  arc.setAttribute('stroke', scoreColor(v));
  const rows = [
    ['MFCC (DTW)', r.scores.mfcc], ['Vowels', r.scores.vowel], ['Consonants', r.scores.consonant],
    ['Duration', r.scores.duration], ['Spectral', r.scores.spectral],
  ];
  const box = $('#subscores');
  box.innerHTML = '';
  for (const [k, s] of rows) {
    box.append(el('div', { class: 'bar-row' }, el('span', { text: k }),
      el('div', { class: 'bar-track' }, el('div', { class: 'bar-fill', style: `width:${s.toFixed(1)}%;background:${scoreColor(s)}` })),
      el('span', { class: 'v', text: `${Math.round(s)}` })));
  }
  const fb = $('#feedback');
  fb.innerHTML = '';
  for (const f of r.feedback) fb.append(el('li', { class: f.level, text: f.text }));
  if (!r.feedback.length) fb.append(el('li', { text: 'No specific issues detected.' }));
}

async function runIdentification(token) {
  const box = $('#ident');
  box.innerHTML = '<span class="small muted"><span class="spinner"></span> Comparing with reference speakers…</span>';
  const w = state.word;
  const exclude = state.stu.mode === 'builtin' ? state.stu.voice : null;
  const voices = state.meta.voices.map((v) => v.id).filter((v) => v !== exclude);
  try {
    const templates = {};
    await Promise.all(w.forms.map(async (f) => {
      templates[f.id] = (await Promise.all(voices.map((v) => analysisFor(wordUrl(w.id, v, f.id))))).map((x) => x.analysis);
    }));
    if (token !== state.token) return;
    const rows = identifyForm(state.stu.analysis, templates);
    box.innerHTML = '';
    rows.forEach((r, i) => {
      const f = w.forms.find((x) => x.id === r.form);
      box.append(el('div', { class: `rank-row ${i === 0 ? 'best' : ''}`, style: 'grid-template-columns: minmax(0,1.1fr) 1fr 52px' },
        el('div', {}, el('div', { class: 'id', text: f.label }), el('div', { class: 'small muted', text: f.error })),
        el('div', { class: 'bar-track' }, el('div', { class: 'bar-fill', style: `width:${(r.prob * 100).toFixed(1)}%;background:${f.id === 'ref' ? cssVar('--good') : cssVar('--accent')}` })),
        el('span', { class: 'v', text: `${Math.round(r.prob * 100)}%` })));
    });
    const top = w.forms.find((x) => x.id === rows[0].form);
    $('#ident-note').textContent = `${voices.length} reference speakers per form${exclude ? ' (the student’s own voice is left out)' : ''}. Best match: ${top.label}${top.id === 'ref' ? ', the correct form' : ` (${top.error})`}.`;
  } catch (e) {
    box.textContent = `Could not run identification: ${e.message}`;
  }
}

function renderTab() {
  const r = state.result;
  if (!r) return;
  const ref = state.ref.analysis, stu = state.stu.analysis;
  const top = r.regions.slice(0, 2);
  const stuRegions = top.map((g) => ({ t0: g.testT0, t1: g.testT1 }));
  const ink = cssVar('--accent'), ink2 = cssVar('--accent-2');
  switch (state.tab) {
    case 'signals':
      drawWaveform($('#c-ref-wave'), ref.signal, { labels: ref.labels, regions: top, color: ink });
      drawSpectrogram($('#c-ref-spec'), ref.spec, { labels: ref.labels, regions: top });
      drawWaveform($('#c-stu-wave'), stu.signal, { labels: stu.labels, regions: stuRegions, color: ink2 });
      drawSpectrogram($('#c-stu-spec'), stu.spec, { labels: stu.labels, regions: stuRegions });
      break;
    case 'mfcc': {
      const strip = (m) => cmvn(m, { variance: false }).map((v) => v.subarray(1));
      drawMatrix($('#c-ref-mfcc'), strip(ref.mfcc), { rowLabel: 'c' });
      drawMatrix($('#c-stu-mfcc'), strip(stu.mfcc), { rowLabel: 'c' });
      drawBars($('#c-coef'), r.coefDiff, { labels: r.coefDiff.map((_, k) => `c${k}`), yFmt: (v) => v.toFixed(1) });
      break;
    }
    case 'align': {
      const c = $('#c-dtw');
      drawDtw(c, r.dtw, { height: Math.min(340, Math.max(220, c.clientWidth * 0.8)) });
      const secs = Array.from(r.perRef, (_, i) => i * 0.01);
      drawLines($('#c-cost'), [{ data: r.perRef, x: secs, color: ink, fill: true }], {
        height: 150, yMin: 0, xFmt: (x) => x.toFixed(2), bands: top.map((g) => ({ x0: g.t0, x1: g.t1, color: 'rgba(220,38,38,0.15)' })),
      });
      drawLines($('#c-tempo'), [{ data: r.tempo, x: secs, color: ink2 }, { data: secs.map(() => 1), x: secs, color: cssVar('--plot-grid'), width: 1 }], {
        height: 120, yMin: 0, yMax: Math.max(2, ...r.tempo), xFmt: (x) => x.toFixed(2), yFmt: (y) => y.toFixed(1),
      });
      break;
    }
    case 'spectral': {
      const n = r.ltasRef.length;
      drawLines($('#c-ltas'), [{ data: r.ltasRef, color: ink, width: 2 }, { data: r.ltasTest, color: ink2, width: 2 }], {
        height: 160, xMax: n - 1, xFmt: (x) => `b${Math.round(x)}`, yFmt: (y) => `${y.toFixed(0)}`,
      });
      const secs = Array.from(ref.f0, (_, i) => i * 0.01).slice(0, r.map.length);
      const stuF0 = Array.from(r.map, (j) => stu.f0[Math.round(j)] || 0);
      drawLines($('#c-pitch'), [{ data: Array.from(ref.f0).slice(0, r.map.length), x: secs, color: ink, skipZero: true, width: 2 }, { data: stuF0, x: secs, color: ink2, skipZero: true, width: 2 }], {
        height: 150, yMin: 60, yMax: 400, xFmt: (x) => x.toFixed(2), yFmt: (y) => y.toFixed(0),
      });
      const cRef = ref.desc.map((d) => d.centroid).slice(0, r.map.length);
      const cStu = Array.from(r.map, (j) => stu.desc[Math.round(j)]?.centroid ?? 0);
      drawLines($('#c-centroid'), [{ data: cRef, x: secs, color: ink, width: 1.8 }, { data: cStu, x: secs, color: ink2, width: 1.8 }], {
        height: 150, yMin: 0, yMax: 6000, xFmt: (x) => x.toFixed(2), yFmt: (y) => `${(y / 1000).toFixed(1)}k`,
      });
      break;
    }
    case 'durations': renderDurations(ref, stu, r); break;
  }
}

function renderDurations(ref, stu, r) {
  const info = (a) => {
    const cnt = (l) => a.labels.filter((x) => x === l).length * 0.01;
    const nuclei = a.segments.filter((s) => s.label === 'V' && s.end - s.start >= 3).length;
    return { dur: a.duration, v: cnt('V'), c: cnt('C'), s: cnt('S'), nuclei, rate: nuclei / a.duration };
  };
  const A = info(ref), B = info(stu);
  const rows = [
    ['Speech duration', A.dur, B.dur, 's'], ['Vowel-like time', A.v, B.v, 's'], ['Consonant-like time', A.c, B.c, 's'],
    ['Silence inside word', A.s, B.s, 's'], ['Vowel share', r.vowelShareRef * 100, r.vowelShareTest * 100, '%'],
    ['Vowel segments (≈ syllables)', A.nuclei, B.nuclei, ''], ['Syllable rate', A.rate, B.rate, '/s'],
  ];
  const t = $('#dur-table');
  t.innerHTML = '';
  t.append(el('tr', {}, el('th', { text: 'Measure' }), el('th', { class: 'num', text: 'Reference' }), el('th', { class: 'num', text: 'Student' }), el('th', { class: 'num', text: 'Ratio' })));
  for (const [k, a, b, u] of rows) {
    const ratio = a ? b / a : NaN;
    const fmt = (x) => (u === '' ? String(Math.round(x)) : x.toFixed(u === '%' ? 0 : 2)) + (u && u !== 's' ? u : u === 's' ? ' s' : '');
    t.append(el('tr', {}, el('td', { text: k }), el('td', { class: 'num', text: fmt(a) }), el('td', { class: 'num', text: fmt(b) }),
      el('td', { class: 'num', text: Number.isFinite(ratio) ? `${ratio.toFixed(2)}×` : '–' })));
  }
}

function bindTabs() {
  const buttons = $$('#tabs button');
  buttons.forEach((b) => b.addEventListener('click', () => {
    buttons.forEach((x) => x.setAttribute('aria-selected', String(x === b)));
    state.tab = b.dataset.tab;
    $$('[data-tabpane]').forEach((p) => p.classList.toggle('hidden', p.dataset.tabpane !== state.tab));
    renderTab();
  }));
}

async function renderEval() {
  try {
    const e = await (await fetch('data/pronunciation_eval.json')).json();
    const p = e.pairwise, id = e.identification;
    const box = $('#eval-stats');
    const stat = (k, v, d) => el('div', { class: 'stat' }, el('div', { class: 'k', text: k }), el('div', { class: 'v', text: v }), el('div', { class: 'd', text: d }));
    box.append(
      stat('Error identification', `${(id.accuracy * 100).toFixed(1)}%`, `4-way, chance 25% · n=${id.n}`),
      stat('Correct, other speaker', `${p.otherSpeakerCorrect.score.p50.toFixed(0)}%`, 'median similarity score'),
      stat('Wrong, other speaker', `${p.otherSpeakerWrong.score.p50.toFixed(0)}%`, 'median similarity score'),
    );
  } catch { $('#eval-card').classList.add('hidden'); }
}

let resizeTimer;
window.addEventListener('resize', () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(renderTab, 150); });
onThemeChange(() => { renderTab(); if (state.result) renderScores(state.result); });

(async function init() {
  try {
    const [meta, lda] = await Promise.all([fetch('data/words.json').then((r) => r.json()), fetch('data/lda.json').then((r) => r.json())]);
    state.meta = meta;
    state.lda = lda;
    state.word = meta.words[0];
    fillVoices();
    buildWordChips();
    bindSource('ref');
    bindSource('stu');
    bindTabs();
    selectWord(meta.words[0].id);
    renderEval();
  } catch (e) {
    setStatus(`Failed to load data: ${e.message}`);
  }
})();
