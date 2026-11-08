// Train one left-to-right HMM per Bhagavad Gita verse on MFCC+Δ+ΔΔ features,
// evaluate on held-out recitations and run leave-one-speaker-out
// cross-validation. Uses exactly the same DSP/HMM code as the website.
//
//   FFMPEG=/path/to/ffmpeg node scripts/train_gita_hmm.mjs
//
// Writes data/gita_hmm.json (models) and data/gita_eval.json (results).
import { readFileSync, writeFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { recognitionFeatures } from '../assets/js/dsp.js';
import { trainHMM, recognize, LeftRightHMM } from '../assets/js/hmm.js';
import { decodeFile, listFiles } from './lib/node-audio.mjs';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const meta = JSON.parse(readFileSync(join(ROOT, 'data/verses.json'), 'utf8'));
const vid = (id) => 'v' + id.replace('.', '_');
const TRAIN_OPTS = { viterbiIters: 4, emIters: 6 };

function load(kind) {
  const items = [];
  for (const v of meta.verses) {
    const dir = join(ROOT, 'audio/gita', kind, vid(v.id));
    for (const f of listFiles(dir)) {
      const voice = f.split('__')[0].replace('.mp3', '');
      const audio = decodeFile(join(dir, f));
      items.push({ verse: v.id, voice, file: `audio/gita/${kind}/${vid(v.id)}/${f}`, audio, feats: recognitionFeatures(audio), seconds: audio.length / 16000 });
    }
  }
  return items;
}

function trainAll(items, log = false) {
  const models = {}, history = {};
  for (const v of meta.verses) {
    const seqs = items.filter((x) => x.verse === v.id).map((x) => x.feats);
    if (!seqs.length) continue;
    const t0 = Date.now();
    const { model, history: h } = trainHMM(seqs, TRAIN_OPTS);
    models[v.id] = model;
    history[v.id] = h;
    if (log) console.log(`  ${v.id.padEnd(6)} ${String(model.N).padStart(2)} states, ${seqs.length} utts, LL/frame ${h.at(-1).avgLogLik.toFixed(2)}  (${Date.now() - t0} ms)`);
  }
  return { models, history };
}

function evaluate(models, items, opts = {}) {
  const ids = meta.verses.map((v) => v.id);
  const confusion = ids.map(() => ids.map(() => 0));
  const results = items.map((it) => {
    const scores = recognize(models, it.feats, opts);
    const pred = scores[0].id;
    confusion[ids.indexOf(it.verse)][ids.indexOf(pred)]++;
    return { file: it.file, voice: it.voice, truth: it.verse, pred, correct: pred === it.verse, margin: scores[0].perFrame - scores[1].perFrame };
  });
  const acc = results.filter((r) => r.correct).length / results.length;
  return { accuracy: acc, confusion, labels: ids, results };
}

console.log('Extracting features…');
const train = load('train');
const test = load('test');
console.log(`  ${train.length} training, ${test.length} test recitations`);

console.log('Training verse HMMs…');
const { models, history } = trainAll(train, true);
const heldOut = evaluate(models, test);
console.log(`Held-out accuracy: ${(heldOut.accuracy * 100).toFixed(1)}% (${test.length} clips)`);

// robustness: additive white noise at several SNRs (deterministic PRNG)
function addNoise(x, snrDb, seed = 1) {
  let s = seed >>> 0;
  const rnd = () => ((s = (s * 1664525 + 1013904223) >>> 0) + 0.5) / 4294967296;
  let p = 0;
  for (const v of x) p += v * v;
  const sd = Math.sqrt(p / x.length / 10 ** (snrDb / 10));
  return Float32Array.from(x, (v) => v + sd * Math.sqrt(-2 * Math.log(rnd())) * Math.cos(2 * Math.PI * rnd()));
}
const noise = [];
for (const snr of [30, 20, 10, 5, 0]) {
  const noisy = test.map((it, i) => ({ ...it, feats: recognitionFeatures(addNoise(it.audio, snr, i + 1)) }));
  const acc = evaluate(models, noisy).accuracy;
  noise.push({ snr, accuracy: acc });
  console.log(`  white noise ${String(snr).padStart(2)} dB SNR: ${(acc * 100).toFixed(1)}%`);
}
// partial recitation: only the first X % of the verse, model may end in any state
const partial = [];
for (const frac of [0.3, 0.5, 0.7]) {
  const cut = test.map((it) => ({ ...it, feats: it.feats.slice(0, Math.round(it.feats.length * frac)) }));
  const strict = evaluate(models, cut).accuracy;
  const any = evaluate(models, cut, { endAnyState: true }).accuracy;
  partial.push({ fraction: frac, strictEnd: strict, anyEnd: any });
  console.log(`  first ${frac * 100}% only: strict end ${(strict * 100).toFixed(1)}%, any end state ${(any * 100).toFixed(1)}%`);
}

// leave-one-speaker-out over every voice in train ∪ test
const all = [...train, ...test];
const voices = [...new Set(all.map((x) => x.voice))];
const lovo = [];
const ids = meta.verses.map((v) => v.id);
const lovoConf = ids.map(() => ids.map(() => 0));
for (const voice of voices) {
  const { models: m } = trainAll(all.filter((x) => x.voice !== voice));
  const r = evaluate(m, all.filter((x) => x.voice === voice));
  r.confusion.forEach((row, i) => row.forEach((c, j) => (lovoConf[i][j] += c)));
  lovo.push({ voice, accuracy: r.accuracy, n: r.results.length });
  console.log(`  LOSO ${voice.padEnd(22)} ${(r.accuracy * 100).toFixed(1)}%`);
}
const lovoAcc = lovo.reduce((a, x) => a + x.accuracy * x.n, 0) / lovo.reduce((a, x) => a + x.n, 0);
console.log(`Leave-one-speaker-out accuracy: ${(lovoAcc * 100).toFixed(1)}%`);

const out = {
  features: { type: 'MFCC13+Δ+ΔΔ, CMVN', dim: 39, sr: 16000, frameMs: 25, hopMs: 10, nMels: 26 },
  training: { ...TRAIN_OPTS, topology: 'left-to-right, self/next/skip', emission: 'diagonal Gaussian', utterances: train.length },
  models: Object.fromEntries(Object.entries(models).map(([k, m]) => [k, m.toJSON()])),
  history,
};
writeFileSync(join(ROOT, 'data/gita_hmm.json'), JSON.stringify(out));
writeFileSync(
  join(ROOT, 'data/gita_eval.json'),
  JSON.stringify({ heldOut: { ...heldOut, noise, partial }, loso: { accuracy: lovoAcc, folds: lovo, confusion: lovoConf, labels: ids } }, null, 1),
);
console.log('Wrote data/gita_hmm.json and data/gita_eval.json');
// sanity: round-trip the JSON models
const back = LeftRightHMM.fromJSON(out.models[ids[0]]);
if (!Number.isFinite(back.forward(test[0].feats).logLik)) throw new Error('model round-trip failed');
