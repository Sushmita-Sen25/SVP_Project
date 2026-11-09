// Calibrates and evaluates the pronunciation comparison on the TTS corpus.
//  1. Pairwise DTW distance statistics (reference vs. student) used to set the
//     logistic score mapping in assets/js/pronounce.js.
//  2. Leave-one-speaker-out evaluation of multi-template error identification.
//   FFMPEG=/path/to/ffmpeg node scripts/calibrate_pronunciation.mjs
// Writes data/pronunciation_eval.json.
import { readFileSync, writeFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { analyzeUtterance, comparePronunciation, identifyForm } from '../assets/js/pronounce.js';
import { decodeFile } from './lib/node-audio.mjs';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const meta = JSON.parse(readFileSync(join(ROOT, 'data/words.json'), 'utf8'));
const LDA = JSON.parse(readFileSync(join(ROOT, 'data/lda.json'), 'utf8'));
const voices = meta.voices.map((v) => v.id);
const cache = new Map();
const get = (word, voice, form) => {
  const k = `${word}/${voice}__${form}`;
  if (!cache.has(k)) cache.set(k, analyzeUtterance(decodeFile(join(ROOT, 'audio/words', `${k}.mp3`)), { lda: LDA }));
  return cache.get(k);
};

const summary = (xs) => {
  const s = [...xs].sort((a, b) => a - b);
  const q = (p) => s[Math.min(s.length - 1, Math.floor(s.length * p))];
  return { n: s.length, mean: s.reduce((a, b) => a + b, 0) / s.length, p10: q(0.1), p50: q(0.5), p90: q(0.9) };
};
const fmt = (o) => `n=${o.n} mean=${o.mean.toFixed(2)} p10=${o.p10.toFixed(2)} p50=${o.p50.toFixed(2)} p90=${o.p90.toFixed(2)}`;

// ---- 1. pairwise statistics
const rows = [];
for (const w of meta.words) for (const rv of voices) {
  const ref = get(w.id, rv, 'ref');
  for (const tv of voices) for (const f of w.forms) {
    if (tv === rv && f.id === 'ref') continue;
    const r = comparePronunciation(ref, get(w.id, tv, f.id));
    rows.push({ same: tv === rv, correct: f.id === 'ref', d: r.distances.dtw, dv: r.distances.vowel, dc: r.distances.consonant, overall: r.overall });
  }
}
const pairwise = {
  otherSpeakerCorrect: rows.filter((r) => !r.same && r.correct),
  otherSpeakerWrong: rows.filter((r) => !r.same && !r.correct),
  sameSpeakerWrong: rows.filter((r) => r.same && !r.correct),
};
const pairwiseOut = {};
for (const [k, g] of Object.entries(pairwise)) {
  pairwiseOut[k] = { dtw: summary(g.map((r) => r.d)), vowel: summary(g.map((r) => r.dv)), consonant: summary(g.map((r) => r.dc)), score: summary(g.map((r) => r.overall)) };
  console.log(k.padEnd(20), 'dtw  ', fmt(pairwiseOut[k].dtw));
  console.log(''.padEnd(20), 'vowel', fmt(pairwiseOut[k].vowel));
  console.log(''.padEnd(20), 'cons ', fmt(pairwiseOut[k].consonant));
  console.log(''.padEnd(20), 'score', fmt(pairwiseOut[k].score));
}

// ---- 2. leave-one-speaker-out multi-template identification
const formIds = ['ref', 'v1', 'v2', 'v3'];
const confusion = formIds.map(() => formIds.map(() => 0));
let ok = 0, tot = 0;
const perWord = {};
for (const w of meta.words) {
  perWord[w.id] = { ok: 0, n: 0 };
  for (const tv of voices) for (const tf of w.forms) {
    const templates = Object.fromEntries(w.forms.map((f) => [f.id, voices.filter((v) => v !== tv).map((v) => get(w.id, v, f.id))]));
    const pred = identifyForm(get(w.id, tv, tf.id), templates)[0].form;
    confusion[formIds.indexOf(tf.id)][formIds.indexOf(pred)]++;
    tot++; perWord[w.id].n++;
    if (pred === tf.id) { ok++; perWord[w.id].ok++; }
  }
}
const accuracy = ok / tot;
// correct-vs-error detection (balanced accuracy)
const tp = confusion[0][0], fn = confusion[0].slice(1).reduce((a, b) => a + b, 0);
const errRows = confusion.slice(1);
const tn = errRows.reduce((a, r) => a + r.slice(1).reduce((x, y) => x + y, 0), 0);
const fp = errRows.reduce((a, r) => a + r[0], 0);
const balanced = 0.5 * (tp / (tp + fn) + tn / (tn + fp));
console.log(`LOSO form identification: ${(accuracy * 100).toFixed(1)}% (chance 25%), n=${tot}`);
console.log(`Correct-vs-error balanced accuracy: ${(balanced * 100).toFixed(1)}%`);

writeFileSync(join(ROOT, 'data/pronunciation_eval.json'), JSON.stringify({
  pairwise: pairwiseOut,
  identification: { accuracy, balancedCorrectVsError: balanced, n: tot, labels: formIds, confusion, perWord },
}, null, 1));
console.log('Wrote data/pronunciation_eval.json');
