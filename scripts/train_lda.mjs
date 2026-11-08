// Learn a speaker-robust LDA projection of MFCC+Δ+ΔΔ frames.
// Classes are HMM states of the verse models (pseudo-phonetic units): every
// state contains frames from several speakers, so LDA keeps the directions
// that separate sounds and suppresses the directions that separate speakers.
//   FFMPEG=/path/to/ffmpeg node scripts/train_lda.mjs
import { readFileSync, writeFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { recognitionFeatures } from '../assets/js/dsp.js';
import { LeftRightHMM } from '../assets/js/hmm.js';
import { lda } from '../assets/js/linalg.js';
import { decodeFile, listFiles } from './lib/node-audio.mjs';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const meta = JSON.parse(readFileSync(join(ROOT, 'data/verses.json'), 'utf8'));
const hmm = JSON.parse(readFileSync(join(ROOT, 'data/gita_hmm.json'), 'utf8'));
const OUT_DIM = Number(process.env.LDA_DIM || 24);

const frames = [], labels = [];
meta.verses.forEach((v, vi) => {
  const model = LeftRightHMM.fromJSON(hmm.models[v.id]);
  const dir = join(ROOT, 'audio/gita/train', 'v' + v.id.replace('.', '_'));
  for (const f of listFiles(dir)) {
    const feats = recognitionFeatures(decodeFile(join(dir, f)));
    const { path } = model.viterbi(feats);
    feats.forEach((x, t) => { frames.push(x); labels.push(vi * 1000 + path[t]); });
  }
});
console.log(`${frames.length} frames, ${new Set(labels).size} classes`);
const t0 = Date.now();
const T = lda(frames, labels, OUT_DIM);
console.log(`LDA ${frames[0].length}→${OUT_DIM} in ${Date.now() - t0} ms; top eigenvalues ${T.values.slice(0, 6).map((x) => x.toFixed(2)).join(', ')}`);
const r = (x) => Number(x.toPrecision(6));
writeFileSync(join(ROOT, 'data/lda.json'), JSON.stringify({ inDim: frames[0].length, outDim: OUT_DIM, classes: new Set(labels).size, frames: frames.length, mean: T.mean.map(r), W: T.W.map((w) => w.map(r)), values: T.values.map(r) }));
console.log('Wrote data/lda.json');
