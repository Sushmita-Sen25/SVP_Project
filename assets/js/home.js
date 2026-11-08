import { $, el, onThemeChange } from './ui.js';
import { fetchAudio } from './audio.js';
import { mfcc, trim, normalizePeak, cmvn } from './dsp.js';
import { drawWaveform, drawSpectrogram, drawMatrix } from './plot.js';

let hero = null;

async function drawHero() {
  try {
    if (!hero) {
      const sig = trim(normalizePeak(await fetchAudio('audio/words/krishna/hi-IN-SwaraNeural__ref.mp3')));
      const m = mfcc(sig);
      hero = { sig, spec: m.spec, mfcc: cmvn(m.mfcc, { variance: false }) };
    }
    drawWaveform($('#hero-wave'), hero.sig, { height: 70 });
    const box = $('#hero-spec');
    drawSpectrogram(box, hero.spec, { height: Math.max(130, box.parentElement.clientHeight - 230) });
    drawMatrix($('#hero-mfcc'), hero.mfcc, { height: 70, rowLabel: 'c' });
  } catch {
    $('.hero-art').classList.add('hidden');
  }
}

async function glance() {
  const box = $('#glance');
  const stat = (k, v, d, href) => el('a', { class: 'stat', href, style: 'text-decoration:none;color:inherit' },
    el('div', { class: 'k', text: k }), el('div', { class: 'v', text: v }), el('div', { class: 'd', text: d }));
  const pct = (x) => `${(x * 100).toFixed(1)}%`;
  try {
    const [g, p] = await Promise.all([fetch('data/gita_eval.json').then((r) => r.json()), fetch('data/pronunciation_eval.json').then((r) => r.json())]);
    const unseen = g.heldOut.results.filter((r) => r.voice.startsWith('ne-'));
    box.append(
      stat('Verse recognition', pct(g.heldOut.accuracy), `${g.heldOut.results.length} held-out recitations, 10 verses`, 'verse.html'),
      stat('Unseen speakers', pct(unseen.filter((r) => r.correct).length / unseen.length), 'two voices never used in training', 'verse.html'),
      stat('With 20 dB noise', pct(g.heldOut.noise.find((n) => n.snr === 20).accuracy), 'clean-trained HMMs, white noise', 'verse.html'),
      stat('Error identification', pct(p.identification.accuracy), 'pronunciation form, 4-way (chance 25%)', 'pronunciation.html'),
    );
  } catch {
    box.append(el('p', { class: 'muted', text: 'Evaluation data could not be loaded.' }));
  }
}

let t;
window.addEventListener('resize', () => { clearTimeout(t); t = setTimeout(drawHero, 150); });
onThemeChange(drawHero);
drawHero();
glance();
