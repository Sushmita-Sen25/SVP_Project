// Browser audio I/O: decode/resample to mono 16 kHz, microphone recording,
// playback of Float32 buffers and WAV export.
import { SR } from './dsp.js';

let ctx;
export function audioContext() {
  if (!ctx) ctx = new (window.AudioContext || window.webkitAudioContext)();
  return ctx;
}

/** Decode an encoded audio ArrayBuffer and resample to mono 16 kHz. */
export async function decodeToMono(arrayBuffer, sr = SR) {
  // An OfflineAudioContext can decode without a user gesture (unlike a
  // suspended AudioContext on some browsers).
  const dec = new OfflineAudioContext(1, 1, 44100);
  const buf = await dec.decodeAudioData(arrayBuffer.slice(0));
  const frames = Math.ceil(buf.duration * sr);
  const off = new OfflineAudioContext(1, Math.max(1, frames), sr);
  const src = off.createBufferSource();
  src.buffer = buf; // multi-channel input is down-mixed to the mono destination
  src.connect(off.destination);
  src.start();
  const out = await off.startRendering();
  return out.getChannelData(0).slice();
}

const fetchCache = new Map();
export function fetchAudio(url) {
  if (!fetchCache.has(url)) {
    fetchCache.set(url, fetch(url).then((r) => {
      if (!r.ok) throw new Error(`HTTP ${r.status} for ${url}`);
      return r.arrayBuffer();
    }).then((ab) => decodeToMono(ab)).catch((e) => { fetchCache.delete(url); throw e; }));
  }
  return fetchCache.get(url);
}

export async function decodeFile(file) {
  return decodeToMono(await file.arrayBuffer());
}

let current;
/** Play a mono Float32Array; returns a stop function. */
export function playSignal(signal, sr = SR, onEnd) {
  stopPlayback();
  const ac = audioContext();
  if (ac.state === 'suspended') ac.resume();
  const b = ac.createBuffer(1, signal.length, sr);
  b.copyToChannel(signal, 0);
  const src = ac.createBufferSource();
  src.buffer = b;
  src.connect(ac.destination);
  src.onended = () => { if (current === src) current = null; onEnd && onEnd(); };
  src.start();
  current = src;
  return () => { try { src.stop(); } catch { /* already stopped */ } };
}
export function stopPlayback() {
  if (current) { try { current.stop(); } catch { /* ignore */ } current = null; }
}

/** MediaRecorder-based microphone recorder. */
export class MicRecorder {
  constructor() { this.chunks = []; this.rec = null; this.stream = null; }
  static supported() { return !!(navigator.mediaDevices?.getUserMedia && window.MediaRecorder); }
  async start() {
    this.stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: true, channelCount: 1 } });
    this.chunks = [];
    this.rec = new MediaRecorder(this.stream);
    this.rec.ondataavailable = (e) => e.data.size && this.chunks.push(e.data);
    this.rec.start();
    this.t0 = performance.now();
  }
  async stop() {
    if (!this.rec) return null;
    const done = new Promise((r) => (this.rec.onstop = r));
    this.rec.stop();
    await done;
    this.stream.getTracks().forEach((t) => t.stop());
    const blob = new Blob(this.chunks, { type: this.rec.mimeType || 'audio/webm' });
    this.rec = null;
    return decodeToMono(await blob.arrayBuffer());
  }
  get recording() { return !!this.rec; }
}

export function encodeWav(signal, sr = SR) {
  const n = signal.length;
  const buf = new ArrayBuffer(44 + n * 2);
  const v = new DataView(buf);
  const w = (o, s) => [...s].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)));
  w(0, 'RIFF'); v.setUint32(4, 36 + n * 2, true); w(8, 'WAVE'); w(12, 'fmt ');
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, sr, true); v.setUint32(28, sr * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  w(36, 'data'); v.setUint32(40, n * 2, true);
  for (let i = 0; i < n; i++) v.setInt16(44 + i * 2, Math.max(-1, Math.min(1, signal[i])) * 0x7fff, true);
  return new Blob([buf], { type: 'audio/wav' });
}
