// Decode any audio file to mono 16 kHz Float32 PCM with ffmpeg.
// Set FFMPEG=/path/to/ffmpeg if it is not on PATH
// (pip install imageio-ffmpeg ships a static binary).
import { execFileSync } from 'node:child_process';
import { readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';

export function decodeFile(path, sr = 16000) {
  const ff = process.env.FFMPEG || 'ffmpeg';
  const buf = execFileSync(ff, ['-v', 'error', '-i', path, '-ac', '1', '-ar', String(sr), '-f', 'f32le', '-'], {
    maxBuffer: 1 << 28,
  });
  return new Float32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
}

export function listFiles(dir, ext = '.mp3') {
  try {
    return readdirSync(dir).filter((f) => f.endsWith(ext) && statSync(join(dir, f)).isFile()).sort();
  } catch {
    return [];
  }
}
