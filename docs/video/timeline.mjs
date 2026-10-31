// Narration lines and timeline for the demo video.
//   node timeline.mjs lines   -> build/lines.json (scene texts, with Kaggle numbers if present)
//   node timeline.mjs build   -> build/timeline.json (scene start/end from the voice WAV lengths)
//                                and build/voice.wav (all lines placed at their scene starts)
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const build = path.join(here, 'build');
mkdirSync(build, { recursive: true });
const results = JSON.parse(readFileSync(path.join(here, '..', 'results.json'), 'utf8'));
const { scenes } = JSON.parse(readFileSync(path.join(here, 'narration.json'), 'utf8'));
const K = results.kaggle;
const sayNumber = (v) => {
  const [a, b] = v.toFixed(1).split('.');
  return b === '0' ? a : `${a} целых ${b} десятых`;
};
const active = scenes.filter((s) => !s.only_with_kaggle || K).map((s) => ({
  ...s,
  say: K ? s.say.replace('{speedup_say}', sayNumber(K.speedup)).replace('{memory_say}', sayNumber(K.memory)) : s.say,
}));

function readWav(file) {
  const buf = readFileSync(file);
  let off = 12, fmt, data;
  while (off < buf.length) {
    const id = buf.toString('ascii', off, off + 4), size = buf.readUInt32LE(off + 4);
    if (id === 'fmt ') fmt = { rate: buf.readUInt32LE(off + 12), channels: buf.readUInt16LE(off + 10) };
    if (id === 'data') data = buf.subarray(off + 8, off + 8 + size);
    off += 8 + size + (size % 2);
  }
  const samples = new Int16Array(data.buffer, data.byteOffset, data.length / 2);
  return { rate: fmt.rate, samples };
}

if (process.argv[2] === 'lines') {
  writeFileSync(path.join(build, 'lines.json'), JSON.stringify(active.map(({ id, say }) => ({ id, say })), null, 1));
  console.log('lines', active.length);
} else {
  const RATE = 48000, LEAD = 0.6;
  let t = 0.4;
  const timeline = [], clips = [];
  for (const s of active) {
    const wav = readWav(path.join(build, `voice-${s.id}.wav`));
    const dur = wav.samples.length / wav.rate;
    timeline.push({ id: s.id, start: +t.toFixed(3), voice: +(t + LEAD).toFixed(3), end: +(t + LEAD + dur + s.pad).toFixed(3) });
    clips.push({ at: t + LEAD, wav });
    t += LEAD + dur + s.pad;
  }
  const total = Math.ceil(t + 0.5);
  const out = new Int16Array(total * RATE);
  for (const { at, wav } of clips) {
    const o = Math.round(at * RATE);
    for (let i = 0; i < wav.samples.length && o + i < out.length; i++) out[o + i] = wav.samples[i];
  }
  const header = Buffer.alloc(44);
  header.write('RIFF', 0); header.writeUInt32LE(36 + out.length * 2, 4); header.write('WAVE', 8);
  header.write('fmt ', 12); header.writeUInt32LE(16, 16); header.writeUInt16LE(1, 20); header.writeUInt16LE(1, 22);
  header.writeUInt32LE(RATE, 24); header.writeUInt32LE(RATE * 2, 28); header.writeUInt16LE(2, 32); header.writeUInt16LE(16, 34);
  header.write('data', 36); header.writeUInt32LE(out.length * 2, 40);
  writeFileSync(path.join(build, 'voice.wav'), Buffer.concat([header, Buffer.from(out.buffer)]));
  writeFileSync(path.join(build, 'timeline.json'), JSON.stringify({ total, scenes: timeline, results }, null, 1));
  console.log('timeline', total, 's', timeline.map((s) => `${s.id}@${s.start}`).join(' '));
}
