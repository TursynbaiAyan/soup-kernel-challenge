// Renders comp/index.html frame by frame with headless Chrome.
//   node render.mjs stills 3,20,60   -> build/stills/*.jpg
//   node render.mjs video            -> build/video-noaudio.mp4
// Needs build/timeline.json (node timeline.mjs build). Uses puppeteer-core and ffmpeg-static from
// PUPPETEER_FROM (a folder whose node_modules has both) and the installed Chrome.
import { createRequire } from 'node:module';
import { spawn } from 'node:child_process';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { pathToFileURL, fileURLToPath } from 'node:url';
import path from 'node:path';

const here = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(path.join(process.env.PUPPETEER_FROM || here, 'package.json'));
const puppeteer = require('puppeteer-core');
const ffmpeg = require('ffmpeg-static');
const FPS = 30;
const [mode, arg] = process.argv.slice(2);
const timeline = JSON.parse(readFileSync(path.join(here, 'build', 'timeline.json'), 'utf8'));

const browser = await puppeteer.launch({
  executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe',
  headless: true,
  args: ['--allow-file-access-from-files', '--force-color-profile=srgb', '--hide-scrollbars'],
});
const page = await browser.newPage();
page.on('pageerror', (e) => console.log('PAGE ERROR', e.message));
await page.setViewport({ width: 1920, height: 1080, deviceScaleFactor: 1 });
await page.evaluateOnNewDocument((tl) => { window.TIMELINE = tl; }, timeline);
await page.goto(pathToFileURL(path.join(here, 'comp', 'index.html')).href, { waitUntil: 'load' });
await page.evaluate(() => window.ready);

const shot = async (t) => {
  await page.evaluate((t) => window.render(t), t);
  const b64 = await page.evaluate(() => document.getElementById('c').toDataURL('image/jpeg', 0.93).split(',')[1]);
  return Buffer.from(b64, 'base64');
};
if (mode === 'stills') {
  const dir = path.join(here, 'build', 'stills');
  mkdirSync(dir, { recursive: true });
  for (const t of arg.split(',').map(Number)) {
    writeFileSync(path.join(dir, `t${t.toFixed(1).padStart(5, '0')}.jpg`), await shot(t));
  }
  console.log('stills done', dir);
} else {
  const N = Math.round(timeline.total * FPS);
  const outFile = path.join(here, 'build', 'video-noaudio.mp4');
  const ff = spawn(ffmpeg, ['-y', '-f', 'image2pipe', '-framerate', String(FPS), '-c:v', 'mjpeg', '-i', '-',
    '-c:v', 'libx264', '-preset', 'medium', '-crf', '19', '-pix_fmt', 'yuv420p', '-r', String(FPS), outFile],
  { stdio: ['pipe', 'ignore', 'inherit'] });
  const t0 = Date.now();
  for (let i = 0; i < N; i++) {
    const buf = await shot(i / FPS);
    if (!ff.stdin.write(buf)) await new Promise((r) => ff.stdin.once('drain', r));
    if (i % 300 === 0) console.log(`frame ${i}/${N} ${((Date.now() - t0) / 1000).toFixed(0)}s`);
  }
  ff.stdin.end();
  await new Promise((r) => ff.on('close', r));
  console.log('video done', outFile);
}
await browser.close();
