// Shared page helpers: theme toggle, toasts, tiny DOM utilities.

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

export function el(tag, attrs = {}, ...children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === null || v === false) continue;
    if (k === 'class') e.className = v;
    else if (k === 'text') e.textContent = v;
    else if (k === 'html') e.innerHTML = v;
    else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v === true ? '' : v);
  }
  for (const c of children.flat()) if (c !== null && c !== undefined) e.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return e;
}

export function fmtTime(sec) {
  if (!Number.isFinite(sec)) return '–';
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = Math.floor(sec % 60);
  const mm = String(m).padStart(h ? 2 : 1, '0');
  return h ? `${h}:${mm}:${String(s).padStart(2, '0')}` : `${mm}:${String(s).padStart(2, '0')}`;
}

let toastEl, toastTimer;
export function toast(msg, ms = 3200) {
  if (!toastEl) {
    toastEl = el('div', { class: 'toast', role: 'status', 'aria-live': 'polite' });
    document.body.append(toastEl);
  }
  toastEl.textContent = msg;
  toastEl.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toastEl.classList.remove('show'), ms);
}

export const nextFrame = () => new Promise((r) => requestAnimationFrame(() => r()));
// Yield to the event loop without depending on rendering (rAF stalls in hidden
// or non-painting windows; timers are throttled in background tabs).
export const yieldUI = () => new Promise((r) => { const ch = new MessageChannel(); ch.port1.onmessage = () => r(); ch.port2.postMessage(0); });
export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

export function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

const THEME_KEY = 'svp-theme';
function applyTheme(t) {
  if (t === 'light' || t === 'dark') document.documentElement.setAttribute('data-theme', t);
  else document.documentElement.removeAttribute('data-theme');
}
function currentTheme() {
  const t = document.documentElement.getAttribute('data-theme');
  if (t) return t;
  return matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

export const themeListeners = new Set();
export function onThemeChange(fn) { themeListeners.add(fn); }

function initTheme() {
  try { applyTheme(localStorage.getItem(THEME_KEY)); } catch { /* storage blocked */ }
  const btn = document.getElementById('theme-toggle');
  const paint = () => { if (btn) btn.innerHTML = currentTheme() === 'dark' ? sunIcon : moonIcon; };
  paint();
  btn?.addEventListener('click', () => {
    const next = currentTheme() === 'dark' ? 'light' : 'dark';
    applyTheme(next);
    try { localStorage.setItem(THEME_KEY, next); } catch { /* ignore */ }
    paint();
    themeListeners.forEach((f) => f());
  });
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { paint(); themeListeners.forEach((f) => f()); });
}

const moonIcon = '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>';
const sunIcon = '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>';

export function download(filename, data, type = 'application/octet-stream') {
  const blob = data instanceof Blob ? data : new Blob([data], { type });
  const url = URL.createObjectURL(blob);
  const a = el('a', { href: url, download: filename });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 4000);
}

export function setPressed(buttons, active) {
  buttons.forEach((b) => b.setAttribute('aria-pressed', String(b === active)));
}

export function scoreColor(v) {
  if (v >= 80) return cssVar('--good');
  if (v >= 60) return cssVar('--warn');
  return cssVar('--bad');
}

initTheme();
