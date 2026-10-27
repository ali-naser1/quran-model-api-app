"""Mushaf-style reader with live microphone streaming (Streamlit custom component v2).

JS → Python: `audio` trigger with 16 kHz int16 PCM chunks (base64), resent until acked.
Python → JS: `data` with word statuses, tooltips and the last acked chunk.
"""

from collections.abc import Callable

import streamlit as st

HTML = """
<div class="reader">
  <div class="toolbar">
    <button class="rec" type="button"></button>
    <button class="reset" type="button">Reset</button>
    <span class="status"></span>
  </div>
  <div class="legend">
    <span><i class="dot ok"></i>Correct</span>
    <span><i class="dot wrong"></i>Wrong / missed</span>
    <span><i class="dot tajweed"></i>Tajweed note</span>
    <span><i class="dot current"></i>Reading now</span>
  </div>
  <div class="mushaf" dir="rtl" lang="ar"></div>
</div>
"""

CSS = """
.reader { font-family: var(--st-font); color: var(--st-text-color); }
.toolbar { display: flex; gap: .5rem; align-items: center; flex-wrap: wrap; margin-bottom: .5rem; }
button {
  font: inherit; cursor: pointer; border-radius: var(--st-button-radius, .5rem);
  border: 1px solid var(--st-border-color, rgba(128,128,128,.4));
  background: var(--st-secondary-background-color); color: var(--st-text-color);
  padding: .4rem .9rem;
}
button.rec { background: var(--st-primary-color); color: #fff; border-color: transparent; min-width: 9.5rem; }
button.rec.on { background: #c62828; animation: pulse 1.4s ease-in-out infinite; }
button:disabled { opacity: .5; cursor: not-allowed; animation: none; }
@keyframes pulse { 50% { box-shadow: 0 0 0 .35rem rgba(198,40,40,.25); } }
.status { font-size: .9rem; opacity: .8; }
.legend { display: flex; gap: 1rem; flex-wrap: wrap; font-size: .8rem; opacity: .75; margin-bottom: .75rem; }
.legend span { display: inline-flex; align-items: center; gap: .35rem; }
.dot { width: .8rem; height: .8rem; border-radius: .25rem; display: inline-block; }

.mushaf {
  font-family: 'Amiri Quran', 'Amiri', serif; font-size: 2.15rem; line-height: 2.7;
  text-align: center; padding: 1rem 1.25rem 1.5rem;
  background: var(--st-secondary-background-color);
  border-radius: 1rem; border: 1px solid rgba(176,141,46,.45);
  box-shadow: inset 0 0 0 .35rem var(--st-background-color), inset 0 0 0 .45rem rgba(176,141,46,.35);
}
.sura-name {
  display: block; width: fit-content; margin: .25rem auto .5rem; padding: 0 2.5rem;
  font-size: 1.6rem; line-height: 2.2; color: #b08d2e;
  border: 2px solid rgba(176,141,46,.6); border-radius: 999px;
}
.basmala { display: block; font-size: 1.9rem; margin-bottom: .25rem; }
.w { padding: 0 .14em; border-radius: .35em; transition: background-color .25s, color .25s; cursor: default; }
.nb { white-space: nowrap; }
.end { color: #b08d2e; font-size: .95em; }

.ok { color: #2e9e5b; background: rgba(46,158,91,.16); }
.wrong, .missed { color: #e0464e; background: rgba(224,70,78,.16); }
.missed { text-decoration: line-through; text-decoration-thickness: .06em; }
.tajweed {
  color: #e0464e; background: rgba(224,70,78,.10);
  text-decoration: underline wavy #e0a03a; text-decoration-thickness: .05em; text-underline-offset: .35em;
}
.current { background: rgba(66,133,244,.18); }
.dot.ok { background: #2e9e5b; }
.dot.wrong { background: #e0464e; }
.dot.tajweed { background: #e0464e; box-shadow: inset 0 -3px 0 #e0a03a; }
.dot.current { background: rgba(66,133,244,.55); }
"""

JS = """
const FONT_URL = "https://fonts.googleapis.com/css2?family=Amiri+Quran&display=swap";
const TARGET_RATE = 16000;
const TICK_MS = 700;
const instances = new WeakMap();

function ensureFont() {
  // Components render in a shadow root, so fonts must be registered on the document.
  if (document.querySelector("link[data-quran-font]")) return;
  const link = document.createElement("link");
  link.rel = "stylesheet";
  link.href = FONT_URL;
  link.dataset.quranFont = "1";
  document.head.appendChild(link);
}

const arabicDigits = (n) => String(n).replace(/[0-9]/g, (d) => "٠١٢٣٤٥٦٧٨٩"[d]);

function toBase64(int16) {
  const bytes = new Uint8Array(int16.buffer);
  let s = "";
  for (let i = 0; i < bytes.length; i += 0x8000) {
    s += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  }
  return btoa(s);
}

function buildMushaf(el, data) {
  el.innerHTML = "";
  const title = document.createElement("span");
  title.className = "sura-name";
  title.textContent = "سورة " + data.name;
  el.appendChild(title);

  let line = null;
  data.words.forEach((w, i) => {
    if (w.a === 0) {
      if (!line) { line = document.createElement("span"); line.className = "basmala"; el.appendChild(line); }
    } else if (!line || line.className === "basmala") {
      line = document.createElement("span");
      el.appendChild(line);
    }
    const word = document.createElement("span");
    word.className = "w pending";
    word.dataset.i = i;
    word.textContent = w.t;
    if (w.e && w.a > 0) {
      const group = document.createElement("span");
      group.className = "nb";
      const end = document.createElement("span");
      end.className = "end";
      end.textContent = "\\u06DD" + arabicDigits(w.a);
      group.append(word, "\\u00A0", end);
      line.append(group, " ");
    } else {
      line.append(word, " ");
    }
  });
}

function flush(inst) {
  if (!inst.buf.length) return;
  let n = inst.leftover.length;
  for (const b of inst.buf) n += b.length;
  const x = new Float32Array(n);
  x.set(inst.leftover, 0);
  let o = inst.leftover.length;
  for (const b of inst.buf) { x.set(b, o); o += b.length; }
  inst.buf = [];

  // Box-filter downsample to 16 kHz (no-op when the context already runs at 16 kHz).
  const r = inst.ratio;
  const m = Math.floor(x.length / r);
  const out = new Int16Array(m);
  for (let j = 0; j < m; j++) {
    const s = Math.floor(j * r), e = Math.max(s + 1, Math.floor((j + 1) * r));
    let acc = 0;
    for (let k = s; k < e; k++) acc += x[k];
    const v = Math.max(-1, Math.min(1, acc / (e - s)));
    out[j] = v < 0 ? v * 0x8000 : v * 0x7fff;
  }
  inst.leftover = x.slice(Math.floor(m * r));
  if (m) inst.chunks.push({ seq: ++inst.seq, b64: toBase64(out) });
}

function send(inst, final) {
  flush(inst);
  const pending = inst.chunks.filter((c) => c.seq > inst.acked);
  const now = performance.now();
  const caughtUp = inst.acked >= inst.lastSentSeq;
  // Don't flood Python: wait for the previous batch to be acked (or retry after a while).
  if (!final && (!pending.length || (!caughtUp && now - inst.lastSend < 2500))) return;
  inst.lastSentSeq = inst.seq;
  inst.lastSend = now;
  inst.setTriggerValue("audio", { session: inst.session, chunks: pending, final });
}

async function start(inst) {
  try {
    inst.stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: true },
    });
  } catch (err) {
    inst.localMessage = "Microphone blocked: " + err.message;
    inst.render();
    return;
  }
  let ctx, src;
  try {
    ctx = new AudioContext({ sampleRate: TARGET_RATE });
    src = ctx.createMediaStreamSource(inst.stream);
  } catch (_) {  // e.g. Firefox can't mix sample rates; resample in JS instead
    ctx = new AudioContext();
    src = ctx.createMediaStreamSource(inst.stream);
  }
  const proc = ctx.createScriptProcessor(4096, 1, 1);
  const mute = ctx.createGain();
  mute.gain.value = 0;
  proc.onaudioprocess = (e) => inst.buf.push(new Float32Array(e.inputBuffer.getChannelData(0)));
  src.connect(proc);
  proc.connect(mute);
  mute.connect(ctx.destination);

  Object.assign(inst, {
    ctx, ratio: ctx.sampleRate / TARGET_RATE, recording: true, finalSent: false,
    session: (crypto.randomUUID ? crypto.randomUUID() : String(Date.now() + Math.random())),
    seq: 0, acked: 0, lastSentSeq: 0, lastSend: 0, chunks: [], buf: [], leftover: new Float32Array(0),
    t0: performance.now(), localMessage: "",
  });
  inst.timer = setInterval(() => { send(inst, false); inst.render(); }, TICK_MS);
  inst.render();
}

function stopAudio(inst) {
  clearInterval(inst.timer);
  if (inst.stream) inst.stream.getTracks().forEach((t) => t.stop());
  if (inst.ctx) inst.ctx.close();
  inst.stream = inst.ctx = null;
  inst.recording = false;
}

function stop(inst) {
  stopAudio(inst);
  inst.finalSent = true;
  send(inst, true);
  // Resend the final batch until Python leaves the "live" phase. Slow on CPU servers,
  // so wait long enough not to interrupt a transcription that's still running.
  clearInterval(inst.retry);
  inst.retry = setInterval(() => {
    const d = inst.data;
    if (!inst.finalSent || d.session !== inst.session || d.phase !== "live") { clearInterval(inst.retry); return; }
    send(inst, true);
  }, 8000);
  inst.render();
}

export default function (component) {
  const { data, parentElement, setTriggerValue } = component;
  ensureFont();

  let inst = instances.get(parentElement);
  if (!inst) {
    inst = { recording: false, session: null, seq: 0, acked: 0, lastSentSeq: 0, chunks: [], buf: [],
             leftover: new Float32Array(0), builtFor: null, localMessage: "" };
    instances.set(parentElement, inst);
  }
  inst.data = data;
  inst.setTriggerValue = setTriggerValue;

  const mushaf = parentElement.querySelector(".mushaf");
  const rec = parentElement.querySelector(".rec");
  const reset = parentElement.querySelector(".reset");
  const status = parentElement.querySelector(".status");

  const builtFor = data.sura + ":" + data.words.length;
  if (inst.builtFor !== builtFor) {
    buildMushaf(mushaf, data);
    inst.builtFor = builtFor;
  }

  if (data.session && data.session === inst.session) {
    inst.acked = Math.max(inst.acked, data.acked || 0);
    inst.chunks = inst.chunks.filter((c) => c.seq > inst.acked);
    if (inst.finalSent && data.phase !== "live") inst.finalSent = false;
  }

  inst.render = () => {
    const d = inst.data;
    mushaf.querySelectorAll(".w").forEach((el) => {
      const i = +el.dataset.i;
      const cls = "w " + (d.status[i] || "pending");
      if (el.className !== cls) el.className = cls;
      el.title = d.tips[i] || "";
    });
    rec.textContent = inst.recording ? "■ Stop" : "● Start reading";
    rec.classList.toggle("on", inst.recording);
    rec.disabled = !d.ready && !inst.recording;
    let msg = inst.localMessage || d.message || "";
    if (inst.recording) {
      const secs = ((performance.now() - inst.t0) / 1000).toFixed(1);
      msg = "Listening… " + secs + "s" + (d.message ? " · " + d.message : "");
    }
    status.textContent = msg;
  };

  rec.onclick = () => (inst.recording ? stop(inst) : start(inst));
  reset.onclick = () => {
    if (inst.recording) stopAudio(inst);
    inst.session = null;
    inst.finalSent = false;
    inst.localMessage = "";
    setTriggerValue("reset", Date.now());
  };

  inst.render();
  return () => {
    stopAudio(inst);
    clearInterval(inst.retry);
    instances.delete(parentElement);
  };
}
"""

_READER = st.components.v2.component("quran_reader", html=HTML, css=CSS, js=JS)


def quran_reader(*, key: str, data: dict, on_audio: Callable[[], None], on_reset: Callable[[], None]):
    return _READER(key=key, data=data, on_audio_change=on_audio, on_reset_change=on_reset)
