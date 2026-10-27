import os

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import base64
import hashlib
from dataclasses import dataclass, field

import numpy as np
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from models import SPECS, ModelManager, memory
from quran import CURRENT, MISSED, OK, WRONG, Alignment, Sura, align, load_sura
from reader_component import quran_reader
from tajweed import ERROR_TYPES_AR, MUAALEM_MODEL, TajweedError, default_moshaf
from transcriber import AZURE_MODEL, SAMPLE_RATE, WHISPER_MODELS, device_info

SURA = 112  # Al-Ikhlas
READER_KEY = "reader"

# Hosting switches (see docker-compose.yml): models to load at start-up, and whether
# visitors may load / unload models from the sidebar.
PRELOAD_MODELS = [n.strip() for n in os.environ.get("PRELOAD_MODELS", "").split(",") if n.strip() in SPECS]
ALLOW_MODEL_CONTROL = os.environ.get("ALLOW_MODEL_CONTROL", "true").lower() != "false"

st.set_page_config(page_title="Quran speech lab", page_icon="🎙️", layout="centered")


@st.cache_resource(show_spinner=False)
def model_manager() -> ModelManager:
    manager = ModelManager()
    if PRELOAD_MODELS:
        manager.preload(PRELOAD_MODELS)
    return manager


mm = model_manager()


def default_model(names) -> str:
    """First already-loaded model, so hosted pages open on something usable."""
    names = list(names)
    return next((n for n in names if mm.is_loaded(n) or n in mm.loading), names[0])


def load_model(name: str):
    with st.spinner(f"Loading {name}… (first run downloads it)"):
        try:
            mm.load(name)
        except Exception as e:
            st.error(f"Could not load {name}: {e}")
            return
    st.rerun()


def require_model(name: str, where: str):
    """Return the engine, or show an inline Load button."""
    engine = mm.get(name)
    if engine is not None:
        return engine
    if name in mm.loading:
        st.caption(f"**{name}** is loading… refresh in a moment.")
    elif not ALLOW_MODEL_CONTROL:
        st.caption(f"**{name}** isn't available on this server.")
    else:
        with st.container(horizontal=True, vertical_alignment="center"):
            st.caption(f"**{name}** isn't loaded.")
            if st.button("Load", key=f"load-{where}-{name}", icon=":material/download:"):
                load_model(name)
    return None


# --- Sidebar: model manager ------------------------------------------------

with st.sidebar:
    st.header("Models")
    st.caption(f"Device: **{device_info()}**")
    label, used, total = memory()
    st.progress(min(used / total, 1.0), text=f"{label} {used / 2**30:.1f} / {total / 2**30:.1f} GB (whole machine)")
    for spec in SPECS.values():
        loaded = mm.is_loaded(spec.name)
        if not spec.local:
            size = "cloud"
        elif loaded:
            size = f"{mm.mem_bytes(spec.name) / 2**30:.2f} GB"
        else:
            size = f"~{spec.approx_gb} GB"
        if loaded:
            badge = ":green[● loaded]"
        elif spec.name in mm.loading:
            badge = ":orange[◌ loading…]"
        elif spec.name in mm.errors:
            badge = ":red[✕ failed]"
        else:
            badge = ":gray[○ off]"
        with st.container(horizontal=True, vertical_alignment="center"):
            st.markdown(f"**{spec.name}**  \n{badge} · {size}", width="stretch")
            if not ALLOW_MODEL_CONTROL or spec.name in mm.loading:
                continue
            if loaded:
                st.button("Unload", key=f"unload-{spec.name}", on_click=mm.unload, args=(spec.name,))
            elif st.button("Load", key=f"load-side-{spec.name}"):
                load_model(spec.name)


# --- Tab 1: transcribe -----------------------------------------------------


def transcribe_tab():
    st.write("Recite a verse or upload a recording, then compare models.")
    options = [*WHISPER_MODELS, AZURE_MODEL]
    name = st.segmented_control("Model", options, default=default_model(options), key="tx_model")
    if not name or (engine := require_model(name, "tx")) is None:
        return

    mic_tab, file_tab = st.tabs([":material/mic: Record", ":material/upload_file: Upload"])
    with mic_tab:
        recorded = st.audio_input("Record your recitation")
    with file_tab:
        uploaded = st.file_uploader("Audio file", type=["wav", "flac", "ogg", "mp3"])
        if uploaded:
            st.audio(uploaded)

    results = st.session_state.setdefault("tx_results", {})
    for audio in (recorded, uploaded):
        if audio is None:
            continue
        data = audio.getvalue()
        cache_key = (name, hashlib.sha1(data).hexdigest())
        if cache_key not in results:
            with st.spinner("Transcribing…"):
                try:
                    results[cache_key] = engine(data)
                except Exception as e:
                    st.error(f"Transcription failed: {e}")
                    continue
        result = results[cache_key]
        st.markdown(
            f"<div dir='rtl' style='font-size:1.6rem;line-height:2.4rem;padding:1rem;"
            f"border-radius:.5rem;background:rgba(128,128,128,.12)'>{result.text or '—'}</div>",
            unsafe_allow_html=True,
        )
        st.caption(
            f"{result.audio_seconds:.1f}s audio · {result.infer_seconds:.2f}s inference · "
            f"{result.audio_seconds / max(result.infer_seconds, 1e-6):.0f}× real-time"
        )


# --- Tab 2: read Quran -----------------------------------------------------


@dataclass
class ReaderState:
    session: str | None = None
    last_seq: int = 0
    pcm: list[np.ndarray] = field(default_factory=list)
    phase: str = "idle"  # idle → live → analyzing → done
    hyp: str = ""
    alignment: Alignment | None = None
    tajweed: list[TajweedError] = field(default_factory=list)
    tajweed_skipped: list[str] = field(default_factory=list)
    tajweed_ran: bool = False
    latency: float = 0.0

    def audio(self) -> np.ndarray:
        if not self.pcm:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self.pcm).astype(np.float32) / 32768


def reader_state() -> ReaderState:
    return st.session_state.setdefault("reader_state", ReaderState())


def on_audio(sura: Sura, whisper_name: str, check_harakat: bool):
    payload = st.session_state[READER_KEY].audio
    if not payload:
        return
    rs = reader_state()
    if payload["session"] != rs.session:
        rs = st.session_state["reader_state"] = ReaderState(session=payload["session"])
    elif rs.phase in ("analyzing", "done"):
        return  # a resent "stop"; re-running would restart the slow tajweed check
    for chunk in payload["chunks"]:
        if chunk["seq"] > rs.last_seq:
            rs.pcm.append(np.frombuffer(base64.b64decode(chunk["b64"]), dtype="<i2"))
            rs.last_seq = chunk["seq"]

    final = payload["final"]
    audio = rs.audio()
    whisper = mm.get(whisper_name)
    if whisper is None or len(audio) < SAMPLE_RATE // 2:
        rs.phase = "done" if final else "live"
        return
    result = whisper.transcribe(audio)
    rs.hyp, rs.latency = result.text, result.infer_seconds
    rs.alignment = align(sura, result.text, check_harakat=check_harakat, live=not final)
    rs.phase = "analyzing" if final else "live"


def on_reset():
    st.session_state["reader_state"] = ReaderState()


def word_view(sura: Sura, rs: ReaderState) -> tuple[list[str], list[str]]:
    n = len(sura.words)
    statuses = [s.status for s in rs.alignment.scores] if rs.alignment else ["pending"] * n
    tips = [""] * n
    if rs.alignment:
        for i, s in enumerate(rs.alignment.scores):
            if s.status in (WRONG, CURRENT) and s.heard:
                tips[i] = f"سُمِع: {s.heard}"
            elif s.status == MISSED:
                tips[i] = "لم تُقرأ"
    for e in rs.tajweed:
        if statuses[e.word] in (OK, WRONG):
            statuses[e.word] = "tajweed"
        tips[e.word] = "\n".join(t for t in (tips[e.word], e.detail) if t)
    return statuses, tips


def status_message(sura: Sura, rs: ReaderState, whisper_loaded: bool) -> str:
    if not whisper_loaded:
        return "Load a Whisper model to start."
    if rs.phase == "live" and rs.hyp:
        return f"Whisper {rs.latency * 1000:.0f} ms"
    if rs.phase == "analyzing":
        return "Checking…"
    if rs.phase == "done" and rs.alignment:
        scores = [s for s, w in zip(rs.alignment.scores, sura.words) if w.ayah > 0]
        ok = sum(s.status == OK for s in scores)
        notes = f" · {len(rs.tajweed)} tajweed notes" if rs.tajweed_ran else ""
        return f"{ok}/{len(scores)} words correct{notes}"
    return "Press start and recite."


@st.fragment
def reader(sura: Sura, whisper_name: str, check_harakat: bool, moshaf_lens: dict):
    rs = reader_state()
    whisper = mm.get(whisper_name)
    statuses, tips = word_view(sura, rs)
    quran_reader(
        key=READER_KEY,
        data={
            "sura": sura.number,
            "name": sura.name,
            "words": [{"t": w.text, "a": w.ayah, "e": w.ayah_end} for w in sura.words],
            "status": statuses,
            "tips": tips,
            "session": rs.session,
            "acked": rs.last_seq,
            "phase": rs.phase,
            "message": status_message(sura, rs, whisper is not None),
            "ready": whisper is not None,
        },
        on_audio=lambda: on_audio(sura, whisper_name, check_harakat),
        on_reset=on_reset,
    )

    if rs.phase == "analyzing":
        tajweed = mm.get(MUAALEM_MODEL)
        if tajweed is not None and whisper is not None:
            with st.spinner("Checking tajweed with Muaalem…"):
                report = tajweed.check(rs.audio(), sura, whisper, default_moshaf(**moshaf_lens))
                rs.tajweed, rs.tajweed_skipped = report.errors, report.skipped
                rs.tajweed_ran = True
        rs.phase = "done"
        st.rerun(scope="fragment")

    if rs.phase != "done":
        return
    if rs.pcm:
        st.audio(rs.audio(), sample_rate=SAMPLE_RATE)
    if rs.tajweed_skipped:
        st.warning(
            "Muaalem couldn't check: " + " · ".join(rs.tajweed_skipped), icon=":material/warning:"
        )
    if not rs.tajweed_ran:
        hint = "Load **Muaalem (tajweed)** in the sidebar to" if ALLOW_MODEL_CONTROL else "Once **Muaalem** is loaded, the reader will"
        st.info(f"{hint} also check tajweed when you stop.", icon=":material/info:")
    elif rs.tajweed:
        st.dataframe(
            [
                {
                    "Word": sura.words[e.word].text,
                    "Type": ERROR_TYPES_AR[e.kind],
                    "Detail": e.detail,
                    "Expected": e.expected,
                    "Heard": e.heard,
                }
                for e in rs.tajweed
            ],
            hide_index=True,
        )
    else:
        st.success("No tajweed notes from Muaalem.", icon=":material/check_circle:")
    with st.expander("What Whisper heard"):
        st.markdown(f"<div dir='rtl' style='font-size:1.3rem'>{rs.hyp or '—'}</div>", unsafe_allow_html=True)


def read_tab():
    sura = load_sura(SURA)
    with st.container(horizontal=True, vertical_alignment="bottom"):
        whisper_name = st.segmented_control(
            "Live model", list(WHISPER_MODELS), default=default_model(WHISPER_MODELS), key="read_model"
        )
        check_harakat = st.toggle("Check harakat", value=True, key="read_harakat")
    with st.expander("Moshaf settings (Hafs · madd lengths)"):
        with st.container(horizontal=True):
            moshaf_lens = {
                "madd_monfasel_len": st.selectbox("Monfasel", [2, 3, 4, 5], index=2),
                "madd_mottasel_len": st.selectbox("Mottasel", [4, 5, 6], index=0),
                "madd_aared_len": st.selectbox("Aared (waqf)", [2, 4, 6], index=1),
            }
    if not whisper_name:
        return
    require_model(whisper_name, "read")
    if not mm.is_loaded(MUAALEM_MODEL):
        st.caption("Tajweed check runs after you stop, if **Muaalem** is loaded.")
    reader(sura, whisper_name, check_harakat, moshaf_lens)
    st.caption("Text: [Tanzil.net](https://tanzil.net) (Uthmani) · Font: Amiri Quran · Tajweed: Quran Muaalem")


tx, rd = st.tabs([":material/graphic_eq: Transcribe", ":material/menu_book: Read Quran"], key="tab", on_change="rerun")
if tx.open:
    with tx:
        transcribe_tab()
if rd.open:
    with rd:
        read_tab()
