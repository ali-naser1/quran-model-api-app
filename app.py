"""Streamlit test client for the API (start it first with .\\run_api.ps1)."""

import base64
import hashlib
import io
import os
import wave
from dataclasses import dataclass, field

import requests
import streamlit as st
from dotenv import load_dotenv

from reader_component import quran_reader

load_dotenv()
# 127.0.0.1, not localhost: on Windows "localhost" tries IPv6 first and adds ~2 s per request.
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")
DOCS_URL = os.environ.get("API_DOCS_URL", f"{API_URL}/docs")  # as the browser sees it
SURAS = {112: "Al-Ikhlas", 114: "An-Nas"}
READER_KEY = "reader"
SAMPLE_RATE = 16_000

st.set_page_config(page_title="Quran speech lab", page_icon="🎙️", layout="centered")


@st.cache_resource
def http() -> requests.Session:
    session = requests.Session()
    if key := os.environ.get("API_KEY"):
        session.headers["X-API-Key"] = key
    return session


def api(method: str, path: str, **kwargs) -> dict:
    resp = http().request(method, f"{API_URL}{path}", timeout=120, **kwargs)
    if not resp.ok:
        raise RuntimeError(f"{resp.status_code}: {resp.text[:300]}")
    return resp.json()


def to_wav(pcm: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)
    return buf.getvalue()


# --- Sidebar: API status ---------------------------------------------------

with st.sidebar:
    st.header("API")
    st.caption(API_URL)
    try:
        health = api("GET", "/health")
    except Exception:
        st.error("API not reachable. Start it with `.\\run_api.ps1`.", icon=":material/cloud_off:")
        st.stop()
    if health["ready"]:
        st.success("Ready", icon=":material/check_circle:")
    else:
        st.warning("Loading models…", icon=":material/hourglass_top:")
    st.caption(f"{health['device']} · {health['vram_gb']} GB VRAM")
    for role, model in health["models"].items():
        st.caption(f"**{role}** · `{model}`")
    st.link_button("API docs", DOCS_URL, icon=":material/description:")


# --- Tab 1: Whisper API ----------------------------------------------------


def transcribe_tab():
    st.write("Record or upload a recitation → `POST /whisper/transcribe`.")
    mic_tab, file_tab = st.tabs([":material/mic: Record", ":material/upload_file: Upload"])
    with mic_tab:
        recorded = st.audio_input("Record your recitation")
    with file_tab:
        uploaded = st.file_uploader("Audio file", type=["wav", "flac", "ogg", "mp3", "webm", "m4a"])
        if uploaded:
            st.audio(uploaded)

    results = st.session_state.setdefault("tx_results", {})
    for audio in (recorded, uploaded):
        if audio is None:
            continue
        data = audio.getvalue()
        key = hashlib.sha1(data).hexdigest()
        if key not in results:
            with st.spinner("Transcribing…"):
                try:
                    results[key] = api("POST", "/whisper/transcribe", files={"file": (audio.name, data)})
                except Exception as e:
                    st.error(f"Transcription failed: {e}")
                    continue
        r = results[key]
        st.markdown(
            f"<div dir='rtl' style='font-size:1.6rem;line-height:2.4rem;padding:1rem;"
            f"border-radius:.5rem;background:rgba(128,128,128,.12)'>{r['text'] or '—'}</div>",
            unsafe_allow_html=True,
        )
        st.caption(
            f"{r['audio_seconds']:.1f}s audio · {r['infer_seconds']:.2f}s inference · "
            f"{r['audio_seconds'] / max(r['infer_seconds'], 1e-6):.0f}× real-time"
        )


# --- Tab 2: Tajweed API (live reader) ---------------------------------------


@dataclass
class ReaderState:
    session: str | None = None
    last_seq: int = 0
    pcm: list[bytes] = field(default_factory=list)
    phase: str = "idle"  # idle → live → analyzing → done
    result: dict | None = None
    error: str = ""

    def wav(self) -> bytes:
        return to_wav(b"".join(self.pcm))


def reader_state() -> ReaderState:
    return st.session_state.setdefault("reader_state", ReaderState())


def check(rs: ReaderState, settings: dict, *, tajweed: bool, live: bool):
    try:
        rs.result = api(
            "POST",
            "/tajweed/check",
            params={"tajweed": tajweed, "live": live, **settings},
            files={"file": ("recitation.wav", rs.wav(), "audio/wav")},
        )
        rs.error = ""
    except Exception as e:
        rs.error = str(e)


def on_audio(settings: dict):
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
            rs.pcm.append(base64.b64decode(chunk["b64"]))
            rs.last_seq = chunk["seq"]
    if payload["final"]:
        rs.phase = "analyzing"  # full check runs in the fragment body, under a spinner
    else:
        rs.phase = "live"
        if sum(map(len, rs.pcm)) >= SAMPLE_RATE:  # ≥ 0.5 s of int16 audio
            check(rs, settings, tajweed=False, live=True)


def on_reset():
    st.session_state["reader_state"] = ReaderState()


def word_view(words: list[dict], rs: ReaderState) -> tuple[list[str], list[str]]:
    if not rs.result:
        return ["pending"] * len(words), [""] * len(words)
    statuses, tips = [], []
    for w in rs.result["words"]:
        lines = [n["detail"] for n in w["notes"]]
        if w["status"] in ("wrong", "current") and w["heard"]:
            lines.insert(0, f"سُمِع: {w['heard']}")
        elif w["status"] == "missed":
            lines.insert(0, "لم تُقرأ")
        statuses.append("tajweed" if w["notes"] else w["status"])
        tips.append("\n".join(lines))
    return statuses, tips


def status_message(rs: ReaderState) -> str:
    r = rs.result
    if rs.error:
        return f"API error: {rs.error[:120]}"
    if rs.phase == "live" and r:
        return f"Whisper {r['whisper_seconds'] * 1000:.0f} ms"
    if rs.phase == "analyzing":
        return "Checking…"
    if rs.phase == "done" and r:
        notes = sum(len(w["notes"]) for w in r["words"])
        return f"{r['correct']}/{r['total']} words correct · {notes} tajweed notes"
    return "Press start and recite."


@st.fragment
def reader(text: dict, settings: dict):
    rs = reader_state()
    statuses, tips = word_view(text["words"], rs)
    quran_reader(
        key=READER_KEY,
        data={
            "sura": text["sura"],
            "name": text["name"],
            "words": [{"t": w["text"], "a": w["ayah"], "e": w["ayah_end"]} for w in text["words"]],
            "status": statuses,
            "tips": tips,
            "session": rs.session,
            "acked": rs.last_seq,
            "phase": rs.phase,
            "message": status_message(rs),
        },
        on_audio=lambda: on_audio(settings),
        on_reset=on_reset,
    )

    if rs.phase == "analyzing":
        with st.spinner("Checking words and tajweed…"):
            check(rs, settings, tajweed=True, live=False)
        rs.phase = "done"
        st.rerun(scope="fragment")

    if rs.phase != "done" or not rs.result:
        return
    r = rs.result
    st.audio(rs.wav())
    st.caption(
        f"{r['audio_seconds']:.1f}s audio · Whisper {r['whisper_seconds']:.2f}s · tajweed {r['tajweed_seconds']:.2f}s"
    )
    if r["skipped"]:
        st.warning("Muaalem couldn't check: " + " · ".join(r["skipped"]), icon=":material/warning:")
    notes = [{"Word": w["text"], **{k.title(): v for k, v in n.items()}} for w in r["words"] for n in w["notes"]]
    if notes:
        st.dataframe(notes, hide_index=True)
    else:
        st.success("No tajweed notes from Muaalem.", icon=":material/check_circle:")
    with st.expander("Raw API response"):
        st.json(r, expanded=False)


@st.cache_data(show_spinner=False)
def load_text(sura: int) -> dict:
    return api("GET", "/tajweed/passage", params={"sura": sura})


def read_tab():
    st.write("Recite while words are checked live → `POST /tajweed/check`.")
    sura = st.segmented_control(
        "Sura", list(SURAS), default=112, required=True, format_func=SURAS.get, key="read_sura", on_change=on_reset
    )
    check_harakat = st.toggle("Check harakat", value=True, key="read_harakat")
    with st.expander("Moshaf settings (Hafs · madd lengths)"):
        with st.container(horizontal=True):
            settings = {
                "sura": sura,
                "check_harakat": check_harakat,
                "madd_monfasel_len": st.selectbox("Monfasel", [2, 3, 4, 5], index=2),
                "madd_mottasel_len": st.selectbox("Mottasel", [4, 5, 6], index=0),
                "madd_aared_len": st.selectbox("Aared (waqf)", [2, 4, 6], index=1),
            }
    reader(load_text(sura), settings)
    st.caption("Text: [Tanzil.net](https://tanzil.net) (Uthmani) · Font: Amiri Quran · Tajweed: Quran Muaalem")


tx, rd = st.tabs([":material/graphic_eq: Whisper API", ":material/menu_book: Tajweed API"], key="tab", on_change="rerun")
if tx.open:
    with tx:
        transcribe_tab()
if rd.open:
    with rd:
        read_tab()
