"""HTTP APIs: Whisper transcription and recitation / tajweed checking.

Both models load once at start-up and stay on the GPU. Interactive docs at /docs.
Run with `.\\run_api.ps1`.
"""

import os
import time
from contextlib import asynccontextmanager

import torch
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, FastAPI, File, Header, HTTPException, Query, UploadFile
from pydantic import BaseModel

from quran import Sura, align, passage
from tajweed import MUAALEM_MODEL, TajweedEngine, default_moshaf
from transcriber import WHISPER_MODEL, WhisperEngine, decode_audio, device_info

load_dotenv()
API_KEY = os.environ.get("API_KEY", "")

engines: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    engines["whisper"] = WhisperEngine()
    engines["tajweed"] = TajweedEngine()
    yield
    engines.clear()


def require_key(x_api_key: str | None = Header(None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(401, "Missing or wrong X-API-Key header")


def read_audio(file: UploadFile):
    try:
        audio = decode_audio(file.file.read())
    except Exception as e:
        raise HTTPException(400, f"Could not decode audio: {e}")
    if len(audio) == 0:
        raise HTTPException(400, "Audio is empty")
    return audio


def get_passage(sura: int, ayah_from: int, ayah_to: int | None) -> Sura:
    try:
        return passage(sura, ayah_from, ayah_to)
    except ValueError as e:
        raise HTTPException(422, str(e))


# --- Schemas -----------------------------------------------------------------


class Transcript(BaseModel):
    text: str
    audio_seconds: float
    infer_seconds: float


class PassageWord(BaseModel):
    index: int
    text: str
    ayah: int  # 0 = basmala
    ayah_end: bool


class Passage(BaseModel):
    sura: int
    name: str
    words: list[PassageWord]


class Note(BaseModel):
    kind: str  # tajweed / tashkeel / normal
    detail: str  # Arabic description, e.g. "تجويد · إبدال · المد الطبيعي"
    expected: str  # phonemes
    heard: str


class CheckedWord(PassageWord):
    status: str  # ok / wrong / missed / pending (not reached) / current (live: being read)
    heard: str  # what Whisper heard for this word
    notes: list[Note]  # Muaalem tajweed notes


class Check(BaseModel):
    sura: int
    name: str
    transcript: str
    words: list[CheckedWord]
    correct: int  # words read correctly (basmala excluded)
    total: int
    tajweed_checked: bool
    skipped: list[str]  # pieces Muaalem couldn't check
    audio_seconds: float
    whisper_seconds: float
    tajweed_seconds: float


def passage_words(p: Sura) -> list[PassageWord]:
    return [PassageWord(index=i, text=w.text, ayah=w.ayah, ayah_end=w.ayah_end) for i, w in enumerate(p.words)]


# --- Routes ------------------------------------------------------------------

app = FastAPI(title="Quran speech API", lifespan=lifespan)
whisper_api = APIRouter(prefix="/whisper", tags=["whisper"], dependencies=[Depends(require_key)])
tajweed_api = APIRouter(prefix="/tajweed", tags=["tajweed"], dependencies=[Depends(require_key)])


@app.get("/health", tags=["status"])
def health():
    vram = torch.cuda.memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
    return {
        "ready": len(engines) == 2,
        "device": device_info(),
        "vram_gb": round(vram, 2),
        "models": {"whisper": WHISPER_MODEL, "tajweed": MUAALEM_MODEL},
    }


@whisper_api.post("/transcribe", response_model=Transcript)
def transcribe(file: UploadFile = File(..., description="Audio: wav, mp3, ogg, webm, m4a…")):
    """Arabic transcript (with harakat) of a Quran recitation."""
    r = engines["whisper"].transcribe(read_audio(file))
    return Transcript(text=r.text, audio_seconds=r.audio_seconds, infer_seconds=r.infer_seconds)


@tajweed_api.get("/passage", response_model=Passage)
def get_text(sura: int = 112, ayah_from: int = 1, ayah_to: int | None = None):
    """Uthmani words of the passage, with the same indices `/check` returns."""
    p = get_passage(sura, ayah_from, ayah_to)
    return Passage(sura=p.number, name=p.name, words=passage_words(p))


@tajweed_api.post("/check", response_model=Check)
def check(
    file: UploadFile = File(..., description="Audio of the recitation: wav, mp3, ogg, webm, m4a…"),
    sura: int = Query(112, ge=1, le=114),
    ayah_from: int = 1,
    ayah_to: int | None = None,
    tajweed: bool = Query(True, description="Run Muaalem (slower). False = words + harakat only, for live use"),
    check_harakat: bool = True,
    live: bool = Query(False, description="Recording still in progress: the last word reached is 'current', not wrong"),
    madd_monfasel_len: int = Query(4, ge=2, le=5),
    madd_mottasel_len: int = Query(4, ge=4, le=6),
    madd_aared_len: int = Query(4, description="2, 4 or 6"),
):
    """Check a recitation against the passage: each word's status, and tajweed notes from Muaalem."""
    p = get_passage(sura, ayah_from, ayah_to)
    audio = read_audio(file)
    whisper = engines["whisper"]
    heard = whisper.transcribe(audio)
    scores = align(p, heard.text, check_harakat=check_harakat, live=live).scores

    notes: dict[int, list[Note]] = {}
    skipped: list[str] = []
    tajweed_seconds = 0.0
    if tajweed:
        start = time.perf_counter()
        moshaf = default_moshaf(
            madd_monfasel_len=madd_monfasel_len, madd_mottasel_len=madd_mottasel_len, madd_aared_len=madd_aared_len
        )
        report = engines["tajweed"].check(audio, p, whisper, moshaf)
        tajweed_seconds = time.perf_counter() - start
        skipped = report.skipped
        for e in report.errors:
            if scores[e.word].status in ("ok", "wrong"):  # notes on unread words are noise
                notes.setdefault(e.word, []).append(Note(kind=e.kind, detail=e.detail, expected=e.expected, heard=e.heard))

    words = [
        CheckedWord(**w.model_dump(), status=s.status, heard=s.heard, notes=notes.get(w.index, []))
        for w, s in zip(passage_words(p), scores)
    ]
    counted = [w for w in words if w.ayah > 0]
    return Check(
        sura=p.number,
        name=p.name,
        transcript=heard.text,
        words=words,
        correct=sum(w.status == "ok" for w in counted),
        total=len(counted),
        tajweed_checked=tajweed,
        skipped=skipped,
        audio_seconds=heard.audio_seconds,
        whisper_seconds=heard.infer_seconds,
        tajweed_seconds=tajweed_seconds,
    )


app.include_router(whisper_api)
app.include_router(tajweed_api)
