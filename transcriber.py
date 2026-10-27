"""Whisper (Tarteel Quran fine-tune) and audio decoding."""

import io
import threading
import time
from dataclasses import dataclass

import av
import numpy as np
import torch
from transformers import GenerationConfig, pipeline
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()  # silences the per-call "chunk_length_s is experimental" warning

SAMPLE_RATE = 16_000
WHISPER_MODEL = "tarteel-ai/whisper-base-ar-quran"
# The Tarteel checkpoint ships an outdated generation config that rejects `language=`,
# so we borrow the config from the original model it was fine-tuned from.
WHISPER_BASE_CONFIG = "openai/whisper-base"


@dataclass
class Result:
    text: str
    audio_seconds: float
    infer_seconds: float


def device_info() -> str:
    if torch.cuda.is_available():
        return f"GPU · {torch.cuda.get_device_name(0)}"
    return "CPU"


def decode_audio(data: bytes) -> np.ndarray:
    """Decode wav/mp3/ogg/opus/webm/m4a… bytes to mono float32 at 16 kHz (no ffmpeg needed)."""
    resampler = av.AudioResampler(format="flt", layout="mono", rate=SAMPLE_RATE)
    chunks = []
    with av.open(io.BytesIO(data)) as container:
        for frame in container.decode(audio=0):
            chunks += [f.to_ndarray()[0] for f in resampler.resample(frame)]
    chunks += [f.to_ndarray()[0] for f in resampler.resample(None)]  # flush
    return np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, np.float32)


class WhisperEngine:
    def __init__(self):
        cuda = torch.cuda.is_available()
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=WHISPER_MODEL,
            device="cuda:0" if cuda else "cpu",
            dtype=torch.float16 if cuda else torch.float32,
        )
        # The pipeline keeps its own copy, so replace both.
        self.pipe.model.generation_config = self.pipe.generation_config = GenerationConfig.from_pretrained(
            WHISPER_BASE_CONFIG
        )
        self._lock = threading.Lock()
        # Warm-up so the first real request isn't slow (CUDA kernels, allocator).
        self.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32))

    def transcribe(self, audio: np.ndarray) -> Result:
        start = time.perf_counter()
        with self._lock:
            out = self.pipe(
                {"raw": audio, "sampling_rate": SAMPLE_RATE},
                chunk_length_s=30,
                batch_size=8,
                generate_kwargs={"language": "arabic", "task": "transcribe"},
            )
        return Result(out["text"].strip(), len(audio) / SAMPLE_RATE, time.perf_counter() - start)
