"""Speech-to-text engines (local Whisper, Azure cloud), kept separate from the UI."""

import io
import math
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import azure.cognitiveservices.speech as speechsdk
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import GenerationConfig, pipeline
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()  # silences the per-call "chunk_length_s is experimental" warning

SAMPLE_RATE = 16_000


def _match_cpu_limit():
    """Size torch's thread pool to the container CPU limit.

    Torch counts host cores, so under `--cpus 4` on a 20-core host it starts 10 threads,
    and oversubscribing the limit made inference ~3× slower.
    """
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
    except (OSError, ValueError):
        return  # not Linux / no cgroup v2
    if quota != "max":
        torch.set_num_threads(max(1, math.ceil(int(quota) / int(period))))


_match_cpu_limit()

# name -> (fine-tuned model, original Whisper it was trained from).
# The Tarteel checkpoints ship an outdated generation config that rejects
# `language=`, so we borrow the config from the original OpenAI model.
WHISPER_MODELS = {
    "Whisper Base (Quran)": ("tarteel-ai/whisper-base-ar-quran", "openai/whisper-base"),
    "Whisper Tiny (Quran)": ("tarteel-ai/whisper-tiny-ar-quran", "openai/whisper-tiny"),
}
AZURE_MODEL = "Azure Speech (ar-SA, cloud)"


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
    """Decode wav/flac/ogg/mp3 bytes to mono float32 at 16 kHz (no ffmpeg needed)."""
    audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        g = np.gcd(sr, SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // g, sr // g).astype(np.float32)
    return audio


class WhisperEngine:
    def __init__(self, name: str):
        model_id, base_id = WHISPER_MODELS[name]
        cuda = torch.cuda.is_available()
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device="cuda:0" if cuda else "cpu",
            dtype=torch.float16 if cuda else torch.float32,
            ignore_warning=True,
        )
        # The pipeline keeps its own copy, so replace both.
        self.pipe.model.generation_config = self.pipe.generation_config = GenerationConfig.from_pretrained(base_id)
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

    def __call__(self, data: bytes) -> Result:
        return self.transcribe(decode_audio(data))


class AzureEngine:
    """Azure Speech SDK continuous recognition (works in every region, any length)."""

    def __init__(self):
        key = os.environ.get("AZURE_SPEECH_KEY")
        endpoint = os.environ.get("AZURE_SPEECH_ENDPOINT", "").rstrip("/")
        if not key or not endpoint:
            raise RuntimeError("Set AZURE_SPEECH_KEY and AZURE_SPEECH_ENDPOINT in .env")
        host = urlparse(endpoint).hostname or ""
        if host.endswith(".api.cognitive.microsoft.com"):
            self.config = speechsdk.SpeechConfig(subscription=key, region=host.split(".")[0])
        else:  # custom-domain resource endpoint
            self.config = speechsdk.SpeechConfig(subscription=key, endpoint=endpoint)
        self.config.speech_recognition_language = "ar-SA"
        self.fmt = speechsdk.audio.AudioStreamFormat(samples_per_second=SAMPLE_RATE, bits_per_sample=16, channels=1)

    def transcribe(self, audio: np.ndarray) -> Result:
        pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
        stream = speechsdk.audio.PushAudioInputStream(self.fmt)
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=self.config, audio_config=speechsdk.audio.AudioConfig(stream=stream)
        )
        texts: list[str] = []
        errors: list[str] = []
        done = threading.Event()

        def on_recognized(evt):
            if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                texts.append(evt.result.text)

        def on_canceled(evt):
            if evt.cancellation_details.reason == speechsdk.CancellationReason.Error:
                errors.append(evt.cancellation_details.error_details)
            done.set()

        recognizer.recognized.connect(on_recognized)
        recognizer.canceled.connect(on_canceled)
        recognizer.session_stopped.connect(lambda _: done.set())

        start = time.perf_counter()
        recognizer.start_continuous_recognition()
        stream.write(pcm)
        stream.close()
        done.wait(timeout=300)
        recognizer.stop_continuous_recognition()
        if errors:
            raise RuntimeError(f"Azure: {errors[0]}")
        return Result(" ".join(texts).strip(), len(audio) / SAMPLE_RATE, time.perf_counter() - start)

    def __call__(self, data: bytes) -> Result:
        # Send the same normalized 16 kHz mono audio the local models see.
        return self.transcribe(decode_audio(data))
