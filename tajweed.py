"""Tajweed checking with Quran Muaalem (obadx/muaalem-model-v3_2)."""

import threading
from dataclasses import dataclass, field

import numpy as np
import torch
from quran_muaalem.inference import Muaalem
from quran_transcript import MoshafAttributes, explain_error, quran_phonetizer

from quran import Sura, align, tokenize
from transcriber import SAMPLE_RATE, WhisperEngine

MUAALEM_MODEL = "obadx/muaalem-model-v3_2"

ERROR_TYPES_AR = {"tajweed": "تجويد", "tashkeel": "تشكيل", "normal": "حروف"}
SPEECH_ERRORS_AR = {"insert": "زيادة", "delete": "نقص", "replace": "إبدال"}


@dataclass
class TajweedError:
    word: int  # word index in the passage
    kind: str  # tajweed / tashkeel / normal
    detail: str
    expected: str
    heard: str


@dataclass
class TajweedReport:
    errors: list[TajweedError] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)  # pieces Muaalem failed on


def default_moshaf(**overrides) -> MoshafAttributes:
    lens = {"madd_monfasel_len": 4, "madd_mottasel_len": 4, "madd_mottasel_waqf": 4, "madd_aared_len": 4}
    return MoshafAttributes(rewaya="hafs", **{**lens, **overrides})


def split_on_pauses(audio: np.ndarray, min_pause=0.35, min_speech=0.25) -> list[tuple[int, int]]:
    """Speech regions separated by pauses (reciters stop between ayat)."""
    hop = SAMPLE_RATE // 100  # 10 ms frames
    n = len(audio) // hop
    if n == 0:
        return []
    rms = np.sqrt(np.mean(audio[: n * hop].reshape(n, hop) ** 2, axis=1) + 1e-10)
    db = 20 * np.log10(rms)
    floor, peak = np.percentile(db, 10), np.percentile(db, 95)
    speech = db > floor + 0.3 * (peak - floor)
    regions, start, silent = [], None, 0
    for i, s in enumerate(speech):
        if s:
            start = i if start is None else start
            silent = 0
        elif start is not None:
            silent += 1
            if silent >= min_pause * 100:
                regions.append((start, i - silent + 1))
                start, silent = None, 0
    if start is not None:
        regions.append((start, n))
    pad = 10  # 100 ms of context each side
    return [
        (max(0, a - pad) * hop, min(n, b + pad) * hop)
        for a, b in regions
        if (b - a) >= min_speech * 100
    ]


class TajweedEngine:
    def __init__(self):
        cuda = torch.cuda.is_available()
        # bfloat16 is what the model was trained with.
        self.model = Muaalem(
            MUAALEM_MODEL, device="cuda" if cuda else "cpu", dtype=torch.bfloat16 if cuda else torch.float32
        )
        self._lock = threading.Lock()
        ref = quran_phonetizer("قُلْ", default_moshaf(), remove_spaces=True)
        self.model([np.zeros(SAMPLE_RATE, dtype=np.float32)], [ref], sampling_rate=SAMPLE_RATE)

    def check(self, audio: np.ndarray, sura: Sura, whisper: WhisperEngine, moshaf: MoshafAttributes) -> TajweedReport:
        """Split at pauses, find which words each piece covers (via Whisper), then run Muaalem per piece.

        Each piece is phonetized on its own, so its end is treated as waqf and its start as ibtida,
        which is what the reciter actually did at the pause.
        """
        pieces = split_on_pauses(audio)
        texts = [whisper.transcribe(audio[a:b]).text for a, b in pieces]

        # Align all pieces at once so repeated words resolve in order.
        alignment = align(sura, " ".join(texts))
        spans, letter = [], 0
        for text in texts:
            n_letters = sum(len(tokenize(w)) for w in text.split())
            owners = [o for o in alignment.hyp_letter_to_ref_word[letter : letter + n_letters] if o is not None]
            letter += n_letters
            spans.append((min(owners), max(owners)) if owners else None)

        jobs = []  # (wave, uthmani text, first word index)
        prev_end = -1
        for (a, b), span in zip(pieces, spans):
            if span is None:
                continue
            first, last = max(span[0], prev_end + 1), span[1]
            if first > last:
                continue
            prev_end = last
            uthmani = " ".join(w.text for w in sura.words[first : last + 1])
            jobs.append((audio[a:b], uthmani, first))

        report = TajweedReport()
        for wave, uthmani, first in jobs:
            try:
                ref = quran_phonetizer(uthmani, moshaf, remove_spaces=True)
                # One piece per call: batching pads the reference ids, and quran-muaalem 0.2.2's
                # decoder then indexes them with an unpadded mask (IndexError on mismatches).
                with self._lock:
                    out = self.model([wave], [ref], sampling_rate=SAMPLE_RATE)[0]
                found = explain_error(uthmani, ref.phonemes, out.phonemes.text, ref.mappings)
            except Exception as e:  # library edge cases shouldn't lose the whole check
                report.skipped.append(f"{uthmani} ({type(e).__name__})")
                continue
            word_starts = np.cumsum([0] + [len(w) + 1 for w in uthmani.split()])
            for e in found:
                word = first + int(np.searchsorted(word_starts, e.uthmani_pos[0], side="right") - 1)
                report.errors.append(TajweedError(word, e.error_type, _describe(e), e.expected_ph, e.preditected_ph))
        return report


def _describe(e) -> str:
    rules = e.ref_tajweed_rules or e.missing_tajweed_rules or []
    parts = [ERROR_TYPES_AR[e.error_type], SPEECH_ERRORS_AR[e.speech_error_type]]
    if rules:
        parts.append(" / ".join(r.name.ar for r in rules))
    if e.expected_len is not None and e.predicted_len is not None:
        parts.append(f"المطلوب {e.expected_len} حركات، المقروء {e.predicted_len}")
    return " · ".join(parts)
