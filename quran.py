"""Quran text from Tanzil and word-level scoring of a transcript against it."""

import unicodedata
import urllib.request
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from quran_transcript import Aya
from rapidfuzz.distance import Levenshtein

# The same download as Abjad's Quran dataset (genai-doc-processing-pipeline,
# quran/dataset/source.py), so every word's text is byte for byte what the reader shows.
TANZIL_URL = (
    "https://tanzil.net/pub/download/index.php"
    "?quranType=uthmani&outType=txt-2&marks=true&sajdah=true&rub=true&agree=true"
)
TANZIL_FILE = Path(__file__).parent / "data" / "tanzil_uthmani.txt"


@dataclass
class Word:
    text: str
    ayah: int  # 0 = basmala
    ayah_end: bool
    position: int  # 1-based within the ayah, as Abjad's quran_words


def _has_letter(token: str) -> bool:
    return any(unicodedata.category(ch).startswith("L") for ch in token)


def split_words(text: str) -> list[str]:
    """An ayah's words. A token with no letter (a pause mark, ۞, ۩) is not a word:
    it joins the word before it, or the first word when it comes first — the same
    rule as Abjad's quran_words (quran/recitation/words.py display_words)."""
    words: list[str] = []
    pending: list[str] = []
    for token in text.split():
        if _has_letter(token):
            words.append(" ".join([*pending, token]))
            pending = []
        elif words:
            words[-1] += f" {token}"
        else:
            pending.append(token)
    return words


def bare_word(word: str) -> str:
    """A word without the pause marks split_words attached to it."""
    return " ".join(t for t in word.split() if _has_letter(t))


def _letters(text: str) -> str:
    """Base letters only, for comparing spellings that differ in marks (بِّسْمِ / بِسْمِ)."""
    return "".join(ch for ch in text if ("ء" <= ch <= "ي" and ch != "ـ") or ch == "ٱ")


@dataclass
class Sura:
    number: int
    name: str
    words: list[Word]


def _tanzil_lines() -> list[str]:
    if not TANZIL_FILE.exists():
        TANZIL_FILE.parent.mkdir(exist_ok=True)
        with urllib.request.urlopen(TANZIL_URL, timeout=60) as resp:
            TANZIL_FILE.write_bytes(resp.read())
    return TANZIL_FILE.read_text(encoding="utf-8").splitlines()


@lru_cache
def load_sura(number: int) -> Sura:
    """Uthmani text from Tanzil. Tanzil prepends the basmala to ayah 1; it's split out as ayah 0."""
    ayat: dict[int, str] = {}
    basmala = ""
    for line in _tanzil_lines():
        parts = line.strip().split("|")
        if len(parts) != 3:
            continue
        s, a, text = int(parts[0]), int(parts[1]), parts[2]
        if s == 1 and a == 1:
            basmala = text
        if s == number:
            ayat[a] = text
    split = {a: split_words(text) for a, text in ayat.items()}
    head = split_words(basmala)
    # Compared by letters: Tanzil writes 95's and 97's Basmala «بِّسْمِ».
    if number not in (1, 9) and _letters("".join(split[1][: len(head)])) == _letters("".join(head)):
        split[0], split[1] = split[1][: len(head)], split[1][len(head) :]
    words = []
    for a in sorted(split):
        tokens = split[a]
        words += [Word(t, a, i == len(tokens) - 1, i + 1) for i, t in enumerate(tokens)]
    return Sura(number, Aya(number, 1).get().sura_name, words)


def passage(sura: int, ayah_from: int = 1, ayah_to: int | None = None) -> Sura:
    """Ayat `ayah_from`..`ayah_to` of a sura; the basmala is included when starting at ayah 1."""
    if not 1 <= sura <= 114:
        raise ValueError("sura must be 1–114")
    full = load_sura(sura)
    last = full.words[-1].ayah
    ayah_to = last if ayah_to is None else ayah_to
    if not 1 <= ayah_from <= ayah_to <= last:
        raise ValueError(f"sura {sura} has ayat 1–{last}")
    first = 0 if ayah_from == 1 else ayah_from
    return Sura(sura, full.name, [w for w in full.words if first <= w.ayah <= ayah_to])


# --- Normalization ---------------------------------------------------------
# Uthmani and Whisper's simple script spell some words differently, so both
# are reduced to base letters + harakat before comparing.

FATHATAN, DAMMATAN, KASRATAN, FATHA, DAMMA, KASRA, SHADDA, SUKUN = (chr(c) for c in range(0x064B, 0x0653))
HARAKAT = {FATHATAN, DAMMATAN, KASRATAN, FATHA, DAMMA, KASRA, SHADDA}
MARK_ALIASES = {
    "ۡ": SUKUN,  # Uthmani sukun
    "ࣰ": FATHATAN,  # open tanween variants
    "ࣱ": DAMMATAN,
    "ࣲ": KASRATAN,
}
SILENT_LETTERS = {"ا", "ٱ"}  # long-vowel alef is spelled inconsistently between scripts
LETTER_ALIASES = {"ى": "ي"}
DAGGER_ALEF = "ٰ"
TATWEEL = "ـ"


@dataclass
class Token:
    letter: str
    marks: frozenset[str]


def tokenize(word: str) -> list[Token]:
    tokens: list[Token] = []
    current: list | None = None  # [letter, set(marks)] of the last kept letter
    for i, ch in enumerate(word):
        ch = MARK_ALIASES.get(ch, ch)
        if ch in HARAKAT or ch == SUKUN:
            if current is not None and ch != SUKUN:  # sukun == no vowel
                current[1].add(ch)
            continue
        if not ("ء" <= ch <= "ي" or ch == "ٱ") or ch == TATWEEL:
            continue  # tatweel, dagger alef, small Uthmani marks, waqf signs
        nxt = word[i + 1] if i + 1 < len(word) else ""
        if ch in SILENT_LETTERS or (ch == "و" and nxt == DAGGER_ALEF):  # الصلوٰة → الصلاة
            current = None
            continue
        current = [LETTER_ALIASES.get(ch, ch), set()]
        tokens.append(current)
    out = [Token(letter, frozenset(marks)) for letter, marks in tokens]
    if out:  # a word-initial shadda comes from idgham with the previous word
        out[0] = Token(out[0].letter, out[0].marks - {SHADDA})
    return out


# --- Scoring ---------------------------------------------------------------

OK, WRONG, MISSED, PENDING, CURRENT = "ok", "wrong", "missed", "pending", "current"


@dataclass
class WordScore:
    status: str
    heard: str = ""
    letters_ok: bool = False
    harakat_ok: bool = False


@dataclass
class Alignment:
    scores: list[WordScore]
    frontier: int  # last reference word that was (at least partly) heard, -1 if none
    hyp_letter_to_ref_word: list[int | None] = field(default_factory=list)


def _best_window(ref: str, hyp: str, starts: list[int]) -> tuple[int, int]:
    """The reference span (from one of `starts`) that the transcript matches best.

    Aligning against the whole sura lets a partial reading match a later repeat of the
    same words (e.g. the last أحد) and mark everything in between as missed.
    """
    if not hyp:
        return 0, 0
    candidates = []
    for s in starts:
        hi = min(len(ref), s + int(len(hyp) * 1.5) + 10)
        candidates += [(Levenshtein.distance(ref[s:e], hyp), e - s, s, e) for e in range(s, hi + 1)]
    _, _, start, end = min(candidates)
    return start, end


def _align_letters(ref: str, hyp: str, ref_owner: list[int], hyp_owner: list[int]) -> list[tuple[int | None, int | None]]:
    """Edit-distance alignment as (ref index, hyp index) pairs, None on the skipped side.

    Among equally cheap alignments it prefers gaps that start and end on word boundaries,
    so a skipped word shows up as missed instead of stealing letters from its neighbours.
    """
    eps = 1e-3  # tie-breaker only; far below one edit
    inf = float("inf")
    n, m = len(ref), len(hyp)
    ref_start = [i == 0 or ref_owner[i] != ref_owner[i - 1] for i in range(n)]
    hyp_start = [j == 0 or hyp_owner[j] != hyp_owner[j - 1] for j in range(m)]
    # States: 0 = letters paired, 1 = ref letter skipped, 2 = extra hyp letter.
    cost = [[[inf] * 3 for _ in range(m + 1)] for _ in range(n + 1)]
    back = [[[0] * 3 for _ in range(m + 1)] for _ in range(n + 1)]
    cost[0][0][0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            if i and j:
                prev = cost[i - 1][j - 1]
                opts = (prev[0], prev[1] + (0 if ref_start[i - 1] else eps), prev[2] + (0 if hyp_start[j - 1] else eps))
                s = min(range(3), key=opts.__getitem__)
                cost[i][j][0] = opts[s] + (ref[i - 1] != hyp[j - 1])
                back[i][j][0] = s
            if i:
                prev, pen = cost[i - 1][j], 0 if ref_start[i - 1] else eps
                opts = (prev[0] + pen, prev[1], prev[2] + pen)
                s = min(range(3), key=opts.__getitem__)
                cost[i][j][1] = opts[s] + 1
                back[i][j][1] = s
            if j:
                prev, pen = cost[i][j - 1], 0 if hyp_start[j - 1] else eps
                opts = (prev[0] + pen, prev[1] + pen, prev[2])
                s = min(range(3), key=opts.__getitem__)
                cost[i][j][2] = opts[s] + 1
                back[i][j][2] = s

    pairs = []
    i, j = n, m
    state = min(range(3), key=cost[n][m].__getitem__)
    while i or j:
        prev_state = back[i][j][state]
        if state == 0:
            i, j = i - 1, j - 1
            pairs.append((i, j))
        elif state == 1:
            i -= 1
            pairs.append((i, None))
        else:
            j -= 1
            pairs.append((None, j))
        state = prev_state
    return pairs[::-1]


def align(sura: Sura, hyp_text: str, *, check_harakat: bool = True, live: bool = False) -> Alignment:
    ref_tokens: list[Token] = []
    ref_owner: list[int] = []
    last_token_of_word: dict[int, int] = {}
    for wi, w in enumerate(sura.words):
        toks = tokenize(w.text)
        ref_tokens += toks
        ref_owner += [wi] * len(toks)
        if toks:
            last_token_of_word[wi] = len(ref_tokens) - 1

    hyp_tokens: list[Token] = []
    hyp_owner: list[int] = []
    hyp_words = hyp_text.split()
    for hi, w in enumerate(hyp_words):
        toks = tokenize(w)
        hyp_tokens += toks
        hyp_owner += [hi] * len(toks)

    # Waqf: the last letter of an ayah is read with sukun, so its haraka isn't checked.
    waqf_tokens = {last_token_of_word[wi] for wi, w in enumerate(sura.words) if w.ayah_end and wi in last_token_of_word}

    n_words = len(sura.words)
    matched = [0] * n_words
    bad = [False] * n_words
    harakat_bad = [False] * n_words
    heard: list[set[int]] = [set() for _ in range(n_words)]
    hyp_to_ref: list[int | None] = [None] * len(hyp_tokens)

    ref_letters = "".join(t.letter for t in ref_tokens)
    hyp_letters = "".join(t.letter for t in hyp_tokens)
    # The basmala is optional, so the reading may start at it or at ayah 1.
    starts = sorted({0, next((i for i, wi in enumerate(ref_owner) if sura.words[wi].ayah > 0), 0)})
    start, end = _best_window(ref_letters, hyp_letters, starts)
    if 0 < end < len(ref_owner) and ref_owner[end] == ref_owner[end - 1]:
        bad[ref_owner[end]] = True  # transcript stops mid-word
    last_ri = start - 1
    for ri, hi in _align_letters(ref_letters[start:end], hyp_letters, ref_owner[start:end], hyp_owner):
        if ri is not None:
            ri += start  # back to sura-level letter positions
            last_ri = ri
            wi = ref_owner[ri]
            if hi is None:
                bad[wi] = True  # letter not read
                continue
            heard[wi].add(hyp_owner[hi])
            hyp_to_ref[hi] = wi
            if ref_letters[ri] != hyp_letters[hi]:
                bad[wi] = True
                continue
            matched[wi] += 1
            if ri not in waqf_tokens and ref_tokens[ri].marks != hyp_tokens[hi].marks:
                harakat_bad[wi] = True
        elif start <= last_ri < end - 1 and ref_owner[last_ri] == ref_owner[last_ri + 1]:
            wi = ref_owner[last_ri]  # extra letter inside a word
            bad[wi] = True
            heard[wi].add(hyp_owner[hi])
            hyp_to_ref[hi] = wi

    frontier = max((wi for wi in range(n_words) if matched[wi]), default=-1)
    scores = []
    for wi, w in enumerate(sura.words):
        letters_ok = matched[wi] > 0 and not bad[wi]
        harakat_ok = letters_ok and not harakat_bad[wi]
        text = " ".join(hyp_words[h] for h in sorted(heard[wi]))
        if wi > frontier:
            status = PENDING
        elif letters_ok and (harakat_ok or not check_harakat):
            status = OK
        elif live and wi == frontier:
            status = CURRENT  # word may still be in progress
        elif w.ayah == 0 and not matched[wi]:
            status = PENDING  # basmala is optional
        else:
            status = WRONG if matched[wi] else MISSED
        scores.append(WordScore(status, text, letters_ok, harakat_ok))
    return Alignment(scores, frontier, hyp_to_ref)
