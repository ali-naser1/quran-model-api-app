# Quran Speech API

Two HTTP APIs for checking Quran recitation:

| API | Endpoint | Model |
|---|---|---|
| **Whisper** | `POST /whisper/transcribe` | `tarteel-ai/whisper-base-ar-quran` |
| **Tajweed** | `POST /tajweed/check` | Whisper + `obadx/muaalem-model-v3_2` |

Plus a Streamlit app to test both by hand.

## Run

| Where | How |
|---|---|
| **VPS** (Docker, CPU) | See [DEPLOY.md](DEPLOY.md) |
| **This PC** (GPU) | `uv sync` once, then `.\run_api.ps1` (API :8000) and `.\run.ps1` (tester :8501) |

Both models load at start-up and stay loaded (~1.3 GB VRAM on GPU, ~3 GB RAM on CPU).
Docs: `http://127.0.0.1:8000/docs`.

Every call except `/health` needs the `X-API-Key` header; the key is `API_KEY` in `.env`.

## API

### `POST /whisper/transcribe`

Multipart `file` (wav, mp3, ogg, webm, m4a…). Returns:

```json
{ "text": "قُلْ هُوَ اللَّهُ أَحَدٌ", "audio_seconds": 2.9, "infer_seconds": 0.16 }
```

### `POST /tajweed/check`

Multipart `file` + query parameters:

| Param | Default | |
|---|---|---|
| `sura` | `112` | 1–114 |
| `ayah_from`, `ayah_to` | `1`, last | Passage to check (basmala included when starting at 1) |
| `tajweed` | `true` | `false` = words + harakat only (~0.2 s, for live use) |
| `live` | `false` | Recording in progress: last word reached is `current`, not wrong |
| `check_harakat` | `true` | Compare diacritics per letter |
| `madd_monfasel_len` / `madd_mottasel_len` / `madd_aared_len` | `4` | Hafs madd lengths |

Returns one entry per word, in the same order as `GET /tajweed/passage`:

```json
{
  "sura": 112, "name": "الإخلاص", "transcript": "قُلْ هُوَ …",
  "correct": 15, "total": 15, "tajweed_checked": true, "skipped": [],
  "words": [
    { "index": 4, "text": "قُلْ", "ayah": 1, "position": 1, "ayah_end": false,
      "status": "ok", "heard": "قُلْ", "notes": [] }
  ],
  "audio_seconds": 16.1, "whisper_seconds": 0.6, "tajweed_seconds": 1.0
}
```

- `status`: `ok` · `wrong` · `missed` · `pending` (not reached) · `current` (live only)
- `ayah` + `position` (from 1 in each ayah) number the words exactly as Abjad's `quran_words`:
  same Tanzil text, and a pause mark / ۞ / ۩ is part of its word (`"رَيْبَ ۛ"`), never a word.
  `ayah: 0` is the Basmala, added when the passage starts at ayah 1 (reading it is optional).
- `notes`: Muaalem findings for that word — `kind` (`tajweed` / `tashkeel` / `normal`), Arabic `detail`.
  A `wrong` word keeps all of them; an `ok` word only its `tajweed` ones (a letter or haraka note
  there contradicts Whisper, and those were the false alarms)
  (e.g. `تجويد · إبدال · المد الطبيعي · المطلوب 2 حركات، المقروء 1`), `expected` / `heard` phonemes

### `GET /tajweed/passage?sura=112&ayah_from=1&ayah_to=4`

The Uthmani words (Tanzil) with the indices `/check` uses — for rendering the text.

### Example (Python, from the app server on the LAN)

```python
import requests

r = requests.post(
    "http://10.0.0.12:8000/tajweed/check",  # the VPS's LAN IP
    params={"sura": 112},
    headers={"X-API-Key": API_KEY},
    files={"file": open("recitation.webm", "rb")},
    timeout=60,
)
for w in r.json()["words"]:
    print(w["text"], w["status"], [n["detail"] for n in w["notes"]])
```

## How the check works

1. Whisper transcribes the audio; it's aligned letter-by-letter with the Uthmani text
   (both reduced to base letters + harakat). The last letter of each ayah isn't vowel-checked (waqf).
2. With `tajweed=true`, the audio is split at pauses, each piece is phonetized as its own
   waqf/ibtida segment and checked by Muaalem (madd, ghunna, qalqala, letter sifat…).

Speed for Ikhlas (16 s): words-only 0.2 s GPU / ~0.6 s CPU · full check ~1.9 s GPU / ~5 s CPU.

## Files

| File | Role |
|---|---|
| `api.py` | FastAPI app (both APIs) |
| `transcriber.py` | Whisper + audio decoding (PyAV, no ffmpeg) |
| `tajweed.py` | Muaalem: pause splitting, error → word mapping |
| `quran.py` | Tanzil text, normalization, alignment & scoring |
| `app.py`, `reader_component.py` | Streamlit test client (live Mushaf reader) |
| `Dockerfile`, `docker-compose*.yml` | VPS deploy (see DEPLOY.md) |

Text: [Tanzil.net](https://tanzil.net) Uthmani, the same download as Abjad's Quran dataset
(downloaded once to `data/`, baked into the Docker image). `samples/` holds
Alafasy recitations from everyayah.com for testing.
