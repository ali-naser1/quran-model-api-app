# Quran Speech API

Two HTTP APIs running on this PC's GPU, reached from the VPS over Tailscale:

| API | Endpoint | Model |
|---|---|---|
| **Whisper** | `POST /whisper/transcribe` | `tarteel-ai/whisper-base-ar-quran` |
| **Tajweed** | `POST /tajweed/check` | Whisper + `obadx/muaalem-model-v3_2` |

Plus a Streamlit app to test both by hand.

## Run (this PC)

```powershell
uv sync                 # first time only (CUDA 12.8 PyTorch)
.\run_api.ps1           # API on :8000, docs at http://127.0.0.1:8000/docs
.\run.ps1               # optional: Streamlit test client on :8501
```

Both models load at start-up (~10 s, ~1.3 GB VRAM) and stay loaded.

## Reach it from the VPS (Tailscale)

1. On this PC: start Tailscale (`tailscale up`). Its Tailscale IP is `100.81.142.105`.
2. On the VPS: install Tailscale and `tailscale up` with the same account.
3. Allow the port on this PC, Tailscale addresses only (admin PowerShell):
   ```powershell
   New-NetFirewallRule -DisplayName "Quran API (Tailscale)" -Direction Inbound -Protocol TCP -LocalPort 8000 -RemoteAddress 100.64.0.0/10 -Action Allow
   ```
4. From the VPS: `curl http://100.81.142.105:8000/health`

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
    { "index": 4, "text": "قُلْ", "ayah": 1, "ayah_end": false,
      "status": "ok", "heard": "قُلْ", "notes": [] }
  ],
  "audio_seconds": 16.1, "whisper_seconds": 0.6, "tajweed_seconds": 1.0
}
```

- `status`: `ok` · `wrong` · `missed` · `pending` (not reached) · `current` (live only)
- `notes`: Muaalem findings for that word — `kind` (`tajweed` / `tashkeel` / `normal`), Arabic `detail`
  (e.g. `تجويد · إبدال · المد الطبيعي · المطلوب 2 حركات، المقروء 1`), `expected` / `heard` phonemes

### `GET /tajweed/passage?sura=112&ayah_from=1&ayah_to=4`

The Uthmani words (Tanzil) with the indices `/check` uses — for rendering the text.

### Example (Python, from the VPS)

```python
import requests

r = requests.post(
    "http://100.81.142.105:8000/tajweed/check",
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

Speed on the RTX 5060 Ti: Ikhlas (16 s) → words-only 0.2 s, full check ~1.9 s.

## Files

| File | Role |
|---|---|
| `api.py` | FastAPI app (both APIs) |
| `transcriber.py` | Whisper + audio decoding (PyAV, no ffmpeg) |
| `tajweed.py` | Muaalem: pause splitting, error → word mapping |
| `quran.py` | Tanzil text, normalization, alignment & scoring |
| `app.py`, `reader_component.py` | Streamlit test client (live Mushaf reader) |

Text: [Tanzil.net](https://tanzil.net) Uthmani (downloaded once to `data/`). `samples/` holds
Alafasy recitations from everyayah.com for testing.
