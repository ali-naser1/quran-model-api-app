# Quran Speech Lab

Local Quran recitation testing: transcription (Whisper / Azure) and a live reader that
checks words, harakat and tajweed.

## Run

```powershell
uv sync          # first time only (installs CUDA 12.8 PyTorch)
.\run.ps1        # opens http://localhost:8501
```

Load models from the sidebar first (first load downloads them).

**Server without a GPU:** see [DEPLOY.md](DEPLOY.md) (Docker + HTTPS).

## Tabs

| Tab | What it does |
|---|---|
| **Transcribe** | Record or upload audio, transcribe with the chosen model |
| **Read Quran** | Surah Al-Ikhlas in Uthmani script. Press start and recite: words turn green/red live; after stop, Muaalem checks tajweed |

## Models

| Model | Used for | VRAM (fp16/bf16) |
|---|---|---|
| Whisper Base (Quran) | Transcribe, live reading | ~0.15 GB |
| Whisper Tiny (Quran) | Transcribe, live reading | ~0.08 GB |
| Muaalem (tajweed) | Tajweed check after stop | ~1.1 GB |
| Azure Speech (ar-SA) | Transcribe (cloud) | — |

All local models together use ~1.4 GB, so they fit side by side on a 16 GB GPU.
Load / unload them in the sidebar.

## How the reader checks a recitation

1. **Live** — the mic streams to Python every 0.7 s; Whisper re-transcribes the whole take and it's
   aligned letter-by-letter with the Uthmani text (both normalized to base letters + harakat).
2. **Harakat** — compared per letter; the last letter of each ayah is skipped (waqf).
3. **Tajweed** — after stop, audio is split at pauses, each piece is phonetized as its own
   waqf/ibtida segment and checked by Muaalem (madd, ghunna, qalqala, sifat …).
   Hover a word to see what was heard / the tajweed note.

## Files

| File | Role |
|---|---|
| `app.py` | Streamlit UI (sidebar model manager, two tabs) |
| `reader_component.py` | Mushaf view + live mic (custom component v2) |
| `quran.py` | Tanzil text, normalization, alignment & scoring |
| `tajweed.py` | Muaalem wrapper, pause splitting, error → word mapping |
| `transcriber.py` | Whisper and Azure engines, audio decoding |
| `models.py` | Load / unload + VRAM / RAM tracking |
| `Dockerfile`, `docker-compose.yml`, `Caddyfile` | CPU server deploy (see DEPLOY.md) |

## Notes

- Text: [Tanzil.net](https://tanzil.net) Uthmani, downloaded once to `data/`.
- Font: Amiri Quran (Google Fonts); ayah markers use U+06DD.
- Azure needs `AZURE_SPEECH_KEY` and `AZURE_SPEECH_ENDPOINT` in `.env`.
- Tarteel checkpoints have an outdated generation config; it's replaced with the
  original `openai/whisper-*` one so `language="arabic"` works.
- `samples/` holds Alafasy recitations from everyayah.com for local testing.
