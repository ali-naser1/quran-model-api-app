# Deploy on a CPU server

Runs with Docker: the app (CPU-only PyTorch) behind Caddy, which handles HTTPS.
HTTPS is required — browsers only allow the microphone on `https://`.

## Server

| | Minimum |
|---|---|
| CPU / RAM | 4 vCPU · 8 GB |
| Disk | 20 GB (image 2.6 GB + models 1.5 GB) |
| OS | Any Linux with Docker Engine + Compose v2.24+ |
| Network | Ports 80 and 443 open |
| DNS | A domain (or subdomain) with an A record → server IP |

## First deploy

```bash
curl -fsSL https://get.docker.com | sh            # installs Docker + compose plugin
git clone <repo-url> quran-speech-lab && cd quran-speech-lab
cp .env.example .env && nano .env                 # set DOMAIN (+ Azure keys if wanted)
docker compose up -d --build
docker compose logs -f app                        # Ctrl+C to stop watching
```

Open `https://<DOMAIN>`. Caddy gets the certificate automatically on first visit.

**First start downloads ~1.5 GB of models** (a few minutes). The sidebar shows
`◌ loading…` until they're ready; they're kept in the `models` volume after that.

## Update

```bash
git pull && docker compose up -d --build
```

## Settings (`.env`)

| Variable | Default | Meaning |
|---|---|---|
| `DOMAIN` | — (required) | Domain pointing at the server |
| `PRELOAD_MODELS` | `Whisper Base (Quran),Muaalem (tajweed)` | Loaded at start-up |
| `ALLOW_MODEL_CONTROL` | `false` | Show Load / Unload buttons to visitors |
| `AZURE_SPEECH_KEY` / `AZURE_SPEECH_ENDPOINT` | empty | Enables the Azure model |

Whisper Tiny is faster on CPU but drops whole ayat, so Base is the default.

## Measured (Docker, 4 CPU / 8 GB limit, no GPU)

| Step | Time |
|---|---|
| Live update while reading | 0.4–0.7 s |
| Stop → final result + tajweed | ~7 s |
| RAM with default models | ~3 GB |

Models are shared by all visitors and run one request at a time, so simultaneous
users wait for each other.

## Useful commands

```bash
docker compose ps                  # status (app should be "healthy")
docker compose restart app         # restart the app only
docker compose down                # stop everything (models volume is kept)
```
