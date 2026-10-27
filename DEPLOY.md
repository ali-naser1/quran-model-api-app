# Deploy on the VPS

Docker runs two containers from one image:

| Container | Port (VPS only) | What |
|---|---|---|
| `api` | `127.0.0.1:8000` | The Whisper + tajweed APIs |
| `tester` | `127.0.0.1:8501` | Streamlit test client |

Neither is public: your project on the VPS calls the API locally, and you open the tester
through an SSH tunnel (the mic works there because the browser sees `localhost`).

## VPS requirements

| | Minimum |
|---|---|
| CPU / RAM | 4 vCPU · 8 GB (no GPU needed) |
| Disk | 20 GB (image ~2.6 GB + models ~1.5 GB) |
| OS | Linux with Docker Engine + Compose v2.24+ |

## 1. Install Docker (once)

```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER   # then log out and back in
```

## 2. Give the VPS read access to the private repo (once)

```bash
ssh-keygen -t ed25519 -N "" -f ~/.ssh/github_quran -C "vps deploy key"
printf 'Host github.com\n  IdentityFile ~/.ssh/github_quran\n' >> ~/.ssh/config
cat ~/.ssh/github_quran.pub
```

Add the printed key as a **read-only deploy key**: GitHub → repo → Settings → Deploy keys → Add,
or from the PC: `gh repo deploy-key add github_quran.pub --repo ali-naser1/quran-transcriber --title vps`.

## 3. Clone, configure, start

```bash
git clone git@github.com:ali-naser1/quran-transcriber.git && cd quran-transcriber
cp .env.example .env
python3 -c "import secrets; print('API_KEY=' + secrets.token_hex(32))" >> .env
docker compose up -d --build
docker compose logs -f api      # first start downloads ~1.5 GB of models; Ctrl+C to stop watching
```

Ready when `docker compose ps` shows `api` as **healthy** (a few minutes the first time).

```bash
curl http://127.0.0.1:8000/health
```

## Update

```bash
cd quran-transcriber && git pull && docker compose up -d --build
```

## Open the tester from your PC

```bash
ssh -L 8501:127.0.0.1:8501 -L 8000:127.0.0.1:8000 user@your-vps
```

Then browse to http://localhost:8501 (tester) and http://localhost:8000/docs (API docs).

## Call the API from your project

- **Project runs directly on the VPS:** `http://127.0.0.1:8000`, header `X-API-Key: <API_KEY from .env>`.
- **Project runs in Docker:** join this stack's network and use `http://api:8000`:
  ```yaml
  # in your project's docker-compose.yml
  services:
    web:
      networks: [default, quran]
  networks:
    quran:
      external: true
      name: quran-transcriber_default
  ```

## Measured (Docker, CPU only)

| Call | Ikhlas (16 s audio) |
|---|---|
| `/whisper/transcribe` | ~0.9 s |
| `/tajweed/check?tajweed=false` (live) | 0.4–0.7 s |
| `/tajweed/check` (full, with Muaalem) | ~5 s |
| RAM | ~3 GB |

Requests run one at a time per model, so simultaneous users wait for each other.

## GPU server instead

With an NVIDIA GPU + NVIDIA Container Toolkit: add `TORCH=cu128` to `.env`, then
`docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build`.

## Useful commands

```bash
docker compose ps                 # status
docker compose restart api        # restart the API
docker compose down               # stop (models volume is kept)
```
