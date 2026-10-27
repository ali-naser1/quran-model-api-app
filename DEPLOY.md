# Deploy on the VPS

Docker runs two containers from one image:

| Container | Listens on | What |
|---|---|---|
| `api` | `<VPS LAN IP>:8000` | The Whisper + tajweed APIs, for the app server on the same LAN |
| `tester` | `127.0.0.1:8501` | Streamlit test client |

The API is only on the LAN (with an API key); the tester is opened through an SSH tunnel
(the mic works there because the browser sees `localhost`).

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

## 2. Clone, configure, start

```bash
git clone https://github.com/ali-naser1/quran-model-api-app.git && cd quran-model-api-app
cp .env.example .env
python3 -c "import secrets; print('API_KEY=' + secrets.token_hex(32))" >> .env
ip -4 -brief addr                # find this VPS's LAN IP, e.g. 10.0.0.12
echo "API_BIND=10.0.0.12" >> .env   # ← use the IP from the line above
docker compose up -d --build
docker compose logs -f api      # first start downloads ~1.5 GB of models; Ctrl+C to stop watching
```

Ready when `docker compose ps` shows `api` as **healthy** (a few minutes the first time).
Then from the **app server**:

```bash
curl http://10.0.0.12:8000/health
```

If that times out, a firewall between the two servers is blocking TCP 8000.

## Update

```bash
cd quran-model-api-app && git pull && docker compose up -d --build
```

## Open the tester from your PC

```bash
ssh -L 8501:127.0.0.1:8501 user@your-vps
```

Then browse to http://localhost:8501. API docs: `http://<VPS LAN IP>:8000/docs` from any machine on the LAN.

## Call the API from the app server

Base URL `http://<VPS LAN IP>:8000`, header `X-API-Key: <API_KEY from the VPS .env>`
(store the key in the app server's own config, not in code).

```bash
curl -H "X-API-Key: $API_KEY" -F file=@recitation.webm "http://10.0.0.12:8000/tajweed/check?sura=112"
```

**Security notes**

- Traffic is plain HTTP inside the LAN; fine for a company network, but anyone on the LAN who can
  sniff traffic could read the key. Put a TLS proxy in front if that matters.
- Docker bypasses `ufw` for published ports, so binding to the LAN IP (not `0.0.0.0`) is what keeps
  the API off other interfaces. To allow **only** the app server (lost on reboot unless saved):
  ```bash
  sudo iptables -I DOCKER-USER -p tcp --dport 8000 ! -s <app-server-ip> -j DROP
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
