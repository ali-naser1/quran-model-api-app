# Serves the APIs on all interfaces so the VPS can reach them over Tailscale.
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
uv run uvicorn api:app --host 0.0.0.0 --port 8000 @args
