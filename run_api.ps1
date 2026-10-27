$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
uv run uvicorn api:app --host 127.0.0.1 --port 8000 @args
