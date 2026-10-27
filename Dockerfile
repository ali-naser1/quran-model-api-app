# CPU-only image: no GPU needed on the server.
FROM python:3.12-slim-bookworm

# Azure Speech SDK needs OpenSSL, ALSA and CA certificates at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libasound2 libssl3 curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.10.12 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/models \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies first so code changes don't reinstall PyTorch.
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups --group cpu

COPY . .
# Bake the Tanzil text into the image so start-up doesn't depend on tanzil.net.
RUN python -c "from quran import load_sura; load_sura(112)"

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD curl -fs http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
