FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.10.12 /uv /bin/

# cpu (default) or cu128 for an NVIDIA GPU server (see docker-compose.gpu.yml).
ARG TORCH=cpu

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/models \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies first so code changes don't reinstall PyTorch.
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups --group ${TORCH}

COPY *.py ./
# Bake the Tanzil text into the image so start-up doesn't depend on tanzil.net.
RUN python -c "from quran import load_sura; load_sura(1)"

EXPOSE 8000 8501
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
