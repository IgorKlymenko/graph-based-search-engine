FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies first — cached layer, only invalidated when the lock changes
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY src ./src
RUN uv sync --frozen --no-dev

RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Railway injects PORT; the app reads it in config.py
CMD ["uv", "run", "--no-dev", "python", "-m", "app.main"]
