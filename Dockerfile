# syntax=docker/dockerfile:1
#
# Includes `embed` by default. An earlier version left it out to keep the image
# small, on the reasoning that search "only needs an encoder for the query" — but
# that is the whole read path: without it `/v1/memories/search` returns 500 while
# `/healthz` still reports ok, which is the worst of both. Verified by building
# and running it.
#
# Build a slimmer read-only image with
#   --build-arg EXTRAS="api"
# and expect search to be unavailable.
#
# Nothing is baked in but code: no datasets, no models, no credentials, no SQLite
# state. The store arrives on a mounted volume, so the image stays reproducible and
# an accidental `docker push` cannot leak a user's memories.

# Multi-architecture index digest verified 2026-08-27. Keep the readable tag so
# update tooling knows what to refresh, and the digest so a rebuild cannot silently
# pick up a different base image.
FROM python:3.13-slim@sha256:7e3a6aca9d74f93cca21a91d86a8dad8c34749afd5b4a98ee481c9c47b9f5ed4 AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # Not decoration: the project supports Windows precisely because every file
    # I/O site declares an encoding, and UTF-8 mode keeps the container's
    # behaviour identical to the developer's machine.
    PYTHONUTF8=1

WORKDIR /app

# uv gives the same resolution the developer and CI use, rather than whatever pip
# happens to pick at build time.
COPY --from=ghcr.io/astral-sh/uv:latest@sha256:88bc6eb1ccd4b82efd0e1b530caffabddf50dc2bf612e66c14ea25b8ee8a4d3d /uv /usr/local/bin/uv

ARG EXTRAS="api,llm,embed"

# Dependency layer first: source changes must not re-resolve the whole tree.
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/
COPY configs/ ./configs/
# The ARG is expanded here rather than hard-coded, or overriding it silently does
# nothing — which is what it did.
#
# Then the CUDA stack is removed. `uv sync` resolves torch to the CUDA wheel, whose
# nvidia-* and triton dependencies are 3.5GB on their own; reinstalling torch from
# the CPU index swaps torch (577MB) but leaves those behind as orphans, because
# nothing asks uv to drop a dependency that is no longer required. Measured inside
# the image: nvidia/ 2.9GB, triton/ 649MB, torch/ 577MB.
#
# Nothing here can reach a GPU — `torch.cuda.is_available()` is False — and encoding
# one query on CPU is milliseconds at this scale, so all of it is dead weight. The
# removal shares this layer with the install; in a later one the files would still
# sit in the image underneath.
RUN set -eu; \
    uv sync --frozen --no-dev $(echo "$EXTRAS" | tr "," "\n" | sed "s/^/--extra /" | tr "\n" " "); \
    if echo "$EXTRAS" | grep -q embed; then \
        uv pip install --python /app/.venv/bin/python \
            --index-url https://download.pytorch.org/whl/cpu --reinstall torch; \
        rm -rf /app/.venv/lib/python*/site-packages/nvidia \
               /app/.venv/lib/python*/site-packages/triton \
               /app/.venv/lib/python*/site-packages/nvidia_* \
               /app/.venv/lib/python*/site-packages/triton-*; \
        /app/.venv/bin/python -c "import torch, sentence_transformers; assert not torch.cuda.is_available()"; \
    fi; \
    rm -rf /root/.cache/uv /tmp/*

# A non-root user, and the store directory it will own. The volume is mounted over
# this path at run time; creating it here means the container also works without
# one, with an empty store.
RUN useradd --create-home --uid 10001 memory \
    && mkdir -p /data/stores /data/results \
    && chown -R memory:memory /data /app
USER memory

ENV PATH="/app/.venv/bin:$PATH" \
    LLTM_STORE_DIR=/data/stores \
    LLTM_RESULTS_DIR=/data/results \
    LLTM_DATA_DIR=/data

EXPOSE 8000

# Hits the real readiness check: /healthz opens the store and counts rows, so a
# container with an unreadable volume reports unhealthy instead of accepting
# traffic it cannot serve.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "llm_long_term_memory.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
