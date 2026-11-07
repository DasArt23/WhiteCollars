FROM ghcr.io/astral-sh/uv:0.9.26 AS uv
FROM python:3.14-slim-bookworm

COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# Lock сохраняет привязку torch к отдельному индексу CUDA 12.6.
COPY pyproject.toml uv.lock README.md requirements.txt ./
RUN uv sync --frozen --extra v100 --no-dev --no-install-project --python /usr/local/bin/python
COPY src ./src
RUN uv sync --frozen --extra v100 --no-dev --python /usr/local/bin/python
COPY checkpoints ./checkpoints

ENV PATH="/app/.venv/bin:$PATH" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
ENTRYPOINT ["project", "predict", "--checkpoint", "/app/checkpoints/best_model.pt"]
CMD ["--help"]
