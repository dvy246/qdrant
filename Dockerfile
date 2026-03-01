# ---------- Build stage ----------
FROM python:3.11-slim AS builder

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src/ src/

RUN pip install --no-cache-dir ".[api]"

FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app

# Pre-download the ChemBERTa model at build time so startup is fast
RUN python -c "from transformers import AutoTokenizer, AutoModel; \
    AutoTokenizer.from_pretrained('seyonec/ChemBERTa-zinc-base-v1'); \
    AutoModel.from_pretrained('seyonec/ChemBERTa-zinc-base-v1')"

# Run as non-root in runtime image
RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000

CMD ["uvicorn", "molsearch.api_server:app", "--host", "0.0.0.0", "--port", "8000"]
