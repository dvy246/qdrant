# Molecule Vector Search with Qdrant

A production-focused molecular similarity search pipeline using ChemBERTa embeddings and Qdrant vector search, with FastAPI and Streamlit interfaces.

## Architecture

```text
SMILES strings
    │
    ▼
RDKit validation and canonicalization
    │
    ▼
ChemBERTa embedding (seyonec/ChemBERTa-zinc-base-v1)
    │
    ▼
Qdrant vector indexing (with descriptor + optional toxicity payloads)
    │
    ▼
Similarity search API (FastAPI) or Web UI (Streamlit)
```

## Features

- SMILES validation and canonicalization with RDKit
- Dense molecular embeddings via ChemBERTa (768-dimensional)
- Optional toxicity metadata support (`toxicity_score`) in payloads and filters
- Qdrant similarity search with cosine distance and deterministic UUID IDs
- Filtered search by molecular weight, LogP, and toxicity score
- FastAPI endpoint for programmatic search
- Streamlit UI for interactive exploration
- Batch upsert with data-integrity checks
- Offline-capable unit tests (no required HuggingFace network access)

## Quick Start

### 1. Install

```bash
# Core
pip install -e .

# API + UI + dev tools
pip install -e ".[all]"
```

### 2. Run pipeline

```bash
molsearch
# or
python -m molsearch.pipeline
```

### 3. Run API

```bash
make run-api
```

### 4. Run Streamlit UI

```bash
make run-ui
```

### 5. Run checks

```bash
make test
make lint
mypy src
```

## API usage

### Search

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"smiles": "CC(=O)Oc1ccccc1C(=O)O", "top_k": 5}'
```

### Search with filters

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "smiles": "CC(=O)Oc1ccccc1C(=O)O",
    "top_k": 5,
    "mw_max": 500,
    "logp_max": 5,
    "toxicity_max": 0.3
  }'
```

### Health

```bash
curl http://localhost:8000/health
```

## Configuration

`src/molsearch/config.py` supports defaults plus environment overrides:

- `MOLSEARCH_MODEL_NAME`
- `MOLSEARCH_COLLECTION_NAME`
- `MOLSEARCH_VECTOR_DIM`
- `MOLSEARCH_BATCH_SIZE`
- `MOLSEARCH_QDRANT_HOST`
- `MOLSEARCH_QDRANT_PORT`
- `MOLSEARCH_PERSISTENT_QDRANT`
- `MOLSEARCH_UPSERT_BATCH_SIZE`
- `MOLSEARCH_MAX_SMILES_LENGTH`

## Docker

```bash
docker compose up -d
```

This starts:
- Qdrant on `6333`
- API on `8000`

## Notes

- `create_collection()` is intentionally non-destructive and reuses existing collections.
- Use `recreate_collection()` only when an explicit destructive reset is intended.
- For production datasets, provide assay-derived `toxicity_score` values during ingestion; the project does not fabricate toxicity labels.

## License

MIT
