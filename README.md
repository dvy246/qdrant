# Molecule Vector Search with Qdrant

## The Problem: Finding the Right Needles in the Chemical Haystack

In computational chemistry and early-stage drug discovery, a common first step is finding molecular analogs. When you identify a promising hit compound, the immediate next questions are usually: *"What else in our existing library shares this scaffold?"* or *"Which known drugs have a similar structure?"*

Historically, this is solved by generating traditional structural fingerprints (like ECFP4) and comparing them using Tanimoto similarity. While these methods are fast and excellent for explicit substructure matching, they have two major limitations:

1. **They struggle with "scaffold hopping."** Traditional methods are very literal. They might find molecules with the exact same atom-level structure but will completely miss molecules that look visibly different on paper yet behave similarly in a biological context.
2. **They don't scale well.** Running brute-force O(N) mathematical comparisons against modern chemical libraries containing tens of millions of compounds quickly becomes a severe computational bottleneck.

## The Solution: Semantic Search for Chemistry

This project provides a modern, machine-learning-driven alternative to traditional fingerprinting. It translates molecules into dense vector representations (embeddings) using a chemical-aware transformer model called ChemBERTa. These vectors are then indexed into a specialized, high-performance vector database named Qdrant.

By representing molecules as 768-dimensional vectors, the system learns and captures deep semantic and topological structural patterns that traditional 1D string matching or 2D fingerprints miss. Qdrant provides **approximate nearest neighbor (ANN) retrieval** with latency that scales depending on collection size, hardware, and HNSW index parameters (`m`, `ef`), while simultaneously filtering by medicinal chemistry rules like Molecular Weight, LogP, and Toxicity natively in the database.

This pipeline isn't just about faster software, it's about empowering chemists to find better, non-obvious analogs.

---

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

- **SMILES Validation:** Ensures clean data ingestion by validating and canonicalizing inputs natively with RDKit.
- **Deep Embeddings:** Encodes deep structural patterns via 768-dimensional ChemBERTa vectors.
- **High-Speed Retrieval:** Qdrant similarity search uses cosine distance and deterministic UUIDs to generate sub-second similarity rankings.
- **Payload Filtering:** Execute metadata-filtered searches concurrently, directly querying by molecular weight, LogP, and toxicity score constraints.
- **Production APIs:** Ready-to-use FastAPI endpoint for headless integration into other services.
- **Interactive UI:** A Streamlit interface for hands-on, visual exploration by medicinal chemists.
- **Robustness:** Includes offline-capable unit tests (no network required) and data-integrity checks for batch upserts.

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

### 5. Test and Verify the Integration (Recommended for Reviewers)

To ensure the environment connects to the models and that tests pass locally:

```bash
# 1. Run the test suite (validates expected functional behavior)
make test

# 2. Run the end-to-end inference/search pipeline directly in the terminal
make run-pipeline

# 3. Start the API locally
make run-api

# 4. In a separate terminal, test the API with cURL
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"smiles": "CC(=O)Oc1ccccc1C(=O)O", "top_k": 3}'

# 5. Stop the API and run the Web UI (Available at http://localhost:8501)
make run-ui
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
