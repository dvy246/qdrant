"""
FastAPI REST API for molecular similarity search.

Provides a /search endpoint that accepts a SMILES string and returns
the top-k most similar molecules from the Qdrant index.

Usage:
    uvicorn molsearch.api_server:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import asyncio
import logging
import math
from contextlib import asynccontextmanager
from functools import partial

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from rdkit import Chem

from molsearch.config import MAX_SMILES_LENGTH, SAMPLE_SMILES
from molsearch.embedder import MoleculeEmbedder
from molsearch.molecule_processor import process_smiles_batch
from molsearch.qdrant_indexer import (
    check_system_health,
    collection_exists_and_populated,
    create_collection,
    create_payload_indexes,
    get_qdrant_client,
    search_similar_molecules,
    upsert_molecules,
)

logger = logging.getLogger(__name__)

# Module-level references populated during lifespan
_embedder: MoleculeEmbedder | None = None
_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize embedder and Qdrant on startup, populate demo index."""
    global _embedder, _client

    try:
        logger.info("Loading ChemBERTa model...")
        _embedder = MoleculeEmbedder()
    except Exception as exc:
        logger.error("Failed to load embedder: %s", exc)
        _embedder = None

    try:
        logger.info("Initializing Qdrant...")
        _client = get_qdrant_client()

        # Ensure collection + payload indexes exist on startup.
        create_collection(_client)
        create_payload_indexes(_client)

        # Only populate the demo index if the collection doesn't already have data.
        if not collection_exists_and_populated(_client):
            if _embedder is not None:
                molecules = process_smiles_batch(SAMPLE_SMILES)
                smiles_list = [m["smiles"] for m in molecules]
                embeddings = _embedder.embed(smiles_list)
                upsert_molecules(_client, molecules, embeddings)
                logger.info("Indexed %d demo molecules.", len(molecules))
            else:
                logger.warning("Embedder not available; skipping demo indexing.")
        else:
            logger.info("Collection already populated — skipping demo indexing.")
    except Exception as exc:
        logger.error("Failed to initialize Qdrant: %s", exc)
        _client = None

    yield

    _client = None
    _embedder = None


app = FastAPI(
    title="Molecule Similarity Search API",
    description="Search for similar molecules using ChemBERTa embeddings and Qdrant",
    version="1.0.0",
    lifespan=lifespan,
)


# ---- Request / Response models ----


class SearchRequest(BaseModel):
    """Request body for the /search endpoint."""

    smiles: str = Field(
        ...,
        min_length=1,
        max_length=MAX_SMILES_LENGTH,
        description="SMILES string of the query molecule",
    )
    top_k: int = Field(default=5, ge=1, le=100, description="Number of results")
    mw_max: float | None = Field(
        default=None,
        ge=0,
        allow_inf_nan=False,
        description="Max molecular weight filter",
    )
    logp_max: float | None = Field(
        default=None,
        allow_inf_nan=False,
        description="Max LogP filter",
    )
    toxicity_max: float | None = Field(
        default=None,
        ge=0,
        allow_inf_nan=False,
        description="Max toxicity score filter",
    )


class MoleculeHit(BaseModel):
    """A single search result."""

    smiles: str
    score: float
    molecular_weight: float
    logp: float
    toxicity_score: float | None = None
    tanimoto_score: float | None = None
    fused_score: float | None = None


class SearchResponse(BaseModel):
    """Response body for the /search endpoint."""

    query_smiles: str
    canonical_smiles: str
    results: list[MoleculeHit]


# ---- Endpoints ----


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Search for molecules similar to the query SMILES.

    Optionally filter results by molecular weight, LogP, and toxicity score.
    Uses run_in_executor to avoid blocking the event loop during
    CPU-bound embedding inference.
    """
    if _embedder is None or _client is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    query_smiles = request.smiles.strip()
    if not query_smiles:
        raise HTTPException(status_code=400, detail="SMILES must not be empty")

    numeric_filters = {
        "mw_max": request.mw_max,
        "logp_max": request.logp_max,
        "toxicity_max": request.toxicity_max,
    }
    for key, value in numeric_filters.items():
        if value is not None and not math.isfinite(value):
            raise HTTPException(status_code=422, detail=f"{key} must be a finite number")

    mol = Chem.MolFromSmiles(query_smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid SMILES: {request.smiles}",
        )

    canonical = Chem.MolToSmiles(mol)

    # Run the CPU-bound search in a thread pool to avoid blocking async
    loop = asyncio.get_running_loop()
    try:
        hits = await loop.run_in_executor(
            None,
            partial(
                search_similar_molecules,
                query_smiles=canonical,
                embedder=_embedder,
                client=_client,
                top_k=request.top_k,
                mw_max=request.mw_max,
                logp_max=request.logp_max,
                toxicity_max=request.toxicity_max,
            ),
        )
    except RuntimeError as exc:
        logger.error("Search failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}") from exc

    return SearchResponse(
        query_smiles=request.smiles,
        canonical_smiles=canonical,
        results=[MoleculeHit(**h) for h in hits],
    )


@app.get("/health")
def health():
    """Health check endpoint — verifies model and Qdrant are initialized."""
    health_status = check_system_health(_embedder, _client)
    status_code = 200 if health_status["status"] == "ok" else 503
    return health_status
