"""
Qdrant indexer: collection management, upsert, and similarity search.

Handles all interactions with the Qdrant vector database, including
collection lifecycle, batch point upsert, and filtered similarity search.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    PayloadSchemaType,
    PointStruct,
    Range,
    VectorParams,
)
from rdkit import Chem
from rdkit.Chem import DataStructs, rdFingerprintGenerator

from molsearch.config import (
    COLLECTION_NAME,
    QDRANT_HOST,
    QDRANT_PORT,
    UPSERT_BATCH_SIZE,
    USE_PERSISTENT_QDRANT,
    VECTOR_DIM,
)
from molsearch.embedder import MoleculeEmbedder

logger = logging.getLogger(__name__)


def check_system_health(
    embedder: MoleculeEmbedder | None,
    client: QdrantClient | None,
) -> dict[str, bool | str]:
    """Quick health check -- returns status dict with 'ok', 'degraded', or 'down'."""
    model_loaded = embedder is not None
    vector_store_reachable = False
    collection_exists_flag = False

    if client is not None:
        try:
            client.get_collections()
            vector_store_reachable = True
            collection_exists_flag = client.collection_exists(
                collection_name=COLLECTION_NAME
            )
        except Exception as exc:
            logger.warning("Health check: Qdrant not reachable: %s", exc)

    if model_loaded and vector_store_reachable and collection_exists_flag:
        status = "ok"
    elif model_loaded or vector_store_reachable:
        status = "degraded"
    else:
        status = "down"

    return {
        "model_loaded": model_loaded,
        "vector_store_reachable": vector_store_reachable,
        "collection_exists": collection_exists_flag,
        "status": status,
    }


def _smiles_to_uuid(smiles: str) -> str:
    """Deterministic UUID from canonical SMILES -- safe for incremental upserts."""
    namespace = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")
    return str(uuid.uuid5(namespace, smiles))


def _extract_vector_size(collection_info: Any) -> int | None:
    """Pull vector size from collection config, if available."""
    vectors = collection_info.config.params.vectors
    size = getattr(vectors, "size", None)
    if isinstance(size, int):
        return size

    if isinstance(vectors, dict) and vectors:
        first = next(iter(vectors.values()))
        named_size = getattr(first, "size", None)
        if isinstance(named_size, int):
            return named_size

    return None


def get_qdrant_client() -> QdrantClient:
    """
    Create a Qdrant client based on configuration.

    Returns an in-memory client for development or a client connected
    to a persistent Qdrant instance for production.

    Returns:
        A QdrantClient instance.

    Raises:
        RuntimeError: If connection to persistent Qdrant fails.
    """
    if USE_PERSISTENT_QDRANT:
        logger.info("Connecting to Qdrant at %s:%d", QDRANT_HOST, QDRANT_PORT)
        try:
            client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=10)
            client.get_collections()
            return client
        except Exception as exc:
            logger.error(
                "Failed to connect to Qdrant at %s:%d: %s",
                QDRANT_HOST,
                QDRANT_PORT,
                exc,
            )
            raise RuntimeError(f"Qdrant connection failed: {exc}") from exc
    logger.info("Using in-memory Qdrant client")
    return QdrantClient(":memory:")


def create_collection(client: QdrantClient) -> None:
    """
    Ensure the molecules collection exists.

    This method is intentionally non-destructive. If the collection already
    exists, it is reused as long as the vector size matches ``VECTOR_DIM``.

    Args:
        client: A QdrantClient instance.

    Raises:
        ValueError: If an existing collection has an incompatible vector size.
    """
    if client.collection_exists(collection_name=COLLECTION_NAME):
        info = client.get_collection(collection_name=COLLECTION_NAME)
        size = _extract_vector_size(info)
        if size is not None and size != VECTOR_DIM:
            raise ValueError(
                f"Existing collection '{COLLECTION_NAME}' has vector size {size}, "
                f"expected {VECTOR_DIM}. Use recreate_collection() if you want "
                "to delete and recreate it."
            )
        logger.info("Collection '%s' already exists; reusing", COLLECTION_NAME)
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_DIM,
            distance=Distance.COSINE,
        ),
    )
    logger.info("Created collection '%s' (dim=%d, cosine)", COLLECTION_NAME, VECTOR_DIM)


def recreate_collection(client: QdrantClient) -> None:
    """
    Delete and recreate the molecules collection.

    Use this only when destructive reset is intended.

    Args:
        client: A QdrantClient instance.
    """
    if client.collection_exists(collection_name=COLLECTION_NAME):
        logger.warning("Deleting existing collection '%s'", COLLECTION_NAME)
        client.delete_collection(collection_name=COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_DIM,
            distance=Distance.COSINE,
        ),
    )
    logger.info(
        "Recreated collection '%s' (dim=%d, cosine)", COLLECTION_NAME, VECTOR_DIM
    )


def collection_exists_and_populated(client: QdrantClient) -> bool:
    """
    Check whether the molecules collection exists and has points.

    Useful for skipping demo re-indexing on startup when the data is
    already loaded.

    Args:
        client: A QdrantClient instance.

    Returns:
        True if the collection exists and contains at least one point.
    """
    if not client.collection_exists(collection_name=COLLECTION_NAME):
        return False
    info = client.get_collection(collection_name=COLLECTION_NAME)
    return (info.points_count or 0) > 0


def create_payload_indexes(client: QdrantClient) -> None:
    indexed_fields = [
        "molecular_weight",
        "logp",
        "toxicity_score",
    ]

    for field_name in indexed_fields:
        try:
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field_name,
                field_schema=PayloadSchemaType.FLOAT,
            )
        except Exception as exc:
            logger.warning(
                "Could not create payload index for '%s': %s", field_name, exc
            )

    logger.info("Ensured payload indexes on %s", ", ".join(indexed_fields))


def upsert_molecules(
    client: QdrantClient,
    molecules: list[dict],
    embeddings: np.ndarray,
    batch_size: int = UPSERT_BATCH_SIZE,
) -> None:
    """
    Upsert molecule vectors and payloads into Qdrant in batches.

    Uses deterministic UUIDs derived from canonical SMILES so that
    re-indexing the same molecule overwrites its existing point safely.

    Processes in chunks of ``batch_size`` to avoid OOM on large datasets.

    Args:
        client: A QdrantClient instance.
        molecules: List of molecule dictionaries from molecule_processor.
        embeddings: Numpy array of shape (n, vector_dim).
        batch_size: Number of points per upsert call.

    Raises:
        ValueError: If molecule and embedding counts/shapes are incompatible.
    """
    if embeddings.ndim != 2:
        raise ValueError(f"Embeddings must be 2D, got shape {embeddings.shape}")

    if len(molecules) != embeddings.shape[0]:
        raise ValueError(
            "Molecule count and embedding rows must match: "
            f"{len(molecules)} != {embeddings.shape[0]}"
        )

    if embeddings.shape[1] != VECTOR_DIM:
        raise ValueError(
            f"Embedding dimension mismatch: {embeddings.shape[1]} != VECTOR_DIM({VECTOR_DIM})"
        )

    n = len(molecules)
    valid_mask = np.ones(n, dtype=bool)
    if np.any(np.isnan(embeddings)):
        valid_mask &= ~np.isnan(embeddings).any(axis=1)
    if np.any(np.isinf(embeddings)):
        valid_mask &= ~np.isinf(embeddings).any(axis=1)

    norms = np.linalg.norm(embeddings, axis=1)
    if np.any(norms == 0.0):
        valid_mask &= norms > 0.0

    failed_indices = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        points = []

        for i, (mol, emb) in enumerate(
            zip(molecules[start:end], embeddings[start:end], strict=False), start=start
        ):
            if not valid_mask[i]:
                logger.warning("Skipping damaged vector at %d", i)
                failed_indices.append(i)
                continue
            try:
                points.append(
                    PointStruct(
                        id=_smiles_to_uuid(mol["smiles"]),
                        vector=emb.tolist(),
                        payload=mol,
                    )
                )
            except Exception as exc:
                logger.warning("Failed to create point for molecule %d: %s", i, exc)
                failed_indices.append(i)

        if points:
            for attempt in range(3):
                try:
                    client.upsert(collection_name=COLLECTION_NAME, points=points)
                    break
                except Exception as exc:
                    if attempt == 2:
                        logger.error("Upsert %d-%d failed: %s", start, end, exc)
                        failed_indices.extend(range(start, start + len(points)))

    if failed_indices:
        logger.warning(
            "Failed to upsert %d molecules: %s", len(failed_indices), failed_indices
        )

    successful = n - len(failed_indices)
    logger.info("Upserted %d/%d molecules into '%s'", successful, n, COLLECTION_NAME)


def search_similar_molecules(
    query_smiles: str,
    embedder: MoleculeEmbedder,
    client: QdrantClient,
    collection_name: str = COLLECTION_NAME,
    top_k: int = 5,
    mw_max: float | None = None,
    logp_max: float | None = None,
    toxicity_max: float | None = None,
) -> list[dict]:
    """
    Search for molecules similar to the query SMILES string.

    The query is embedded (caller is responsible for validation) and
    then used to search the Qdrant collection. Optional filters on
    molecular properties can be applied.

    Args:
        query_smiles: A validated SMILES string for the query molecule.
        embedder: A MoleculeEmbedder instance.
        client: A QdrantClient instance.
        collection_name: Name of the Qdrant collection.
        top_k: Number of results to return.
        mw_max: Maximum molecular weight filter (optional).
        logp_max: Maximum LogP filter (optional).
        toxicity_max: Maximum toxicity score filter (optional).

    Returns:
        List of dictionaries, each containing:
            - smiles: SMILES of the hit molecule
            - score: cosine similarity score
            - tanimoto_score: ECFP structural similarity score
            - fused_score: Combined similarity score
            - molecular_weight: MW of the hit
            - logp: LogP of the hit
            - toxicity_score: Optional toxicity score if available

    Raises:
        RuntimeError: If Qdrant query fails.
    """
    try:
        embeddings = embedder.embed([query_smiles])
        if np.any(np.isnan(embeddings)):
            raise ValueError("query vector generation failed (nan)")
        query_vector = embeddings[0].tolist()
    except Exception as exc:
        logger.error("Failed to embed query SMILES: %s", exc)
        raise RuntimeError(f"Query embedding failed: {exc}") from exc

    conditions = []
    if mw_max is not None:
        conditions.append(
            FieldCondition(
                key="molecular_weight",
                range=Range(lte=mw_max),
            )
        )
    if logp_max is not None:
        conditions.append(
            FieldCondition(
                key="logp",
                range=Range(lte=logp_max),
            )
        )
    if toxicity_max is not None:
        conditions.append(
            FieldCondition(
                key="toxicity_score",
                range=Range(lte=toxicity_max),
            )
        )

    query_filter = Filter(must=conditions) if conditions else None  # type: ignore[arg-type]

    fetch_limit = top_k * 5

    results = None
    for attempt in range(3):
        try:
            results = client.query_points(
                collection_name=collection_name,
                query=query_vector,
                query_filter=query_filter,
                limit=fetch_limit,
                with_payload=True,
                timeout=30,
            )
            break
        except Exception as exc:
            if attempt == 2:
                logger.error("Qdrant query failed after 3 attempts: %s", exc)
                return []

    if results is None:
        return []

    query_mol = Chem.MolFromSmiles(query_smiles)
    if query_mol is not None:
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        query_fp = generator.GetFingerprint(query_mol)
    else:
        query_fp = None

    hits = []
    for point in results.points:
        payload = point.payload or {}
        toxicity = payload.get("toxicity_score")
        hit_smiles = payload.get("smiles", "")

        tanimoto = 0.0
        if query_fp is not None and hit_smiles:
            hit_mol = Chem.MolFromSmiles(hit_smiles)
            if hit_mol is not None:
                hit_fp = generator.GetFingerprint(hit_mol)
                tanimoto = DataStructs.TanimotoSimilarity(query_fp, hit_fp)

        # TODO: make fusion weights configurable
        fused_score = 0.5 * point.score + 0.5 * tanimoto

        hits.append(
            {
                "smiles": hit_smiles,
                "score": round(point.score, 4),
                "tanimoto_score": round(tanimoto, 4),
                "fused_score": round(fused_score, 4),
                "molecular_weight": payload.get("molecular_weight", 0.0),
                "logp": payload.get("logp", 0.0),
                "toxicity_score": (
                    toxicity if isinstance(toxicity, (int, float)) else None
                ),
            }
        )

    hits.sort(key=lambda x: x["fused_score"], reverse=True)
    return hits[:top_k]
