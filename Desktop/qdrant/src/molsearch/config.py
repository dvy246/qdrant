"""
Shared configuration constants for the molecule search pipeline.

All tunable parameters live here. Other modules import from this file
so that changes propagate consistently across the pipeline, API, and UI.
"""

from __future__ import annotations

import logging
import os


# ---- Env helpers ----
def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value is not None else default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logging.getLogger(__name__).warning(
            "Invalid integer for %s=%r. Using default=%d",
            name,
            value,
            default,
        )
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


# ---- Logging ----
LOG_LEVEL: str = _env_str("MOLSEARCH_LOG_LEVEL", "INFO")
LOG_FORMAT: str = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)

# ---- HuggingFace model ----
MODEL_NAME: str = _env_str("MOLSEARCH_MODEL_NAME", "seyonec/ChemBERTa-zinc-base-v1")

# ---- Qdrant settings ----
COLLECTION_NAME: str = _env_str("MOLSEARCH_COLLECTION_NAME", "molecules")

# Embedding vector dimension -- must match model.config.hidden_size.
# The MoleculeEmbedder validates this at startup automatically.
VECTOR_DIM: int = _env_int("MOLSEARCH_VECTOR_DIM", 768)

# Batch size for embedding inference
BATCH_SIZE: int = _env_int("MOLSEARCH_BATCH_SIZE", 32)

# Qdrant connection settings (used when USE_PERSISTENT_QDRANT is True)
QDRANT_HOST: str = _env_str("MOLSEARCH_QDRANT_HOST", "localhost")
QDRANT_PORT: int = _env_int("MOLSEARCH_QDRANT_PORT", 6333)

# Set to True to use a persistent Qdrant instance instead of in-memory
USE_PERSISTENT_QDRANT: bool = _env_bool("MOLSEARCH_PERSISTENT_QDRANT", False)

# Batch size for upserting points into Qdrant (avoids OOM at scale)
UPSERT_BATCH_SIZE: int = _env_int("MOLSEARCH_UPSERT_BATCH_SIZE", 1000)

# API validation limits
MAX_SMILES_LENGTH: int = _env_int("MOLSEARCH_MAX_SMILES_LENGTH", 2048)

# ---- Demo data (single source of truth) ----
SAMPLE_SMILES: list[str] = [
    "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # Ibuprofen
    "CC(=O)Nc1ccc(O)cc1",  # Acetaminophen
    "O=C(O)Cc1ccccc1Nc1c(Cl)cccc1Cl",  # Diclofenac
    "COc1ccc2cc(CC(C)C(=O)O)ccc2c1",  # Naproxen
    "OC(=O)c1ccccc1O",  # Salicylic acid
    "c1ccc(cc1)C(=O)O",  # Benzoic acid
    "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",  # Testosterone
    "OC(=O)c1cc(O)c(O)c(O)c1",  # Gallic acid
    "c1ccc2c(c1)ccc1ccccc12",  # Naphthalene
]
