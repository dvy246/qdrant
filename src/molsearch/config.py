"""
Shared configuration constants for the molecule search pipeline.

All tunable parameters live here. Other modules import from this file
so that changes propagate consistently across the pipeline, API, and UI.
"""

from __future__ import annotations

import logging
import os


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value is not None else default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
        if parsed < 0:
            logging.getLogger(__name__).warning(
                "invalid neg value for %s=%r", name, value
            )
            raise ValueError(f"bad env value for {name}: {value}")
        return parsed
    except ValueError as exc:
        if "bad env" not in str(exc):
            logging.getLogger(__name__).warning("invalid int for %s=%r", name, value)
            raise ValueError(f"bad env value for {name}: {value}") from exc
        raise


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    value_lower = value.strip().lower()
    if value_lower in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value_lower in {"0", "false", "f", "no", "n", "off"}:
        return False
    logging.getLogger(__name__).warning("invalid bool for %s=%r", name, value)
    raise ValueError(f"bad bool format for {name}: {value}")


LOG_LEVEL: str = _env_str("MOLSEARCH_LOG_LEVEL", "INFO")
LOG_FORMAT: str = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)


MODEL_NAME: str = _env_str("MOLSEARCH_MODEL_NAME", "seyonec/ChemBERTa-zinc-base-v1")


COLLECTION_NAME: str = _env_str("MOLSEARCH_COLLECTION_NAME", "molecules")

# Must match model.config.hidden_size; validated at startup.
VECTOR_DIM: int = _env_int("MOLSEARCH_VECTOR_DIM", 768)


BATCH_SIZE: int = _env_int("MOLSEARCH_BATCH_SIZE", 32)


QDRANT_HOST: str = _env_str("MOLSEARCH_QDRANT_HOST", "localhost")
QDRANT_PORT: int = _env_int("MOLSEARCH_QDRANT_PORT", 6333)


USE_PERSISTENT_QDRANT: bool = _env_bool("MOLSEARCH_PERSISTENT_QDRANT", False)


UPSERT_BATCH_SIZE: int = _env_int("MOLSEARCH_UPSERT_BATCH_SIZE", 1000)


MAX_SMILES_LENGTH: int = _env_int("MOLSEARCH_MAX_SMILES_LENGTH", 2048)


# Dataset configuration
DATASET_SIZE: int = _env_int("MOLSEARCH_DATASET_SIZE", 2000)

DATA_CACHE_DIR: str = _env_str(
    "MOLSEARCH_DATA_CACHE_DIR",
    os.path.join(os.path.expanduser("~"), ".cache", "molsearch"),
)
