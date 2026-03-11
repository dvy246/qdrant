"""
Dataset loader for real molecular data.

Downloads and caches the ZINC-250k dataset (drug-like molecules from the
ZINC database), validates SMILES with RDKit, deduplicates, and returns
a clean subset ready for the embedding pipeline.

The ZINC database is the same source ChemBERTa was trained on, making
it a natural fit for this system.
"""

from __future__ import annotations

import csv
import hashlib
import logging
import os
from pathlib import Path

from rdkit import Chem

from molsearch.config import DATA_CACHE_DIR, DATASET_SIZE

logger = logging.getLogger(__name__)

ZINC_250K_URL = (
    "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/"
    "master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv"
)

ZINC_FILENAME = "zinc_250k.csv"

# Fallback molecules if the download fails (for offline/CI environments)
_FALLBACK_SMILES: list[str] = [
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


def _get_cache_path() -> Path:
    """Return the local cache path for the ZINC CSV."""
    cache_dir = Path(DATA_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / ZINC_FILENAME


def _download_file(url: str, dest: Path) -> None:
    """Download a file from a URL to a local path."""
    import urllib.request

    logger.info("Downloading ZINC-250k dataset from %s ...", url)
    try:
        urllib.request.urlretrieve(url, str(dest))
        logger.info("Downloaded to %s (%.1f MB)", dest, dest.stat().st_size / 1e6)
    except Exception as exc:
        logger.error("Download failed: %s", exc)
        if dest.exists():
            dest.unlink()
        raise


def _parse_zinc_csv(csv_path: Path, max_molecules: int) -> list[str]:
    """
    Parse the ZINC-250k CSV and return validated, deduplicated SMILES.

    The CSV has columns: smiles, logP, qed, SAS
    """
    seen: set[str] = set()
    valid_smiles: list[str] = []
    total_rows = 0
    invalid_count = 0
    duplicate_count = 0

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            total_rows += 1
            raw_smiles = row.get("smiles", "").strip()

            if not raw_smiles:
                invalid_count += 1
                continue

            mol = Chem.MolFromSmiles(raw_smiles)
            if mol is None or mol.GetNumAtoms() == 0:
                invalid_count += 1
                continue

            canonical = Chem.MolToSmiles(mol)

            if canonical in seen:
                duplicate_count += 1
                continue

            seen.add(canonical)
            valid_smiles.append(canonical)

            if len(valid_smiles) >= max_molecules:
                break

    logger.info(
        "ZINC dataset stats: %d rows scanned, %d valid, %d invalid, %d duplicates, %d returned",
        total_rows,
        len(valid_smiles),
        invalid_count,
        duplicate_count,
        len(valid_smiles),
    )
    return valid_smiles


def load_dataset(
    max_molecules: int = DATASET_SIZE,
) -> tuple[list[str], list[float | None]]:
    """
    Load molecules from the ZINC-250k dataset.

    Downloads the dataset on first call, caches locally for subsequent runs.
    Returns validated, deduplicated SMILES with None toxicity scores
    (ZINC does not include toxicity data).

    Args:
        max_molecules: Maximum number of molecules to return (1k-10k range).

    Returns:
        Tuple of (smiles_list, toxicity_scores) where toxicity_scores
        is a list of None values (ZINC has no toxicity annotations).

    Falls back to a small built-in molecule list if download fails.
    """
    cache_path = _get_cache_path()

    if not cache_path.exists():
        try:
            _download_file(ZINC_250K_URL, cache_path)
        except Exception:
            logger.warning(
                "Could not download ZINC dataset; using %d fallback molecules",
                len(_FALLBACK_SMILES),
            )
            smiles = _FALLBACK_SMILES[:max_molecules]
            return smiles, [None] * len(smiles)

    smiles = _parse_zinc_csv(cache_path, max_molecules)

    if not smiles:
        logger.warning("No valid molecules parsed; using fallback")
        smiles = _FALLBACK_SMILES[:max_molecules]

    toxicity_scores: list[float | None] = [None] * len(smiles)

    logger.info("Dataset ready: %d molecules", len(smiles))
    return smiles, toxicity_scores
