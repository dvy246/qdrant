"""
Molecule processor: SMILES validation, canonicalization, and descriptor computation.

Uses RDKit for all cheminformatics operations.
"""

from __future__ import annotations

import logging
import math

from rdkit import Chem
from rdkit.Chem import Descriptors

logger = logging.getLogger(__name__)


def validate_and_canonicalize(
    smiles: str,
    toxicity_score: float | None = None,
) -> dict | None:
    """
    Validate a SMILES string and return canonical form with basic descriptors.

    Returns a dictionary with canonical SMILES and computed molecular
    descriptors, or None if the SMILES string is invalid.

    Args:
        smiles: A SMILES string representing a molecule.
        toxicity_score: Optional precomputed toxicity score metadata.

    Returns:
        A dictionary containing:
            - smiles: Canonical SMILES string
            - molecular_weight: Molecular weight in Da
            - logp: Calculated LogP (Wildman-Crippen)
            - num_h_donors: Number of hydrogen bond donors
            - num_h_acceptors: Number of hydrogen bond acceptors
            - tpsa: Topological polar surface area
            - toxicity_score: Optional toxicity score (if provided)
        Returns None if the input SMILES is invalid.

    Raises:
        ValueError: If toxicity_score is provided but not finite.
    """
    normalized_smiles = smiles.strip()
    if not normalized_smiles:
        return None

    mol = Chem.MolFromSmiles(normalized_smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    canonical_smiles = Chem.MolToSmiles(mol)

    payload = {
        "smiles": canonical_smiles,
        "molecular_weight": round(Descriptors.MolWt(mol), 2),  # type: ignore[attr-defined]
        "logp": round(Descriptors.MolLogP(mol), 2),  # type: ignore[attr-defined]
        "num_h_donors": Descriptors.NumHDonors(mol),  # type: ignore[attr-defined]
        "num_h_acceptors": Descriptors.NumHAcceptors(mol),  # type: ignore[attr-defined]
        "tpsa": round(Descriptors.TPSA(mol), 2),  # type: ignore[attr-defined]
    }

    if toxicity_score is not None:
        if not math.isfinite(toxicity_score):
            raise ValueError("toxicity_score must be a finite float")
        payload["toxicity_score"] = float(toxicity_score)

    return payload


def process_smiles_batch(
    smiles_list: list[str],
    toxicity_scores: list[float | None] | None = None,
) -> list[dict]:
    """
    Validate and canonicalize a batch of SMILES strings.

    Invalid SMILES are skipped with a logged warning.

    Args:
        smiles_list: List of SMILES strings.
        toxicity_scores: Optional list of toxicity scores aligned to smiles_list.

    Returns:
        List of dictionaries for valid molecules (see validate_and_canonicalize).

    Raises:
        ValueError: If toxicity_scores length does not match smiles_list length.
    """
    if toxicity_scores is not None and len(toxicity_scores) != len(smiles_list):
        raise ValueError(
            "toxicity_scores length must match smiles_list length: "
            f"{len(toxicity_scores)} != {len(smiles_list)}"
        )

    results = []
    for i, smi in enumerate(smiles_list):
        toxicity_score = toxicity_scores[i] if toxicity_scores is not None else None

        try:
            result = validate_and_canonicalize(smi, toxicity_score=toxicity_score)
        except ValueError:
            logger.warning("Skipping SMILES with invalid toxicity score: %s", smi)
            continue

        if result is not None:
            results.append(result)
        else:
            logger.warning("Skipping invalid SMILES: %s", smi)

    return results
