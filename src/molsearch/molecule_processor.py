"""
SMILES validation and descriptor computation.
"""

from __future__ import annotations

import logging
import math

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

logger = logging.getLogger(__name__)


def compute_toxicity_proxy(mol: Chem.Mol) -> float:
    """
    Estimate a heuristic toxicity score from RDKit descriptors.

    This is NOT a real toxicity prediction. It is a rule-based proxy
    derived from molecular properties commonly associated with toxicity
    risk (Lipinski violations, high aromaticity, extreme LogP, etc.).

    The score is deterministic, fast, and uses no external dependencies.

    Args:
        mol: An RDKit Mol object.

    Returns:
        A float between 0.0 (low estimated risk) and 1.0 (high estimated risk).
    """
    mw = Descriptors.MolWt(mol)  # type: ignore[attr-defined]
    logp = Descriptors.MolLogP(mol)  # type: ignore[attr-defined]
    hbd = Descriptors.NumHDonors(mol)  # type: ignore[attr-defined]
    hba = Descriptors.NumHAcceptors(mol)  # type: ignore[attr-defined]
    tpsa = Descriptors.TPSA(mol)  # type: ignore[attr-defined]
    n_aromatic = rdMolDescriptors.CalcNumAromaticRings(mol)

    # Continuous sub-scores with lower thresholds so drug-like molecules
    # still produce meaningful variation instead of flat zeros.

    # MW: starts contributing above 350, saturates at 800
    mw_score = max(0.0, min((mw - 350) / 450, 1.0))

    # LogP: starts above 3, saturates at 8
    logp_score = max(0.0, min((logp - 3) / 5, 1.0))

    # HBD: starts above 2, saturates at 7
    hbd_score = max(0.0, min((hbd - 2) / 5, 1.0))

    # HBA: starts above 5, saturates at 15
    hba_score = max(0.0, min((hba - 5) / 10, 1.0))

    # Aromatic rings: linear 0–6, most drug-like molecules have 1–3
    arom_score = min(n_aromatic / 6, 1.0)

    # TPSA: low TPSA (< 75) increases risk — membrane disruption concern
    tpsa_score = max(0.0, min((75 - tpsa) / 75, 1.0))

    # Weighted combination
    raw = (
        0.25 * mw_score
        + 0.25 * logp_score
        + 0.15 * hbd_score
        + 0.15 * hba_score
        + 0.10 * arom_score
        + 0.10 * tpsa_score
    )

    return round(max(0.0, min(raw, 1.0)), 3)


def validate_and_canonicalize(
    smiles: str,
    toxicity_score: float | None = None,
) -> dict | None:
    """
    Validate a SMILES string and return canonical form with basic descriptors.

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
    if mol is None:
        return None

    num_atoms = mol.GetNumAtoms()
    if num_atoms == 0 or mol.GetNumBonds() == 0:
        logger.warning("invalid structure: %s", normalized_smiles)
        return None

    try:
        canonical_smiles = Chem.MolToSmiles(mol)
    except Exception:
        logger.warning("canonicalization failed: %s", normalized_smiles)
        return None

    verify_mol = Chem.MolFromSmiles(canonical_smiles)
    if verify_mol is None or verify_mol.GetNumAtoms() != num_atoms:
        logger.warning("canonical integrity check failed: %s", normalized_smiles)
        return None

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
    else:
        payload["toxicity_score"] = compute_toxicity_proxy(mol)

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
            logger.warning("invalid toxicity score for smiles: %s", smi)
            continue

        if result is not None:
            results.append(result)
        else:
            logger.warning("skipping invalid smiles: %s", smi)

    return results
