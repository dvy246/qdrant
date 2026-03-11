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
    n_atoms = mol.GetNumHeavyAtoms()

    # Each sub-score maps a descriptor to [0, 1] based on ranges
    # commonly associated with drug-likeness violations.

    # MW: drug-like < 500, concerning > 800
    mw_score = max(0.0, min((mw - 500) / 300, 1.0)) if mw > 500 else 0.0

    # LogP: drug-like -0.4 to 5.6, concerning > 5
    logp_score = max(0.0, min((logp - 5) / 3, 1.0)) if logp > 5 else 0.0

    # HBD: drug-like <= 5, concerning > 5
    hbd_score = max(0.0, min((hbd - 5) / 5, 1.0)) if hbd > 5 else 0.0

    # HBA: drug-like <= 10, concerning > 10
    hba_score = max(0.0, min((hba - 10) / 5, 1.0)) if hba > 10 else 0.0

    # Aromatic rings: > 3 raises concern (PAH-like structures)
    arom_score = max(0.0, min((n_aromatic - 3) / 3, 1.0)) if n_aromatic > 3 else 0.0

    # TPSA: very low TPSA (< 20) can indicate membrane-disrupting compounds
    tpsa_score = max(0.0, min((20 - tpsa) / 20, 1.0)) if tpsa < 20 else 0.0

    # Heavy atom count: very large molecules (> 50 atoms) correlate with toxicity
    size_score = max(0.0, min((n_atoms - 50) / 30, 1.0)) if n_atoms > 50 else 0.0

    # Weighted combination — MW and LogP dominate since they are the
    # strongest predictors of ADMET liability in simple rule-based models.
    weights = {
        "mw": 0.25,
        "logp": 0.25,
        "hbd": 0.10,
        "hba": 0.10,
        "arom": 0.15,
        "tpsa": 0.05,
        "size": 0.10,
    }

    raw = (
        weights["mw"] * mw_score
        + weights["logp"] * logp_score
        + weights["hbd"] * hbd_score
        + weights["hba"] * hba_score
        + weights["arom"] * arom_score
        + weights["tpsa"] * tpsa_score
        + weights["size"] * size_score
    )

    return round(max(0.0, min(raw, 1.0)), 4)


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
