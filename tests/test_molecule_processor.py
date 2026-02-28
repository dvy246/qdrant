"""
Tests for molecule_processor.py - SMILES validation, canonicalization, descriptors.
"""

from __future__ import annotations

import math

import pytest

from molsearch.molecule_processor import process_smiles_batch, validate_and_canonicalize


class TestValidateAndCanonicalize:
    def test_valid_smiles_returns_dict(self):
        result = validate_and_canonicalize("CC(=O)Oc1ccccc1C(=O)O")
        assert result is not None
        assert isinstance(result, dict)

    def test_canonical_smiles_key(self):
        result = validate_and_canonicalize("CC(=O)Oc1ccccc1C(=O)O")
        assert result is not None
        assert "smiles" in result
        assert isinstance(result["smiles"], str)
        assert len(result["smiles"]) > 0

    def test_canonicalization_is_deterministic(self):
        r1 = validate_and_canonicalize("OC(=O)c1ccccc1O")
        r2 = validate_and_canonicalize("c1ccc(O)c(C(=O)O)c1")
        assert r1 is not None and r2 is not None
        assert r1["smiles"] == r2["smiles"]

    def test_descriptors_present(self):
        result = validate_and_canonicalize("CC(=O)Oc1ccccc1C(=O)O")
        assert result is not None
        for key in [
            "smiles",
            "molecular_weight",
            "logp",
            "num_h_donors",
            "num_h_acceptors",
            "tpsa",
        ]:
            assert key in result

    def test_invalid_smiles_returns_none(self):
        assert validate_and_canonicalize("NOT_A_MOLECULE") is None

    def test_empty_string_returns_none(self):
        assert validate_and_canonicalize("") is None

    def test_whitespace_only_smiles_returns_none(self):
        assert validate_and_canonicalize("   ") is None

    def test_toxicity_score_is_included_when_provided(self):
        result = validate_and_canonicalize("CC(=O)Oc1ccccc1C(=O)O", toxicity_score=0.42)
        assert result is not None
        assert result["toxicity_score"] == 0.42

    def test_non_finite_toxicity_raises(self):
        with pytest.raises(ValueError):
            validate_and_canonicalize("CC(=O)Oc1ccccc1C(=O)O", toxicity_score=math.inf)


class TestProcessSmilesBatch:
    def test_empty_list(self):
        assert process_smiles_batch([]) == []

    def test_all_valid(self):
        smiles = ["CC(=O)Oc1ccccc1C(=O)O", "CC(=O)Nc1ccc(O)cc1"]
        results = process_smiles_batch(smiles)
        assert len(results) == 2

    def test_all_invalid(self):
        smiles = ["INVALID_1", "INVALID_2"]
        results = process_smiles_batch(smiles)
        assert len(results) == 0

    def test_mixed_valid_and_invalid(self):
        smiles = ["CC(=O)Oc1ccccc1C(=O)O", "NOT_VALID", "CC(=O)Nc1ccc(O)cc1"]
        results = process_smiles_batch(smiles)
        assert len(results) == 2

    def test_invalid_smiles_logged(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            process_smiles_batch(["INVALID"])
        assert "skipping invalid smiles" in caplog.text.lower()

    def test_toxicity_scores_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            process_smiles_batch(
                ["CC(=O)Oc1ccccc1C(=O)O", "CC(=O)Nc1ccc(O)cc1"],
                toxicity_scores=[0.2],
            )

    def test_toxicity_scores_are_attached(self):
        results = process_smiles_batch(
            ["CC(=O)Oc1ccccc1C(=O)O", "CC(=O)Nc1ccc(O)cc1"],
            toxicity_scores=[0.3, 0.7],
        )
        assert len(results) == 2
        assert results[0]["toxicity_score"] == 0.3
        assert results[1]["toxicity_score"] == 0.7
