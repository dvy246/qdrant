"""Tests for the data_loader module."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from molsearch.data_loader import (
    _FALLBACK_SMILES,
    _parse_zinc_csv,
    load_dataset,
)


@pytest.fixture
def zinc_csv(tmp_path: Path) -> Path:
    """Create a minimal ZINC-format CSV for testing."""
    csv_path = tmp_path / "zinc_test.csv"
    rows = [
        {"smiles": "CC(=O)Oc1ccccc1C(=O)O"},  # Aspirin
        {"smiles": "CC(C)Cc1ccc(cc1)C(C)C(=O)O"},  # Ibuprofen
        {"smiles": "CC(=O)Nc1ccc(O)cc1"},  # Acetaminophen
        {"smiles": "INVALID_SMILES"},  # Should be skipped
        {"smiles": "CC(=O)Oc1ccccc1C(=O)O"},  # Duplicate of Aspirin
        {"smiles": "COc1ccc2cc(CC(C)C(=O)O)ccc2c1"},  # Naproxen
        {"smiles": ""},  # Empty — should be skipped
        {"smiles": "OC(=O)c1ccccc1O"},  # Salicylic acid
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["smiles"])
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


class TestParseZincCSV:
    """Tests for _parse_zinc_csv."""

    def test_parses_valid_smiles(self, zinc_csv: Path) -> None:
        result = _parse_zinc_csv(zinc_csv, max_molecules=100)
        assert (
            len(result) == 5
        )  # 3 valid + 1 dup + 1 invalid + 1 empty = 5 unique valid

    def test_deduplicates(self, zinc_csv: Path) -> None:
        result = _parse_zinc_csv(zinc_csv, max_molecules=100)
        assert len(result) == len(set(result))

    def test_respects_max_molecules(self, zinc_csv: Path) -> None:
        result = _parse_zinc_csv(zinc_csv, max_molecules=2)
        assert len(result) == 2

    def test_skips_invalid_smiles(self, zinc_csv: Path) -> None:
        result = _parse_zinc_csv(zinc_csv, max_molecules=100)
        assert "INVALID_SMILES" not in result

    def test_returns_canonical_smiles(self, zinc_csv: Path) -> None:
        result = _parse_zinc_csv(zinc_csv, max_molecules=100)
        from rdkit import Chem

        for smi in result:
            mol = Chem.MolFromSmiles(smi)
            assert mol is not None
            assert Chem.MolToSmiles(mol) == smi


class TestLoadDataset:
    """Tests for load_dataset."""

    def test_fallback_on_download_failure(self, tmp_path: Path) -> None:
        """If download fails and no cache exists, returns fallback molecules."""
        with (
            patch(
                "molsearch.data_loader._get_cache_path",
                return_value=tmp_path / "missing.csv",
            ),
            patch(
                "molsearch.data_loader._download_file",
                side_effect=Exception("no network"),
            ),
        ):
            smiles, toxicities = load_dataset(max_molecules=5)

        assert len(smiles) <= 5
        assert len(smiles) == len(toxicities)
        assert all(t is None for t in toxicities)

    def test_returns_tuple_of_smiles_and_none_toxicities(self, zinc_csv: Path) -> None:
        """Returns (smiles, [None, ...]) since ZINC has no toxicity data."""
        with patch("molsearch.data_loader._get_cache_path", return_value=zinc_csv):
            smiles, toxicities = load_dataset(max_molecules=100)

        assert len(smiles) > 0
        assert len(smiles) == len(toxicities)
        assert all(t is None for t in toxicities)

    def test_max_molecules_respected(self, zinc_csv: Path) -> None:
        with patch("molsearch.data_loader._get_cache_path", return_value=zinc_csv):
            smiles, _ = load_dataset(max_molecules=2)

        assert len(smiles) == 2

    def test_fallback_smiles_are_valid(self) -> None:
        """All built-in fallback SMILES must be valid."""
        from rdkit import Chem

        for smi in _FALLBACK_SMILES:
            mol = Chem.MolFromSmiles(smi)
            assert mol is not None, f"Fallback SMILES invalid: {smi}"
