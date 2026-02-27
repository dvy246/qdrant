"""
Tests for embedder.py — ChemBERTa embedding, mean pooling, normalization.

These tests run fully offline by mocking HuggingFace model/tokenizer loading.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import molsearch.embedder as embedder_module
from molsearch.config import VECTOR_DIM
from molsearch.embedder import MoleculeEmbedder, _get_device


class _FakeTokenizer:
    """Deterministic tokenizer-like stub used for offline tests."""

    def __call__(
        self,
        batch: list[str],
        padding: bool,
        truncation: bool,
        return_tensors: str,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        assert padding is True
        assert truncation is False
        assert return_tensors == "pt"

        token_rows: list[list[int]] = []
        max_seq_len = 0

        for smiles in batch:
            cleaned = smiles.strip()
            if not cleaned:
                cleaned = "_"

            weighted_sum = sum((i + 1) * ord(ch) for i, ch in enumerate(cleaned))
            row = [
                (len(cleaned) % 251) + 1,
                (sum(ord(ch) for ch in cleaned) % 257) + 1,
                (weighted_sum % 263) + 1,
                (ord(cleaned[0]) % 269) + 1,
            ]
            token_rows.append(row)
            max_seq_len = max(max_seq_len, len(row))

        input_ids = torch.zeros((len(batch), max_seq_len), dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_seq_len), dtype=torch.long)

        for i, row in enumerate(token_rows):
            row_tensor = torch.tensor(row, dtype=torch.long)
            input_ids[i, : len(row)] = row_tensor
            attention_mask[i, : len(row)] = 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


class _FakeModel(torch.nn.Module):
    """Deterministic encoder-like stub with the same interface as HF AutoModel."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=VECTOR_DIM)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> SimpleNamespace:
        _ = attention_mask
        batch_size, seq_len = input_ids.shape
        hidden = torch.zeros(
            (batch_size, seq_len, VECTOR_DIM),
            dtype=torch.float32,
            device=input_ids.device,
        )

        batch_ix = torch.arange(batch_size, device=input_ids.device)
        for pos in range(seq_len):
            token_ids = input_ids[:, pos]
            idx1 = token_ids % VECTOR_DIM
            idx2 = (token_ids * 37 + pos * 11) % VECTOR_DIM
            hidden[batch_ix, pos, idx1] += 1.0
            hidden[batch_ix, pos, idx2] += 0.5

        return SimpleNamespace(last_hidden_state=hidden)


@pytest.fixture(autouse=True)
def _mock_hf_loading(monkeypatch):
    """Patch HF loaders so tests are deterministic and network-free."""

    def _tokenizer_from_pretrained(model_name: str):
        if model_name == "nonexistent/model-that-does-not-exist":
            raise OSError("mocked missing tokenizer")
        return _FakeTokenizer()

    def _model_from_pretrained(model_name: str):
        if model_name == "nonexistent/model-that-does-not-exist":
            raise OSError("mocked missing model")
        return _FakeModel()

    monkeypatch.setattr(
        embedder_module.AutoTokenizer,
        "from_pretrained",
        staticmethod(_tokenizer_from_pretrained),
    )
    monkeypatch.setattr(
        embedder_module.AutoModel,
        "from_pretrained",
        staticmethod(_model_from_pretrained),
    )


@pytest.fixture
def embedder():
    return MoleculeEmbedder()


class TestGetDevice:
    """Test device auto-detection."""

    def test_returns_torch_device(self):
        device = _get_device()
        assert isinstance(device, torch.device)

    def test_device_is_known_type(self):
        device = _get_device()
        assert device.type in ("cpu", "cuda", "mps")


class TestMoleculeEmbedder:
    """Tests for MoleculeEmbedder initialization and configuration."""

    def test_vector_dim_is_768(self, embedder):
        assert embedder.vector_dim == 768

    def test_model_is_in_eval_mode(self, embedder):
        assert not embedder.model.training

    def test_invalid_model_name_raises(self):
        with pytest.raises(OSError):
            MoleculeEmbedder(model_name="nonexistent/model-that-does-not-exist")


class TestEmbed:
    """Tests for the embed() method."""

    def test_output_shape_single(self, embedder):
        result = embedder.embed(["CC(=O)Oc1ccccc1C(=O)O"])
        assert result.shape == (1, 768)

    def test_output_shape_batch(self, embedder):
        smiles = [
            "CC(=O)Oc1ccccc1C(=O)O",
            "CC(=O)Nc1ccc(O)cc1",
            "OC(=O)c1ccccc1O",
        ]
        result = embedder.embed(smiles)
        assert result.shape == (3, 768)

    def test_empty_list_returns_empty(self, embedder):
        result = embedder.embed([])
        assert result.shape == (0, 768)
        assert result.dtype == np.float32

    def test_output_is_float32(self, embedder):
        result = embedder.embed(["CC(=O)Oc1ccccc1C(=O)O"])
        assert result.dtype == np.float32

    def test_vectors_are_l2_normalized(self, embedder):
        result = embedder.embed(["CC(=O)Oc1ccccc1C(=O)O", "CC(=O)Nc1ccc(O)cc1"])
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_same_smiles_gives_same_embedding(self, embedder):
        v1 = embedder.embed(["CC(=O)Oc1ccccc1C(=O)O"])
        v2 = embedder.embed(["CC(=O)Oc1ccccc1C(=O)O"])
        np.testing.assert_array_almost_equal(v1, v2)

    def test_different_smiles_give_different_embeddings(self, embedder):
        result = embedder.embed(["CC(=O)Oc1ccccc1C(=O)O", "c1ccccc1"])
        assert not np.allclose(result[0], result[1])

    def test_batching_is_consistent(self, embedder):
        smiles = ["CC(=O)Oc1ccccc1C(=O)O", "CC(=O)Nc1ccc(O)cc1", "OC(=O)c1ccccc1O"]
        v_batch = embedder.embed(smiles, batch_size=32)
        v_single = embedder.embed(smiles, batch_size=1)
        np.testing.assert_array_almost_equal(v_batch, v_single, decimal=5)
