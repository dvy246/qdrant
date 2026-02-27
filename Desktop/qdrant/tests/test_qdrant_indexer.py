"""
Tests for qdrant_indexer.py — collection management, upsert, and search.

Uses an in-memory Qdrant client so no server is required.
"""

from __future__ import annotations

import numpy as np
import pytest

from molsearch.config import COLLECTION_NAME, VECTOR_DIM
from molsearch.molecule_processor import process_smiles_batch
from molsearch.qdrant_indexer import (
    _smiles_to_uuid,
    collection_exists_and_populated,
    create_collection,
    create_payload_indexes,
    get_qdrant_client,
    search_similar_molecules,
    upsert_molecules,
)


class DummyEmbedder:
    """Small deterministic embedder used to keep tests offline and fast."""

    def embed(self, smiles_list: list[str]) -> np.ndarray:
        vectors = np.zeros((len(smiles_list), VECTOR_DIM), dtype=np.float32)
        for i, smiles in enumerate(smiles_list):
            base = sum((j + 1) * ord(ch) for j, ch in enumerate(smiles))
            idx1 = base % VECTOR_DIM
            idx2 = (base * 31 + 17) % VECTOR_DIM
            idx3 = (base * 17 + 5) % VECTOR_DIM
            vectors[i, idx1] = 1.0
            vectors[i, idx2] = 0.5
            vectors[i, idx3] = 0.25

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return vectors / norms


# ---- Fixtures ----


@pytest.fixture
def embedder():
    return DummyEmbedder()


@pytest.fixture
def client():
    """Fresh in-memory Qdrant client per test."""
    return get_qdrant_client()


@pytest.fixture
def populated_index(client, embedder):
    """Client with a small index of 3 molecules."""
    smiles = [
        "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
        "CC(=O)Nc1ccc(O)cc1",  # Acetaminophen
        "OC(=O)c1ccccc1O",  # Salicylic acid
    ]
    toxicity_scores = [0.25, 0.65, 0.12]

    molecules = process_smiles_batch(smiles, toxicity_scores=toxicity_scores)
    embeddings = embedder.embed([m["smiles"] for m in molecules])
    create_collection(client)
    upsert_molecules(client, molecules, embeddings)
    create_payload_indexes(client)
    return client


# ---- UUID generation ----


class TestSmilesToUuid:
    """Tests for deterministic UUID generation."""

    def test_returns_string(self):
        result = _smiles_to_uuid("CC(=O)Oc1ccccc1C(=O)O")
        assert isinstance(result, str)

    def test_deterministic(self):
        u1 = _smiles_to_uuid("CC(=O)Oc1ccccc1C(=O)O")
        u2 = _smiles_to_uuid("CC(=O)Oc1ccccc1C(=O)O")
        assert u1 == u2

    def test_different_smiles_give_different_uuids(self):
        u1 = _smiles_to_uuid("CC(=O)Oc1ccccc1C(=O)O")
        u2 = _smiles_to_uuid("CC(=O)Nc1ccc(O)cc1")
        assert u1 != u2


# ---- Collection management ----


class TestCollectionManagement:
    """Tests for collection creation and existence checks."""

    def test_create_collection(self, client):
        create_collection(client)
        assert client.collection_exists(collection_name=COLLECTION_NAME)

    def test_create_collection_idempotent(self, client):
        create_collection(client)
        create_collection(client)  # Should not raise
        assert client.collection_exists(collection_name=COLLECTION_NAME)

    def test_collection_not_populated_when_empty(self, client):
        create_collection(client)
        assert not collection_exists_and_populated(client)

    def test_collection_populated_after_upsert(self, populated_index):
        assert collection_exists_and_populated(populated_index)

    def test_collection_not_exists_returns_false(self, client):
        assert not collection_exists_and_populated(client)


# ---- Upsert ----


class TestUpsert:
    """Tests for upserting molecules into Qdrant."""

    def test_upsert_stores_correct_count(self, populated_index):
        info = populated_index.get_collection(collection_name=COLLECTION_NAME)
        assert info.points_count == 3

    def test_upsert_idempotent(self, client, embedder):
        """Upserting the same molecules twice should not duplicate points."""
        smiles = ["CC(=O)Oc1ccccc1C(=O)O"]
        molecules = process_smiles_batch(smiles)
        embeddings = embedder.embed([m["smiles"] for m in molecules])
        create_collection(client)
        upsert_molecules(client, molecules, embeddings)
        upsert_molecules(client, molecules, embeddings)
        info = client.get_collection(collection_name=COLLECTION_NAME)
        assert info.points_count == 1  # UUID is deterministic

    def test_upsert_length_mismatch_raises(self, client, embedder):
        smiles = ["CC(=O)Oc1ccccc1C(=O)O", "CC(=O)Nc1ccc(O)cc1"]
        molecules = process_smiles_batch(smiles)
        embeddings = embedder.embed([molecules[0]["smiles"]])

        create_collection(client)
        with pytest.raises(ValueError):
            upsert_molecules(client, molecules, embeddings)

    def test_payload_indexes(self, populated_index):
        """Creating payload indexes should not raise."""
        create_payload_indexes(populated_index)


# ---- Search ----


class TestSearch:
    """Tests for similarity search."""

    def test_search_returns_list(self, populated_index, embedder):
        results = search_similar_molecules(
            query_smiles="CC(=O)Oc1ccccc1C(=O)O",
            embedder=embedder,
            client=populated_index,
            top_k=3,
        )
        assert isinstance(results, list)

    def test_search_top_k(self, populated_index, embedder):
        results = search_similar_molecules(
            query_smiles="CC(=O)Oc1ccccc1C(=O)O",
            embedder=embedder,
            client=populated_index,
            top_k=2,
        )
        assert len(results) == 2

    def test_search_result_has_required_keys(self, populated_index, embedder):
        results = search_similar_molecules(
            query_smiles="CC(=O)Oc1ccccc1C(=O)O",
            embedder=embedder,
            client=populated_index,
            top_k=1,
        )
        assert len(results) == 1
        hit = results[0]
        assert "smiles" in hit
        assert "score" in hit
        assert "molecular_weight" in hit
        assert "logp" in hit
        assert "toxicity_score" in hit
        assert "tanimoto_score" in hit
        assert "fused_score" in hit

    def test_self_search_returns_score_near_1(self, populated_index, embedder):
        """Searching for an indexed molecule should return itself with score ~1.0."""
        results = search_similar_molecules(
            query_smiles="CC(=O)Oc1ccccc1C(=O)O",
            embedder=embedder,
            client=populated_index,
            top_k=1,
        )
        assert results[0]["score"] >= 0.99

    def test_search_with_mw_filter(self, populated_index, embedder):
        """Filter by MW should reduce results."""
        results = search_similar_molecules(
            query_smiles="CC(=O)Oc1ccccc1C(=O)O",
            embedder=embedder,
            client=populated_index,
            top_k=10,
            mw_max=155.0,
        )
        for hit in results:
            assert hit["molecular_weight"] <= 155.0

    def test_search_with_logp_filter(self, populated_index, embedder):
        results = search_similar_molecules(
            query_smiles="CC(=O)Oc1ccccc1C(=O)O",
            embedder=embedder,
            client=populated_index,
            top_k=10,
            logp_max=1.5,
        )
        for hit in results:
            assert hit["logp"] <= 1.5

    def test_search_with_toxicity_filter(self, populated_index, embedder):
        results = search_similar_molecules(
            query_smiles="CC(=O)Oc1ccccc1C(=O)O",
            embedder=embedder,
            client=populated_index,
            top_k=10,
            toxicity_max=0.3,
        )
        for hit in results:
            assert hit["toxicity_score"] is not None
            assert hit["toxicity_score"] <= 0.3

    def test_search_with_strict_filter_returns_empty(self, populated_index, embedder):
        """Very strict filter should return no results."""
        results = search_similar_molecules(
            query_smiles="CC(=O)Oc1ccccc1C(=O)O",
            embedder=embedder,
            client=populated_index,
            top_k=10,
            mw_max=1.0,
        )
        assert len(results) == 0
