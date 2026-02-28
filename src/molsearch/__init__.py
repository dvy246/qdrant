"""
molsearch - Molecular similarity search with ChemBERTa and Qdrant.

Provides SMILES validation, ChemBERTa embedding, Qdrant indexing,
and similarity search for drug discovery workflows.
"""

from __future__ import annotations

__version__ = "1.0.0"

from molsearch.embedder import MoleculeEmbedder
from molsearch.molecule_processor import process_smiles_batch, validate_and_canonicalize
from molsearch.qdrant_indexer import search_similar_molecules

__all__ = [
    "MoleculeEmbedder",
    "process_smiles_batch",
    "search_similar_molecules",
    "validate_and_canonicalize",
]
