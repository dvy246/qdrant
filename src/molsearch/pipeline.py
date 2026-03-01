"""
pipeline.py

End-to-end molecular similarity search pipeline.
SMILES -> RDKit -> ChemBERTa embeddings -> Qdrant -> search results.
"""

from __future__ import annotations

import logging

import numpy as np

from molsearch.config import COLLECTION_NAME, SAMPLE_SMILES
from molsearch.embedder import MoleculeEmbedder
from molsearch.molecule_processor import process_smiles_batch
from molsearch.qdrant_indexer import (
    create_collection,
    create_payload_indexes,
    get_qdrant_client,
    search_similar_molecules,
    upsert_molecules,
)

logger = logging.getLogger(__name__)


def main():
    """Run the full molecule search pipeline."""

    logger.info("Processing SMILES...")
    molecules = process_smiles_batch(SAMPLE_SMILES)
    logger.info("  %d valid molecules.\n", len(molecules))

    logger.info("Generating ChemBERTa embeddings...")
    embedder = MoleculeEmbedder()
    smiles_list = [m["smiles"] for m in molecules]
    embeddings = embedder.embed(smiles_list)
    logger.info("  Embedding matrix shape: %s", embeddings.shape)
    logger.info("  Vector norm (first): %.4f\n", np.linalg.norm(embeddings[0]))

    logger.info("Indexing in Qdrant...")
    client = get_qdrant_client()
    create_collection(client)
    upsert_molecules(client, molecules, embeddings)
    create_payload_indexes(client)

    info = client.get_collection(collection_name=COLLECTION_NAME)
    logger.info("  Collection: %s, %d points\n", info.status, info.points_count)

    logger.info("Running similarity search...")

    queries = [
        ("CC(=O)Oc1ccccc1C(=O)O", "Aspirin"),
        ("COc1ccc2cc(CC(C)C(=O)O)ccc2c1", "Naproxen"),
    ]

    for query_smi, query_name in queries:
        logger.info("\n  Query: %s (%s)", query_name, query_smi)
        hits = search_similar_molecules(
            query_smiles=query_smi,
            embedder=embedder,
            client=client,
            top_k=5,
        )

        print(f"  {'Rank':<6}{'SMILES':<50}{'Score':<10}{'MW':<10}{'LogP'}")
        print(f"  {'-' * 80}")
        for i, hit in enumerate(hits, 1):
            print(
                f"  {i:<6}{hit['smiles']:<50}{hit['score']:<10}"
                f"{hit['molecular_weight']:<10}{hit['logp']}"
            )

    logger.info("\n\nFiltered search (MW < 200)...")
    hits = search_similar_molecules(
        query_smiles="CC(=O)Oc1ccccc1C(=O)O",
        embedder=embedder,
        client=client,
        top_k=5,
        mw_max=200.0,
    )

    print(f"\n  {'Rank':<6}{'SMILES':<50}{'Score':<10}{'MW':<10}{'LogP'}")
    print(f"  {'-' * 80}")
    for i, hit in enumerate(hits, 1):
        print(
            f"  {i:<6}{hit['smiles']:<50}{hit['score']:<10}"
            f"{hit['molecular_weight']:<10}{hit['logp']}"
        )

    logger.info("\nPipeline complete.")


if __name__ == "__main__":
    main()
