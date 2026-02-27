"""
Streamlit interactive UI for molecular similarity search.

Provides a visual interface for entering SMILES queries, viewing
molecular structures, and browsing search results.

Usage:
    streamlit run src/molsearch/streamlit_app.py
"""

from __future__ import annotations

import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw

from molsearch.config import SAMPLE_SMILES
from molsearch.embedder import MoleculeEmbedder
from molsearch.molecule_processor import process_smiles_batch
from molsearch.qdrant_indexer import (
    collection_exists_and_populated,
    create_collection,
    create_payload_indexes,
    get_qdrant_client,
    search_similar_molecules,
    upsert_molecules,
)

# ---- Page config ----
st.set_page_config(page_title="Molecule Similarity Search", layout="wide")
st.title("Molecule Similarity Search")
st.caption("Powered by ChemBERTa embeddings and Qdrant vector search")


@st.cache_resource
def load_resources():
    """Load the embedder and Qdrant client once, populate demo index."""
    embedder = MoleculeEmbedder()
    client = get_qdrant_client()

    create_collection(client)
    create_payload_indexes(client)

    # Only populate if the collection is empty or missing
    if not collection_exists_and_populated(client):
        molecules = process_smiles_batch(SAMPLE_SMILES)
        smiles_list = [m["smiles"] for m in molecules]
        embeddings = embedder.embed(smiles_list)
        upsert_molecules(client, molecules, embeddings)

    return embedder, client


embedder, client = load_resources()

# ---- Sidebar controls ----
st.sidebar.header("Search Parameters")

query_smiles = st.sidebar.text_input(
    "SMILES string",
    value="CC(=O)Oc1ccccc1C(=O)O",
    help="Enter a valid SMILES string to search for similar molecules",
)

top_k = st.sidebar.slider("Number of results", min_value=1, max_value=20, value=5)

mw_filter = st.sidebar.number_input(
    "Max molecular weight (0 = no filter)",
    value=0.0,
    step=50.0,
    min_value=0.0,
)

logp_filter = st.sidebar.number_input(
    "Max LogP (0 = no filter)",
    value=0.0,
    step=0.5,
)

toxicity_filter = st.sidebar.number_input(
    "Max toxicity score (0 = no filter)",
    value=0.0,
    step=0.1,
    min_value=0.0,
)

search_clicked = st.sidebar.button("Search", type="primary", use_container_width=True)


# ---- Main content ----
if search_clicked:
    query_smiles = query_smiles.strip()
    mol = Chem.MolFromSmiles(query_smiles)

    if mol is None or mol.GetNumAtoms() == 0:
        st.error(f"Invalid SMILES: {query_smiles}")
    else:
        canonical_query = Chem.MolToSmiles(mol)

        # Display query molecule
        col_query, col_info = st.columns([1, 2])
        with col_query:
            st.subheader("Query Molecule")
            img = Draw.MolToImage(mol, size=(300, 300))
            st.image(img, caption=canonical_query)
        with col_info:
            st.subheader("Query Info")
            st.metric("Molecular Weight", f"{Descriptors.MolWt(mol):.2f}")  # type: ignore[attr-defined]
            st.metric("LogP", f"{Descriptors.MolLogP(mol):.2f}")  # type: ignore[attr-defined]

        st.divider()

        # Run search
        mw_max = mw_filter if mw_filter > 0 else None
        logp_max = logp_filter if logp_filter > 0 else None
        toxicity_max = toxicity_filter if toxicity_filter > 0 else None

        results = search_similar_molecules(
            query_smiles=canonical_query,
            embedder=embedder,
            client=client,
            top_k=top_k,
            mw_max=mw_max,
            logp_max=logp_max,
            toxicity_max=toxicity_max,
        )

        st.subheader(f"Top {len(results)} Similar Molecules")

        if not results:
            st.info("No results found. Try relaxing the filters.")
        else:
            for i, hit in enumerate(results):
                with st.container():
                    c1, c2 = st.columns([1, 2])
                    with c1:
                        hit_mol = Chem.MolFromSmiles(hit["smiles"])
                        if hit_mol:
                            hit_img = Draw.MolToImage(hit_mol, size=(250, 250))
                            st.image(hit_img)
                    with c2:
                        st.markdown(
                            f"**Rank {i + 1}** | "
                            f"Fused: **{hit['fused_score']}** "
                            f"(Tanimoto: {hit['tanimoto_score']}, Latent: {hit['score']})"
                        )
                        st.code(hit["smiles"], language=None)
                        toxicity_text = (
                            f"{hit['toxicity_score']:.3f}"
                            if isinstance(hit.get("toxicity_score"), (int, float))
                            else "n/a"
                        )
                        st.write(
                            f"MW: {hit['molecular_weight']} | "
                            f"LogP: {hit['logp']} | "
                            f"Toxicity: {toxicity_text}"
                        )
                    st.divider()
else:
    st.info("Enter a SMILES string in the sidebar and click Search to find similar molecules.")
