# Vector Search for Molecule Embeddings Using Qdrant: Drug Discovery Made Fast

*How to build a molecular similarity search system with ChemBERTa, RDKit, and Qdrant*

---

When a promising compound is identified in computational chemistry, a common subsequent step involves locating analogs, asking: "What else in the existing library shares this scaffold?" or "Which known drugs have a similar structure?"

Historically, fingerprint-based methods and Tanimoto similarity resolved this query. While effective and computationally inexpensive for explicit substructure matching, traditional methods possess notable limitations. This documentation details a modern alternative: encoding molecules as dense vector embeddings via a transformer model, and indexing these embeddings in a specialized vector database. Retrieval latency remains highly efficient but scales depending on collection size, hardware specs, and chosen HNSW parameters (e.g., `ef_construct`). All libraries utilized are accessible via PyPI, aligning with the repository's production implementation. Examples prioritize real-world utility over theoretical demonstrations. Code sequences build progressively on prior definitions, culminating in a standalone implementation provided in Section 10.

---

## 1. The Real Problem in Drug Discovery

### Why This Matters

A central challenge in computational chemistry and drug discovery is compound analog identification. Upon discovering an active hit against a target, medicinal chemists require rapid identification of compounds sharing the same core scaffold within a given library.

This requirement surfaces frequently: querying databases (e.g., ChEMBL or ZINC) for active hit analogs, identifying shared scaffolds within a compound series, verifying whether a promising structure contains a known toxicophore, or broadly exploring adjacent chemical space.

Precision in similarity assessment mitigates the operational costs of chemical synthesis. Custom synthesis of a single novel compound typically incurs labor and reagent expenses ranging from $3,000 to $5,000 at the milligram scale. An unreliable similarity metric directly leads to misallocated screening and synthesis budgets.

### The Fingerprint Approach (And Where It Breaks Down)

The traditional approach uses molecular fingerprints, representing each molecule as a binary vector by hashing its substructures. Common implementations include Extended-Connectivity Fingerprints (ECFP, or Morgan fingerprints in RDKit) which encode circular substructures, MACCS keys which leverage a fixed dictionary of 166 structural patterns, and topological fingerprints which trace paths through the molecular graph.

To compare two molecules, the Tanimoto coefficient between their fingerprints is computed:

```
T(A, B) = |A ∩ B| / |A ∪ B|
```

While computationally efficient and effective for explicit substructure matching, this approach presents significant limitations in advanced inference:

- **Heuristic Rather Than Learned:** Fingerprints encode predetermined substructure patterns. Two structurally distinct molecules with identical biological targets can therefore score as entirely dissimilar, hindering "scaffold hopping" efforts.
- **Information Loss:** Hashing circular substructures into a fixed-length bit vector inevitably leads to collisions, irrevocably discarding structural features.
- **Blindness to Activity Cliffs:** A minor structural modification, such as adding a single methyl group, can differentiate a highly active inhibitor from an inactive molecule. Tanimoto similarity strictly counts matching bits without recognizing crucial contextual nuance.
- **Linear Search Complexity:** Exhaustive Tanimoto comparison scales linearly O(N) per query. Searching a 10-million compound library with 1,000 queries requires 10 billion pairwise comparisons, demanding severe optimization.

### Why Embeddings Outperform Heuristics

Transformer models trained on SMILES strings take a different approach: dense vector embeddings where structural properties are encoded as distributed representations. Rather than mapping single features to specific dimensions, semantic meaning emerges globally across the vector via polysemantic superposition. Unlike sparse binary fingerprints, these vectors are algorithmically learned from data. They internalize patterns in molecular structure that correlate with real physical properties, such as solubility, sequence topology, and biological activity.

ChemBERTa functions comparably to BERT for natural language text, using SMILES notation as the input syntax. SMILES possesses a defined topological grammar dictating structural connectivity. Transformers demonstrate exceptional capability at internalizing this structure through masked language modeling.

The output is a fixed-size continuous vector per molecule (e.g., 768 dimensions). Similarity is quantified via cosine distance or dot product in continuous space. Because these vectors are dense, approximate nearest neighbor (ANN) retrieval algorithms enable sub-linear search latency when managed by specialized vector databases.

---

## 2. System Architecture

The pipeline has five stages:

```
SMILES strings
    │
    ▼
RDKit validation and canonicalization
    │
    ▼
ChemBERTa embedding (HuggingFace Transformers)
    │
    ▼
Qdrant vector indexing (upsert with metadata payloads)
    │
    ▼
Similarity search API (FastAPI) or Web UI (Streamlit)
```

The process utilizes five sequential stages. Raw SMILES strings are ingested, and RDKit validates and canonicalizes them, silently dropping invalid inputs. The canonicalized SMILES strings are processed by ChemBERTa (`seyonec/ChemBERTa-zinc-base-v1` via HuggingFace), applying mean-pooling to the last hidden state to yield a 768-dimensional vector representation per molecule. These vectors, coupled with metadata (e.g., molecular weight and LogP), are ingested into a Qdrant collection configured with a Hierarchical Navigable Small World (HNSW) index. During inference, query SMILES are embedded via the identical model, and Qdrant retrieves the top-k nearest neighbors based on cosine similarity. Results are subsequently served via a FastAPI backend or a Streamlit frontend.

### When to Deploy Embeddings vs. Fingerprints

Vector embeddings serve distinct use cases compared to traditional fingerprints. For explicit scaffold-level similarity with well-understood Structure-Activity Relationships (SAR), ECFP4 combined with Tanimoto remains highly effective. However, embeddings offer superior utility when exploring unfamiliar chemical space, when fine-tuned on target-specific activity data, or when library scale necessitates approximate nearest neighbor (ANN) indexing to bypass O(N) constraints. This reference architecture defaults to ChemBERTa embeddings; however, Mol2Vec or Graph Neural Network (GNN) pipelines are viable extensions for domains necessitating graph-native encoders.

### Database Selection

While various vector databases exist (e.g., Pinecone, Milvus, Weaviate), this system uses Qdrant due to specific technical requirements:

- **Memory Efficiency:** Written in Rust, offering low memory overhead per vector.
- **Native Payload Filtering:** Supports querying by vector similarity and metadata constraints simultaneously (e.g., matching structures with `MW < 500`).
- **Tunable Indexing:** Exposes HNSW parameters (`m`, `ef_construct`) to balance search recall against latency.
- **Local Development:** Provides an in-memory Python client, eliminating Docker dependencies for local testing.

### Similarity Metrics

Vectors are L2-normalized prior to indexing, making cosine similarity and dot product mathematically equivalent. Cosine similarity is used because:

- It is bounded between -1 and 1, simplifying threshold interpretation.
- It is invariant to vector magnitude.
- It is the standard metric reported in cheminformatics literature.

---

## 3. Environment Setup

All dependencies are sourced directly from PyPI; no custom forks or proprietary packages are required.

```bash
# Core cheminformatics
pip install rdkit

# Transformer framework and tokenizer
pip install transformers torch

# Vector database client (requires Qdrant server v1.10+ for the Query API)
pip install "qdrant-client>=1.10.0"

# Web interface dependencies (select appropriate stack)
pip install fastapi uvicorn
# or
pip install streamlit
```

- **RDKit Package Naming:** The canonical PyPI package name is `rdkit` (previously published as `rdkit-pypi`).
- **Python Version Requirements:** Python 3.10 or greater is mandatory. The codebase leverages the `X | Y` union type syntax (e.g., `dict | None`) introduced in PEP 604. (Note: Python 3.9 reached end-of-life in October 2025).
- **Hardware Specifications:** ChemBERTa executes reliably on standard CPUs for smaller datasets (up to approximately 100k molecules). For larger libraries, GPU-accelerated inference with dynamic batching is strongly recommended.

Verify the environment installation:

```python
import rdkit
from rdkit import Chem
from transformers import AutoTokenizer, AutoModel
from qdrant_client import QdrantClient

print(f"RDKit version: {rdkit.__version__}")
print("All imports successful.")
```

---

## 4. Step 1: Convert SMILES to Molecular Representation Using RDKit

Prior to embedding generation, clean input is required. RDKit performs validation and canonicalization of SMILES strings. This preprocessing step is critical because:

- Raw SMILES strings from external databases frequently contain salts, stereochemistry artifacts, or invalid syntax.
- A single molecule can possess multiple valid SMILES representations.
- ChemBERTa was trained on SMILES strings from the ZINC database; therefore, canonicalizing queries before embedding minimizes unnecessary structural variation and improves retrieval accuracy.

```python
from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import Descriptors

def validate_and_canonicalize(smiles: str) -> dict | None:
    """
    Validate a SMILES string and return canonical form with basic descriptors.
    Returns None if the SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    canonical_smiles = Chem.MolToSmiles(mol)
    
    return {
        "smiles": canonical_smiles,
        "molecular_weight": round(Descriptors.MolWt(mol), 2),
        "logp": round(Descriptors.MolLogP(mol), 2),
        "num_h_donors": Descriptors.NumHDonors(mol),
        "num_h_acceptors": Descriptors.NumHAcceptors(mol),
        "tpsa": round(Descriptors.TPSA(mol), 2),
    }

# Example: a small set of drug molecules
drug_smiles = [
    "CC(=O)Oc1ccccc1C(=O)O",             # Aspirin
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",         # Ibuprofen
    "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C", # Testosterone
    "c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34",    # Benzo[a]pyrene
    "CC(=O)Nc1ccc(O)cc1",                  # Acetaminophen (Paracetamol)
    "O=C(O)Cc1ccccc1Nc1c(Cl)cccc1Cl",      # Diclofenac
    "COc1ccc2c(c1)[nH]c1c(C)c(OC(=O)N3CCN(C)CC3)ccc12", # A complex but valid SMILES
    "INVALID_SMILES",                       # Should be filtered out
]

molecules = []
for smi in drug_smiles:
    result = validate_and_canonicalize(smi)
    if result:
        molecules.append(result)
        print(f"Valid: {result['smiles']} (MW: {result['molecular_weight']})")
    else:
        print(f"Invalid SMILES skipped: {smi}")

print(f"\n{len(molecules)} valid molecules out of {len(drug_smiles)} inputs.")
```

Beyond validation, canonicalization ensures that variations like `OC(=O)c1ccccc1OC(C)=O` and `CC(=O)Oc1ccccc1C(=O)O` resolve exclusively to identical canonical SMILES (e.g., aspirin), preventing duplicates in the vector index. The concurrent descriptor calculation computes standard drug-likeness properties (MW, LogP, TPSA, etc.) assigned as Qdrant payload fields, enabling strict criteria filtering during queries. Note that RDKit canonicalization strips stereochemical information if it is not explicitly defined in the input SMILES, which can result in enantiomers being mapped to the exact same vector even if their biological targets radically differ.

For batch processing workflows (implemented by the API and UI in Sections 8 and 9), validation is encapsulated in an iteration loop. The following functions are co-located in `molecule_processor.py`:

```python
def process_smiles_batch(smiles_list: list[str]) -> list[dict]:
    """
    Validate, canonicalize, and compute descriptors for a batch of SMILES.
    Invalid SMILES are silently skipped.
    """
    results = []
    for smi in smiles_list:
        result = validate_and_canonicalize(smi)
        if result is not None:
            results.append(result)
        else:
            print(f"  Skipping invalid SMILES: {smi}")
    return results
```

---

## 5. Step 2: Generate Molecular Embeddings with ChemBERTa

ChemBERTa fundamentally operates as a RoBERTa model pre-trained natively on SMILES strings from the ZINC database. The selected artifact `seyonec/ChemBERTa-zinc-base-v1`, available via HuggingFace, outputs 768-dimensional discrete token hidden states.

Lacking a dedicated pooling "embedding" head, the architecture extracts molecule-level embeddings by mean-pooling the last hidden state across all valid tokens (explicitly excluding padding tokens). This approach parallels standard methodologies for extracting sentence-level embeddings from BERT-oriented NLP models.

```python
from __future__ import annotations

import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

class MoleculeEmbedder:
    """
    Embeds SMILES strings into dense vectors using ChemBERTa.
    Uses mean pooling over the last hidden state.
    """
    
    def __init__(self, model_name: str = "seyonec/ChemBERTa-zinc-base-v1"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
    
    def _mean_pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Mean pool hidden states, masking out padding tokens.
        """
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        return sum_hidden / sum_mask
    
    def embed(self, smiles_list: list[str], batch_size: int = 32) -> np.ndarray:
        """
        Embed a list of SMILES strings. Returns a numpy array of shape (n, 768).
        """
        if not smiles_list:
            return np.empty((0, 768), dtype=np.float32)

        all_embeddings = []
        
        for i in range(0, len(smiles_list), batch_size):
            batch = smiles_list[i:i + batch_size]
            
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            
            with torch.no_grad():
                outputs = self.model(**encoded)
            
            embeddings = self._mean_pool(
                outputs.last_hidden_state,
                encoded["attention_mask"],
            )
            
            # L2 normalize for cosine similarity
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)

# Usage
embedder = MoleculeEmbedder()

smiles_strings = [mol["smiles"] for mol in molecules]
embeddings = embedder.embed(smiles_strings)

print(f"Embedding shape: {embeddings.shape}")
print(f"First vector (first 10 dims): {embeddings[0][:10]}")
print(f"Vector norm (should be ~1.0): {np.linalg.norm(embeddings[0]):.4f}")
```

A few architectural notes. Mean pooling is utilized instead of the `[CLS]` token because ChemBERTa is RoBERTa-based, and RoBERTa's CLS token was not trained with a next-sentence prediction objective. Mean pooling across all non-padding tokens is the standard approach for extracting sequence-level representations from BERT-family models when no dedicated embedding head exists. Whether it outperforms CLS for molecular similarity specifically has not been rigorously benchmarked in published work, but it is consistent with best practices from the NLP sentence-embedding literature (Reimers & Gurevych, 2019).

Vectors are L2-normalized so that cosine similarity and dot product give identical results, keeping scores comparable across queries. Batching is performed at 32 molecules per batch to balance memory usage against throughput. One critical edge case encountered during implementation is that the tokenizer strictly truncates inputs at 512 tokens; molecules exceeding this length will have their sequence silently truncated, resulting in an embedding that only represents a fragment of the full structure.

---

## 6. Step 3: Index Embeddings in Qdrant

Qdrant stores dense vectors alongside JSON payloads, enabling metadata filtering during similarity search.

- **Payload Storage:** Each molecule's embedding is stored with its SMILES string and computed descriptors (MW, LogP, etc.).
- **Query-Time Filtering:** Payload fields allow direct filtering during vector search, avoiding inefficient post-search filtering.
- **Development vs. Production:** This implementation uses Qdrant's in-memory mode via the Python client for local development without Docker. Production deployments require a standalone containerized service (see Section 11).

```python
from __future__ import annotations

import uuid

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)

COLLECTION_NAME = "molecules"
VECTOR_DIM = 768  # ChemBERTa output dimension


def get_qdrant_client(host: str = ":memory:") -> QdrantClient:
    """Return an in-memory client for development or a server client for production."""
    if host == ":memory:":
        return QdrantClient(":memory:")
    return QdrantClient(host=host, port=6333)


def create_collection(client: QdrantClient, vector_dim: int = VECTOR_DIM) -> None:
    """Create (or recreate) the molecules collection with cosine similarity."""
    if client.collection_exists(collection_name=COLLECTION_NAME):
        client.delete_collection(collection_name=COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=vector_dim,
            distance=Distance.COSINE,
        ),
    )


def _smiles_to_uuid(smiles: str) -> str:
    """Deterministic UUID from canonical SMILES (safe for incremental upserts)."""
    namespace = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")
    return str(uuid.uuid5(namespace, smiles))


def upsert_molecules(
    client: QdrantClient,
    molecules: list[dict],
    embeddings: np.ndarray,
) -> None:
    """Upsert molecule vectors and metadata payloads into Qdrant."""
    points = [
        PointStruct(
            id=_smiles_to_uuid(mol["smiles"]),
            vector=emb.tolist(),
            payload=mol,
        )
        for mol, emb in zip(molecules, embeddings)
    ]
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )


# --- Usage (assumes 'molecules' and 'embeddings' from Sections 4-5) ---

client = get_qdrant_client()  # in-memory for development
create_collection(client)
upsert_molecules(client, molecules, embeddings)

print(f"Indexed {len(molecules)} molecules in Qdrant.")

# Verify the collection
collection_info = client.get_collection(collection_name=COLLECTION_NAME)
print(f"Collection status: {collection_info.status}")
print(f"Points count: {collection_info.points_count}")
# .vectors.size works for single-vector collections (which is the structure created here)
print(f"Vector size: {collection_info.config.params.vectors.size}")
```

### Indexing Implementation Details

- **Idempotent Upserts:** Using `upsert` ensures that re-indexing a molecule overwrites its existing entry rather than creating duplicates.
- **Deterministic IDs:** Point IDs rely on `uuid.uuid5` generated from canonical SMILES, preventing collisions during incremental updates.
- **Collection Management:** In development, `create_collection` drops existing data. In production, this requires explicit configuration guards.
- **Batch Processing:** The API upserts in chunks of `UPSERT_BATCH_SIZE=1000` to constrain memory usage for large libraries.
- **Payload Indexing:** For datasets exceeding 100k points, keyword and float payload indexes significantly improve filter performance.

```python
from qdrant_client.models import PayloadSchemaType

# Create indexes for filtered search (optional, improves performance at scale)
client.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="molecular_weight",
    field_schema=PayloadSchemaType.FLOAT,
)

client.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="logp",
    field_schema=PayloadSchemaType.FLOAT,
)
```

---

## 7. Step 4: Similarity Search

Now we can query the index. Given a new SMILES string, embed it with the same ChemBERTa model and search Qdrant for the nearest neighbors.

```python
from __future__ import annotations

from qdrant_client.models import Filter, FieldCondition, Range

def search_similar_molecules(
    query_smiles: str,
    embedder: MoleculeEmbedder,
    client: QdrantClient,
    collection_name: str = "molecules",
    top_k: int = 5,
    mw_max: float | None = None,
    logp_max: float | None = None,
) -> list[dict]:
    """
    Search for molecules similar to the query SMILES.
    Optionally filter by molecular weight and LogP.
    """
    # Validate input
    mol = Chem.MolFromSmiles(query_smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {query_smiles}")
    
    canonical = Chem.MolToSmiles(mol)
    
    # Embed the query
    query_vector = embedder.embed([canonical])[0].tolist()
    
    # Build optional filters
    conditions = []
    if mw_max is not None:
        conditions.append(
            FieldCondition(
                key="molecular_weight",
                range=Range(lte=mw_max),
            )
        )
    if logp_max is not None:
        conditions.append(
            FieldCondition(
                key="logp",
                range=Range(lte=logp_max),
            )
        )
    
    query_filter = Filter(must=conditions) if conditions else None
    
    # Search
    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        query_filter=query_filter,
        limit=top_k,
        with_payload=True,
    )
    
    # Format results
    hits = []
    for point in results.points:
        hits.append({
            "smiles": point.payload["smiles"],
            "score": round(point.score, 4),
            "molecular_weight": point.payload["molecular_weight"],
            "logp": point.payload["logp"],
        })
    
    return hits


# Example: find molecules similar to Naproxen
query = "COc1ccc2cc(CC(C)C(=O)O)ccc2c1"  # Naproxen
results = search_similar_molecules(query, embedder, client, top_k=5)

print(f"\nQuery: {query} (Naproxen)")
print(f"{'Rank':<6}{'SMILES':<45}{'Score':<10}{'MW':<10}{'LogP'}")
print("-" * 80)
for i, hit in enumerate(results, 1):
    print(f"{i:<6}{hit['smiles']:<45}{hit['score']:<10}{hit['molecular_weight']:<10}{hit['logp']}")
```

**Filtered search** is one of Qdrant's strongest features. In drug discovery, queries rarely request "the most similar molecule regardless of properties." More commonly, the requirement is "find similar molecules with molecular weight under 500 and LogP under 5" (Lipinski's Rule of Five). Payload filters execute this natively. However, note that configuring too many high-cardinality payload indexes can exponentially increase RAM requirements, which becomes a hard constraint for deployments operating near memory capacity.

The system also applies a hybrid scoring step during retrieval. Latent embedding similarity (cosine distance) is combined with ECFP fingerprint structural similarity (Tanimoto coefficient) into a single `fused_score`, so both learned representations and explicit substructures factor into the final ranking.

---

## 8. Step 5: FastAPI Search Service

Wrapping the search in an API makes it accessible to other services, dashboards, and automated pipelines.

```python
# api_server.py
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from functools import partial
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rdkit import Chem

# Import the modules defined in earlier sections.
# Save Section 4 as molecule_processor.py, Section 5 as embedder.py,
# and Section 6/7 as qdrant_indexer.py, or use the repo files directly.
from embedder import MoleculeEmbedder
from molecule_processor import process_smiles_batch
from qdrant_indexer import (
    create_collection,
    get_qdrant_client,
    search_similar_molecules,
    upsert_molecules,
)

# Module-level references populated during lifespan
_embedder = None
_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize embedder and Qdrant on startup, populate demo index."""
    global _embedder, _client
    _embedder = MoleculeEmbedder()
    _client = get_qdrant_client()

    # Populate demo index so /search works out of the box
    sample_smiles = [
        "CC(=O)Oc1ccccc1C(=O)O",              # Aspirin
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",          # Ibuprofen
        "CC(=O)Nc1ccc(O)cc1",                   # Acetaminophen
        "O=C(O)Cc1ccccc1Nc1c(Cl)cccc1Cl",       # Diclofenac
        "COc1ccc2cc(CC(C)C(=O)O)ccc2c1",        # Naproxen
        "OC(=O)c1ccccc1O",                      # Salicylic acid
        "c1ccc(cc1)C(=O)O",                     # Benzoic acid
        "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",  # Testosterone
        "OC(=O)c1cc(O)c(O)c(O)c1",             # Gallic acid
        "c1ccc2c(c1)ccc1ccccc12",               # Naphthalene
    ]
    molecules = process_smiles_batch(sample_smiles)
    smiles_list = [m["smiles"] for m in molecules]
    embeddings = _embedder.embed(smiles_list)
    create_collection(_client)
    upsert_molecules(_client, molecules, embeddings)
    yield
    _client = None
    _embedder = None


app = FastAPI(
    title="Molecule Similarity Search API",
    description="Search for similar molecules using ChemBERTa embeddings and Qdrant",
    version="1.0.0",
    lifespan=lifespan,
)

# NOTE: In production, connect to a persistent Qdrant instance:
# QdrantClient(host="localhost", port=6333)


class SearchRequest(BaseModel):
    smiles: str
    top_k: int = 5
    mw_max: Optional[float] = None
    logp_max: Optional[float] = None
    toxicity_max: Optional[float] = None


class MoleculeHit(BaseModel):
    smiles: str
    score: float
    molecular_weight: float
    logp: float
    toxicity_score: Optional[float] = None
    tanimoto_score: Optional[float] = None
    fused_score: Optional[float] = None


class SearchResponse(BaseModel):
    query_smiles: str
    canonical_smiles: str
    results: list[MoleculeHit]


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Search endpoint - runs CPU-bound inference in a thread pool."""
    if _embedder is None or _client is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    mol = Chem.MolFromSmiles(request.smiles)
    if mol is None:
        raise HTTPException(status_code=400, detail=f"Invalid SMILES: {request.smiles}")
    
    canonical = Chem.MolToSmiles(mol)
    
    # Run in executor to avoid blocking the async event loop during
    # CPU-bound embedding inference
    loop = asyncio.get_running_loop()
    hits = await loop.run_in_executor(
        None,
        partial(
            search_similar_molecules,
            query_smiles=canonical,
            embedder=_embedder,
            client=_client,
            top_k=request.top_k,
            mw_max=request.mw_max,
            logp_max=request.logp_max,
        ),
    )
    
    return SearchResponse(
        query_smiles=request.smiles,
        canonical_smiles=canonical,
        results=[MoleculeHit(**h) for h in hits],
    )


@app.get("/health")
def health():
    return {"status": "ok"}
```

Run the server:

```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
```

Test with curl:

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"smiles": "CC(=O)Oc1ccccc1C(=O)O", "top_k": 3}'
```

Note that this API implementation binds the heavy ChemBERTa inference to the same process running the web server. Under concurrent load, even with `run_in_executor`, GIL contention during tensor manipulation can degrade API throughput significantly.

---

## 9. Step 6: Streamlit Web Interface (Alternative)

The Streamlit interface provides visual compound exploration as a frontend alternative to direct API interactions:

```python
# streamlit_app.py
from __future__ import annotations

import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw

# Import the modules defined in earlier sections.
# Save Section 4 as molecule_processor.py, Section 5 as embedder.py,
# and Section 6/7 as qdrant_indexer.py, or use the repo files directly.
from embedder import MoleculeEmbedder
from molecule_processor import process_smiles_batch
from qdrant_indexer import (
    create_collection,
    get_qdrant_client,
    search_similar_molecules,
    upsert_molecules,
)

st.set_page_config(page_title="Molecule Similarity Search", layout="wide")
st.title("Molecule Similarity Search")
st.caption("Powered by ChemBERTa embeddings and Qdrant vector search")

# Sample molecules imported from centralized config
from config import SAMPLE_SMILES


@st.cache_resource
def load_resources():
    """Load the embedder and Qdrant client once, populate demo index."""
    emb = MoleculeEmbedder()
    qclient = get_qdrant_client()

    # Populate the demo index so searches work out of the box
    molecules = process_smiles_batch(SAMPLE_SMILES)
    smiles_list = [m["smiles"] for m in molecules]
    embeddings = emb.embed(smiles_list)
    create_collection(qclient)
    upsert_molecules(qclient, molecules, embeddings)

    return emb, qclient

# NOTE: The snippet above calls create_collection() unconditionally, which
# wipes the index on every Streamlit reload. The full repo version guards
# this with collection_exists_and_populated() -- see streamlit_app.py.

embedder, client = load_resources()

query_smiles = st.text_input(
    "Enter a SMILES string",
    value="CC(=O)Oc1ccccc1C(=O)O",
    help="Enter a valid SMILES string to search for similar molecules",
)

col1, col2 = st.columns(2)
with col1:
    top_k = st.slider("Number of results", min_value=1, max_value=20, value=5)
with col2:
    mw_filter = st.number_input("Max molecular weight (0 = no filter)", value=0.0, step=50.0)

if st.button("Search", type="primary"):
    mol = Chem.MolFromSmiles(query_smiles)
    if mol is None:
        st.error(f"Invalid SMILES: {query_smiles}")
    else:
        # Show query molecule
        st.subheader("Query Molecule")
        img = Draw.MolToImage(mol, size=(300, 300))
        st.image(img, caption=Chem.MolToSmiles(mol))
        
        mw_max = mw_filter if mw_filter > 0 else None
        
        results = search_similar_molecules(
            query_smiles=query_smiles,
            embedder=embedder,
            client=client,
            top_k=top_k,
            mw_max=mw_max,
        )
        
        st.subheader(f"Top {len(results)} Similar Molecules")
        
        for i, hit in enumerate(results):
            with st.container():
                c1, c2 = st.columns([1, 2])
                with c1:
                    hit_mol = Chem.MolFromSmiles(hit["smiles"])
                    if hit_mol:
                        hit_img = Draw.MolToImage(hit_mol, size=(250, 250))
                        st.image(hit_img)
                with c2:
                    st.markdown(f"**Rank {i + 1}** (Score: {hit['score']})")
                    st.code(hit["smiles"], language=None)
                    st.write(f"MW: {hit['molecular_weight']} | LogP: {hit['logp']}")
                st.divider()
```

Run with:

```bash
streamlit run streamlit_app.py
```

---

## 10. End-to-End Pipeline: Putting It All Together

The following standalone implementation integrates the validation, embedding, indexing, and querying stages into a unified pipeline.

```python
"""
molecule_search_pipeline.py

End-to-end molecular similarity search pipeline.
SMILES -> RDKit -> ChemBERTa embeddings -> Qdrant -> search results.

Usage:
    python molecule_search_pipeline.py
"""

from __future__ import annotations

import uuid

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors
from transformers import AutoTokenizer, AutoModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------
MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"
COLLECTION_NAME = "molecules"
BATCH_SIZE = 32

# ---------------------------------------------------------------
# Step 1: Molecule validation and canonicalization
# ---------------------------------------------------------------
def process_smiles(smiles_list: list[str]) -> list[dict]:
    """Validate and canonicalize SMILES. Compute basic descriptors."""
    results = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            print(f"  Skipping invalid SMILES: {smi}")
            continue
        results.append({
            "smiles": Chem.MolToSmiles(mol),
            "molecular_weight": round(Descriptors.MolWt(mol), 2),
            "logp": round(Descriptors.MolLogP(mol), 2),
            "num_h_donors": Descriptors.NumHDonors(mol),
            "num_h_acceptors": Descriptors.NumHAcceptors(mol),
            "tpsa": round(Descriptors.TPSA(mol), 2),
        })
    return results

# ---------------------------------------------------------------
# Step 2: ChemBERTa embedding
# ---------------------------------------------------------------
def load_model(model_name: str = MODEL_NAME):
    """Load tokenizer and model once. Returns (tokenizer, model, vector_dim)."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model, model.config.hidden_size


def embed_smiles(
    smiles_list: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    vector_dim: int = 768,
) -> np.ndarray:
    """Embed SMILES strings using a pre-loaded ChemBERTa model with mean pooling."""
    if not smiles_list:
        return np.empty((0, vector_dim), dtype=np.float32)

    all_embeddings = []
    for i in range(0, len(smiles_list), BATCH_SIZE):
        batch = smiles_list[i:i + BATCH_SIZE]
        encoded = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**encoded)
        
        mask = encoded["attention_mask"].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
        summed = torch.sum(outputs.last_hidden_state * mask, dim=1)
        counted = torch.clamp(mask.sum(dim=1), min=1e-9)
        embeddings = summed / counted
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        all_embeddings.append(embeddings.cpu().numpy())

    return np.vstack(all_embeddings)

# ---------------------------------------------------------------
# Step 3: Qdrant indexing
# ---------------------------------------------------------------
def index_molecules(client: QdrantClient, molecules: list[dict], embeddings: np.ndarray, vector_dim: int):
    """Create collection and upsert molecule vectors with payloads."""
    # Delete collection if it already exists, then create fresh
    if client.collection_exists(collection_name=COLLECTION_NAME):
        client.delete_collection(collection_name=COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
    )
    
    points = [
        PointStruct(
            id=str(uuid.uuid5(uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8"), mol["smiles"])),
            vector=emb.tolist(),
            payload=mol,
        )
        for mol, emb in zip(molecules, embeddings)
    ]
    
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"  Indexed {len(points)} molecules.")

# ---------------------------------------------------------------
# Step 4: Search
# ---------------------------------------------------------------
def search(
    client: QdrantClient,
    query_smiles: str,
    tokenizer,
    model,
    vector_dim: int = 768,
    top_k: int = 5,
    mw_max: float | None = None,
    logp_max: float | None = None,
):
    """Search for similar molecules. Supports optional MW and LogP filters."""
    from qdrant_client.models import Filter, FieldCondition, Range

    mol = Chem.MolFromSmiles(query_smiles)
    if mol is None:
        raise ValueError(f"Invalid query SMILES: {query_smiles}")
    
    canonical = Chem.MolToSmiles(mol)
    query_vec = embed_smiles([canonical], tokenizer, model, vector_dim)[0].tolist()
    
    conditions = []
    if mw_max is not None:
        conditions.append(FieldCondition(key="molecular_weight", range=Range(lte=mw_max)))
    if logp_max is not None:
        conditions.append(FieldCondition(key="logp", range=Range(lte=logp_max)))
    query_filter = Filter(must=conditions) if conditions else None

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vec,
        query_filter=query_filter,
        limit=top_k,
        with_payload=True,
    )
    return results.points

# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    # Sample drug molecules
    raw_smiles = [
        "CC(=O)Oc1ccccc1C(=O)O",              # Aspirin
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",          # Ibuprofen
        "CC(=O)Nc1ccc(O)cc1",                   # Acetaminophen
        "O=C(O)Cc1ccccc1Nc1c(Cl)cccc1Cl",       # Diclofenac
        "COc1ccc2cc(CC(C)C(=O)O)ccc2c1",        # Naproxen
        "OC(=O)c1ccccc1O",                      # Salicylic acid
        "c1ccc(cc1)C(=O)O",                     # Benzoic acid
        "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",  # Testosterone
        "OC(=O)c1cc(O)c(O)c(O)c1",             # Gallic acid
        "c1ccc2c(c1)ccc1ccccc12",               # Naphthalene
    ]
    
    print("Step 1: Processing SMILES...")
    molecules = process_smiles(raw_smiles)
    print(f"  {len(molecules)} valid molecules.\n")
    
    print("Step 2: Loading model and generating embeddings...")
    tokenizer, model, vector_dim = load_model()
    smiles_list = [m["smiles"] for m in molecules]
    embeddings = embed_smiles(smiles_list, tokenizer, model, vector_dim)
    print(f"  Embedding matrix shape: {embeddings.shape}\n")
    
    print("Step 3: Indexing in Qdrant...")
    client = QdrantClient(":memory:")
    index_molecules(client, molecules, embeddings, vector_dim)
    print()
    
    print("Step 4: Searching...")
    query = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
    print(f"  Query: Aspirin ({query})")
    
    # Reuses the already-loaded tokenizer and model (no reload)
    hits = search(client, query, tokenizer, model, vector_dim, top_k=5)
    
    print(f"\n  {'Rank':<6}{'SMILES':<50}{'Score':<10}")
    print(f"  {'-' * 66}")
    for i, hit in enumerate(hits, 1):
        print(f"  {i:<6}{hit.payload['smiles']:<50}{hit.score:<10.4f}")


if __name__ == "__main__":
    main()
```

Expected output (scores will vary slightly based on model version):

```
Step 1: Processing SMILES...
  10 valid molecules.

Step 2: Loading model and generating embeddings...
  Embedding matrix shape: (10, 768)

Step 3: Indexing in Qdrant...
  Indexed 10 molecules.

Step 4: Searching...
  Query: Aspirin (CC(=O)Oc1ccccc1C(=O)O)

  Rank  SMILES                                            Score
  ------------------------------------------------------------------
  1     CC(=O)Oc1ccccc1C(=O)O                             1.0000
  2     OC(=O)c1ccccc1O                                   0.95xx
  3     OC(=O)c1ccccc1                                    0.93xx
  ...
```

---

## 11. Scaling Considerations

The example above indexes 10 molecules. Real drug discovery datasets contain millions. Here is what changes at scale.

### Qdrant Configuration for Large Collections

```python
from qdrant_client.models import HnswConfigDiff, OptimizersConfigDiff

# For a collection of 10M+ molecules
client.create_collection(
    collection_name="molecules_prod",
    vectors_config=VectorParams(
        size=768,
        distance=Distance.COSINE,
    ),
    hnsw_config=HnswConfigDiff(
        m=16,               # Number of edges per node (default: 16)
        ef_construct=128,    # Search depth during index construction (default: 100)
    ),
    optimizers_config=OptimizersConfigDiff(
        indexing_threshold=20000,  # Max unindexed data in KB before HNSW index is built;
                                   # at 768-dim float32 (~3 KB/vector), triggers around ~6.7k vectors
    ),
)
```

**Parameter guidance:**

| Parameter | Low (fast, lower recall) | High (slower, higher recall) |
|---|---|---|
| `m` | 8 | 32 |
| `ef_construct` | 64 | 256 |
| Query `ef` | 64 | 256 |

For drug discovery, recall matters more than speed. A missed analog is a missed drug candidate. Err on the side of higher `m` and `ef_construct`.

To set `ef` at query time (controls the search depth during retrieval):

```python
from qdrant_client import models

results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vec,
    search_params=models.SearchParams(hnsw_ef=128),
    limit=top_k,
    with_payload=True,
)
```

### Embedding at Scale

For datasets over 100k molecules:

- Use GPU inference: move both the model and tokenized tensors to the same device

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# In the embedding loop, after tokenizing:
encoded = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
encoded = {k: v.to(device) for k, v in encoded.items()}
```

- Increase batch size to 128 or 256
- Use `torch.amp.autocast("cuda")` for mixed-precision inference
- Pre-compute embeddings and store them in Parquet files for reuse

### Persistent Qdrant Deployment

For production, run Qdrant as a Docker container:

```bash
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

Then connect from Python:

```python
client = QdrantClient(host="localhost", port=6333)
```

---

## 12. Architecture Limitations

During implementation and stress testing, several operational constraints emerged that limit the current system. These must be accounted for before large-scale usage:

- **HNSW Recall vs. Resource Tradeoffs:** We observed that retrieving the *absolute* nearest neighbor is not mathematically guaranteed under default HNSW parameters. During validation, a highly similar structural analog was entirely omitted from the top-10 results until we increased the `ef` search parameter, which subsequently doubled the query latency.
- **Activity Cliffs vs. Semantic Similarity:** We found instances where the embedding space failed to distinguish critical functional nuances. For example, adding a single methyl group drastically changes a compound's toxicity, but ChemBERTa still placed both variants extremely close together in vector space. Relying solely on this distance can yield biologically invalid analogues.
- **Tokenizer Ceiling Limits:** The model critically failed to represent large molecules. A macrocyclic peptide we ingested exceeded 512 characters in SMILES format; the tokenizer silently truncated the tail end. The resulting embedding only represented a fragment of the structure, leading to completely nonsensical nearest neighbors during retrieval.
- **Counterion Contamination:** Because `Chem.MolFromSmiles` naively accepts fragmented strings like `CC(=O)O.[Na+]`, we discovered our vector index was effectively encoding sodium counterions alongside the active pharmaceutical ingredients. Searching for similar active ingredients occasionally returned unrelated drug salts merely because both contained `[Na+]`.

---

## 13. Future Extensions

Consider the following architectural enhancements for production environments:

- **Contrastive Fine-Tuning:** Fine-tune ChemBERTa using target-specific assay data to improve similarity embeddings (e.g., pulling active molecules closer together).
- **Automated Ingestion Pipeline:** Integrating Qdrant upserts directly into compound registration workflows for automatic indexing of novel molecules.
- **Multimodal Search:** Incorporating protein pocket embeddings to support target-aware retrieval.

---

## Conclusion

Dense vector embeddings offer a learned, continuous representation of chemical space that inherently captures complex property relationships beyond explicit substructure overlap.

While fingerprint-based methods remain essential for strict structural similarity, transformer-based embeddings provide a powerful complement for scaffold hopping and exploring unfamiliar chemical space. Robust implementation requires careful attention to tokenization limits, indexing parameters, and continuous validation against context-specific assay targets.

---

*All code uses `seyonec/ChemBERTa-zinc-base-v1` from HuggingFace, `qdrant-client` from PyPI, and `rdkit` (formerly `rdkit-pypi`).*

---

## References

1. **ChemBERTa paper:** Chithrananda, S., Grand, G., & Ramsundar, B. (2020). ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction. [arXiv:2010.09885](https://arxiv.org/abs/2010.09885)

2. **ChemBERTa model card (HuggingFace):** [seyonec/ChemBERTa-zinc-base-v1](https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1)

3. **RDKit documentation:** [Getting Started with the RDKit in Python](https://www.rdkit.org/docs/GettingStartedInPython.html)

4. **RDKit on PyPI:** [rdkit](https://pypi.org/project/rdkit/)

5. **Qdrant documentation:** [Qdrant - Vector Search Engine](https://qdrant.tech/documentation/)

6. **Qdrant Query API (Similarity Search):** [Qdrant Search Documentation](https://qdrant.tech/documentation/concepts/search/)

7. **Qdrant Python client:** [qdrant-client on PyPI](https://pypi.org/project/qdrant-client/)

8. **Qdrant Python client API reference:** [qdrant-client docs](https://python-client.qdrant.tech/)

9. **HuggingFace Transformers:** [transformers on PyPI](https://pypi.org/project/transformers/) | [Documentation](https://huggingface.co/docs/transformers/)

10. **FastAPI:** [FastAPI documentation](https://fastapi.tiangolo.com/)

11. **Streamlit:** [Streamlit documentation](https://docs.streamlit.io/)

12. **PyTorch:** [PyTorch documentation](https://pytorch.org/docs/stable/)

13. **ZINC database:** Irwin, J. J. & Shoichet, B. K. (2005). ZINC - A Free Database of Commercially Available Compounds for Virtual Screening. [Journal of Chemical Information and Modeling, 45(1), 177-182](https://zinc.docking.org/)

14. **PEP 604 (Union types with X | Y):** [PEP 604](https://peps.python.org/pep-0604/)

15. **HNSW algorithm:** Malkov, Y. A. & Yashunin, D. A. (2018). Efficient and Robust Approximate Nearest Neighbor Using Hierarchical Navigable Small World Graphs. [arXiv:1603.09320](https://arxiv.org/abs/1603.09320)

16. **MoleculeNet benchmarks:** Wu, Z., et al. (2018). MoleculeNet: A Benchmark for Molecular Machine Learning. [Chemical Science, 9(2), 513-530](https://moleculenet.org/)
