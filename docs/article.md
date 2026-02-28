# Vector Search for Molecule Embeddings Using Qdrant: Drug Discovery Made Fast

*How to build a molecular similarity search system with ChemBERTa, RDKit, and Qdrant*

---

If you work in computational chemistry, you have asked this question many times: "What else looks like this molecule?" A medicinal chemist finds a promising compound, and the immediate follow-up is always about analogs. What is in our library that shares the same scaffold? What known drugs have a similar structure?

For decades, the standard answer was fingerprint-based methods and Tanimoto similarity. Those tools still work, and they are not going anywhere. But they have real blind spots. This walkthrough covers a modern alternative: encoding molecules as dense vector embeddings with a transformer model, indexing them in a vector database built for this kind of search, and retrieving neighbors in milliseconds. Every library used here is on PyPI, and the examples are aligned with the current repository implementation and cited references. The goal is real-world use, not just a notebook demo. Code snippets build on definitions from earlier sections, and Section 10 has the full standalone script you can copy and run.

---

## 1. The Real Problem in Drug Discovery

### Why This Matters

Every drug discovery program, at some point, comes down to the same question: "What else looks like this?" You find a compound that hits your target. Great. Now you need analogs. You need to know what's in your library that shares the same core scaffold, and you need to know fast.

That question shows up everywhere. You're searching ChEMBL or ZINC for analogs of an active hit. You're looking at a compound series and trying to spot shared scaffolds. You've got a structure that looks promising but also looks suspiciously close to a known toxicophore, and you want to check. Or you're just exploring around a hit, trying to see what's nearby in chemical space.

Getting this right matters because synthesis isn't cheap. Putting a molecule into a synthesis queue costs somewhere between $2,000 and $50,000 depending on complexity. If your similarity metric is unreliable, you're burning money on the wrong molecules.

### The Fingerprint Approach (And Where It Breaks Down)

The traditional way to do this is molecular fingerprints. You represent each molecule as a binary vector by hashing its substructures. There are a few flavors: ECFP (also called Morgan fingerprints in RDKit) encodes circular substructures around each atom, MACCS keys use a fixed dictionary of 166 structural patterns, and topological fingerprints trace paths through the molecular graph.

To compare two molecules, you compute the Tanimoto coefficient between their fingerprints:

```
T(A, B) = |A ∩ B| / |A ∪ B|
```

Simple, fast, and honestly? It works pretty well for a lot of cases. But there are problems you can't ignore once you start relying on it seriously.

First, fingerprints don't learn anything. They encode whatever substructure patterns were baked in at design time. So two molecules that look completely different structurally but hit the same biological target can score as dissimilar. That's a real problem if you're trying to hop scaffolds.

Second, you lose information. When you hash circular substructures into a fixed-length bit vector, collisions happen. Some of your molecular information just disappears.

Third, activity cliffs are invisible. A single methyl group can be the difference between a 1 nM inhibitor and a completely inactive molecule. Tanimoto doesn't know that. It just counts matching bits.

And finally, scaling is painful. Tanimoto comparison is O(n) per query. If you've got 10 million compounds and 1,000 queries, that's 10 billion pairwise comparisons. You can precompute and optimize, but at some point you're fighting the math.

### Why Embeddings Are Worth Trying

Transformer models trained on SMILES strings give you something qualitatively different: dense vector embeddings where every dimension carries information. Unlike sparse binary fingerprints, these vectors are learned from data. They pick up on patterns in molecular structure that correlate with real properties like solubility, toxicity, and biological activity.

ChemBERTa works exactly like BERT does for English text, except the "language" is SMILES notation. SMILES has a grammar, it has rules for how atoms and bonds get expressed, and transformers turn out to be very good at internalizing that structure through masked language modeling.

What you get out is a fixed-size vector per molecule (768 dimensions in our case). You measure similarity with cosine distance or dot product in that continuous space. And because these are dense vectors, you can throw approximate nearest neighbor algorithms at them and search in sub-linear time. That's where vector databases come in.

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

The whole thing boils down to five steps. Raw SMILES come in, RDKit validates and canonicalizes them (anything invalid gets dropped). Those clean SMILES go through ChemBERTa (`seyonec/ChemBERTa-zinc-base-v1` from HuggingFace) where we mean-pool the last hidden state to get a 768-dim vector per molecule. The vectors, along with metadata like molecular weight and LogP, get stored in a Qdrant collection with an HNSW index. When a query comes in, we embed it with the same model and ask Qdrant for the top-k nearest neighbors by cosine similarity. Results come back through either a FastAPI endpoint or a Streamlit app.

### When to Use Embeddings vs. Fingerprints

I don't want to oversell embeddings. For scaffold-level similarity where you've got well-understood SAR, ECFP4 plus Tanimoto is still hard to beat. But embeddings are worth it when you're exploring unfamiliar chemical space, when you've fine-tuned on your target's activity data, or when your library is big enough that ANN indexing actually saves time over brute-force comparison. This setup defaults to ChemBERTa embeddings, while Mol2Vec/GNN pipelines are practical extensions for teams that need graph-native encoders.

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

Everything here comes from PyPI. No custom forks, no made-up packages.

```bash
# Core cheminformatics
pip install rdkit

# Transformer model and tokenizer
pip install transformers torch

# Vector database client (requires Qdrant server v1.10+ for the Query API)
pip install "qdrant-client>=1.10.0"

# Web interface (choose one)
pip install fastapi uvicorn
# or
pip install streamlit
```

**A note on naming:** The package used to be published as `rdkit-pypi`, but the current name on PyPI is just `rdkit`.

**Python version:** You need 3.10 or later. The `X | Y` union type syntax used throughout (e.g., `dict | None`) is a Python 3.10 feature (PEP 604). Python 3.9 reached end-of-life in October 2025.

**Hardware:** ChemBERTa runs fine on CPU for small to medium datasets (up to about 100k molecules). For larger libraries, you will want GPU inference with batching.

Verify the installation:

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

Before we can embed anything, we need clean input. RDKit handles validation and canonicalization of SMILES strings. This step matters because:

- Raw SMILES from databases often contain salts, stereochemistry artifacts, or invalid syntax
- The same molecule can have multiple valid SMILES representations
- ChemBERTa was trained on SMILES strings from the ZINC database, so canonicalizing inputs before embedding is good practice to reduce unnecessary variation

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

There's more going on here than just validation. Canonicalization means that `OC(=O)c1ccccc1OC(C)=O` and `CC(=O)Oc1ccccc1C(=O)O` both end up as the same canonical SMILES for aspirin, so you don't get duplicates in your index. The descriptor calculation gives you drug-likeness properties (MW, LogP, TPSA, etc.) that become Qdrant payload fields, so you can filter at query time ("find similar molecules with LogP under 5"). And if a SMILES string is malformed, the function just skips it instead of blowing up the whole run.

For batch processing (used by the API and UI in Sections 8 and 9), wrap the validation in a simple loop. Save both functions together as `molecule_processor.py`:

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

ChemBERTa is a RoBERTa model pre-trained on SMILES strings from the ZINC database. The model we use is `seyonec/ChemBERTa-zinc-base-v1`, available on HuggingFace. It outputs 768-dimensional hidden states.

There is no dedicated "embedding" head on this model. We extract embeddings by mean-pooling the last hidden state across all tokens (excluding padding). This approach is standard for extracting sentence-level embeddings from BERT-family models.

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

A few notes on the choices here. We use mean pooling instead of just grabbing the `[CLS]` token because ChemBERTa is RoBERTa-based, and RoBERTa's CLS token wasn't trained with a next-sentence prediction objective. Mean pooling across all non-padding tokens is the standard approach for extracting sequence-level representations from BERT-family models when no dedicated embedding head exists. Whether it outperforms CLS for molecular similarity specifically hasn't been rigorously benchmarked in published work, but it's consistent with best practices from the NLP sentence-embedding literature (Reimers & Gurevych, 2019).

We L2-normalize the vectors so that cosine similarity and dot product give the same results. This keeps things simple and makes scores comparable across queries. And we batch at 32 molecules at a time to balance memory usage against throughput. On a CPU, expect something like 100 to 300 molecules per second, depending on your hardware and how long the SMILES are.

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
# .vectors.size works for single-vector collections (which is what we create here)
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

**Filtered search** is one of Qdrant's strongest features. In drug discovery, you rarely want "the most similar molecule regardless of properties." More often, the question is "find me similar molecules with molecular weight under 500 and LogP under 5" (Lipinski's Rule of Five). Payload filters handle this without post-processing.

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


class MoleculeHit(BaseModel):
    smiles: str
    score: float
    molecular_weight: float
    logp: float


class SearchResponse(BaseModel):
    query_smiles: str
    canonical_smiles: str
    results: list[MoleculeHit]


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Search endpoint — runs CPU-bound inference in a thread pool."""
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

---

## 9. Step 6: Streamlit Web Interface (Alternative)

For interactive exploration by medicinal chemists who prefer a visual interface over API calls:

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

Here is the complete, runnable script that ties every step into a single pipeline. Copy this into a file, install the dependencies, and run it.

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

## 12. Limitations and Honest Caveats

This approach has real blind spots, and you should know about them before relying on it.

ChemBERTa only sees SMILES strings. It picks up topological patterns, but it has no concept of 3D structure. For targets where binding is all about shape (which is most of them), you'll want to combine this with 3D-aware models or shape-based methods.

The base model also isn't task-specific. It was trained with masked language modeling on ZINC. It never saw any activity data. If you fine-tune it on your target's assay data, the similarity rankings will improve dramatically.

Canonical SMILES can vary between RDKit versions, so always canonicalize with the same version you used during indexing. Otherwise you'll get different embeddings for the same molecule.

HNSW search is approximate. It doesn't guarantee finding the true nearest neighbor. For critical results, crank up `ef` at query time or run exact search on a shortlist to verify.

Evaluation is genuinely hard. "Similar" depends on context. Two molecules can be structurally close but biologically different, or structurally different but active against the same target. You need to benchmark against a retrieval task that actually matters for your use case (MoleculeNet benchmarks or your internal assay data are good starting points).

Watch out for long SMILES. The tokenizer truncates at 512 tokens, and polymers, macrocycles, or large natural products can exceed that. The embedding only represents what fits in the context window, which can be misleading. Consider adding a warning when token count goes over the limit.

Salt forms are another gotcha. SMILES like `[Na+].[Cl-]` or `CC(=O)O.[Na+]` contain disconnected fragments. `Chem.MolFromSmiles` accepts them without complaint, but the embedding will encode the salt alongside the active molecule. For production pharma work, strip salts first with `rdkit.Chem.SaltRemover.SaltRemover`.

---

## 13. Future Extensions

Consider the following architectural enhancements for production environments:

- **Contrastive Fine-Tuning:** Fine-tune ChemBERTa using target-specific assay data to improve similarity embeddings (e.g., pulling active molecules closer together).
- **Hybrid Similarity Indices:** Combining ChemBERTa embeddings with traditional structural fingerprints to allow users to toggle between learned embeddings and explicit sub-structure matches.
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
