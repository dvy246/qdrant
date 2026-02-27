"""
Molecule embedder: ChemBERTa-based SMILES-to-vector embedding.

Uses the seyonec/ChemBERTa-zinc-base-v1 model from HuggingFace with
mean pooling over the last hidden state to produce dense vectors.

Mean pooling is standard for extracting sequence-level embeddings from
RoBERTa-family models. We L2-normalize so cosine similarity == dot product.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from molsearch.config import BATCH_SIZE, MODEL_NAME, VECTOR_DIM

logger = logging.getLogger(__name__)


def _get_device() -> torch.device:
    """Select the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class MoleculeEmbedder:
    """
    Embeds SMILES strings into dense vectors using ChemBERTa.

    Uses mean pooling over the last hidden state (excluding padding tokens)
    followed by L2 normalization. This produces unit-length vectors suitable
    for cosine (or dot-product) similarity search.

    Attributes:
        tokenizer: HuggingFace tokenizer for ChemBERTa.
        model: HuggingFace model for ChemBERTa.
        vector_dim: Dimension of the output embeddings (derived from the model).
        device: Torch device used for inference.
    """

    def __init__(self, model_name: str = MODEL_NAME):
        """
        Initialize the embedder by loading the tokenizer and model.

        Automatically selects GPU (CUDA or MPS) if available, otherwise
        falls back to CPU.

        Args:
            model_name: HuggingFace model identifier.
                        Default: seyonec/ChemBERTa-zinc-base-v1

        Raises:
            ValueError: If the model's hidden size does not match VECTOR_DIM
                        in config.py. Update VECTOR_DIM when switching models.
            OSError: If the model cannot be downloaded (network error, invalid
                     model name, etc.).
        """
        self.device = _get_device()
        logger.info("Using device: %s", self.device)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
        except OSError as exc:
            logger.error(
                "Failed to load model '%s'. Check the model name and your "
                "network connection. Error: %s",
                model_name,
                exc,
            )
            raise

        self.model.to(self.device)
        self.model.eval()
        self.vector_dim = self.model.config.hidden_size

        if self.vector_dim != VECTOR_DIM:
            raise ValueError(
                f"Model hidden size ({self.vector_dim}) does not match "
                f"VECTOR_DIM ({VECTOR_DIM}) in config.py. Update VECTOR_DIM "
                f"to match your model before creating a Qdrant collection."
            )

        logger.info("Loaded %s (dim=%d) on %s", model_name, self.vector_dim, self.device)

    def _mean_pool(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mean pool hidden states, masking out padding tokens.

        Args:
            hidden_states: Tensor of shape (batch_size, seq_len, hidden_dim).
            attention_mask: Tensor of shape (batch_size, seq_len).

        Returns:
            Pooled tensor of shape (batch_size, hidden_dim).
        """
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        return sum_hidden / sum_mask

    def embed(
        self,
        smiles_list: list[str],
        batch_size: int = BATCH_SIZE,
    ) -> np.ndarray:
        """
        Embed a list of SMILES strings into dense vectors.

        Processes in batches and pre-allocates the output array to avoid
        the memory doubling that np.vstack causes with large datasets.

        Args:
            smiles_list: List of SMILES strings to embed.
            batch_size: Number of SMILES to process per batch.

        Returns:
            Numpy array of shape (n, vector_dim) with L2-normalized embeddings.
        """
        n = len(smiles_list)
        if n == 0:
            return np.empty((0, self.vector_dim), dtype=np.float32)

        # Pre-allocate to avoid memory doubling from vstack
        result = np.empty((n, self.vector_dim), dtype=np.float32)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = smiles_list[start:end]

            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=False,
                return_tensors="pt",
            )

            if encoded["input_ids"].shape[1] > 512:
                raise ValueError(
                    "One or more SMILES strings exceed the maximum token length of 512 "
                    "and would be completely biologically invalid if truncated."
                )
            # Move input tensors to the same device as the model
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.no_grad():
                outputs = self.model(**encoded)

            embeddings = self._mean_pool(
                outputs.last_hidden_state,
                encoded["attention_mask"],
            )

            # L2 normalize so cosine similarity equals dot product
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            result[start:end] = embeddings.cpu().numpy()

        return result
