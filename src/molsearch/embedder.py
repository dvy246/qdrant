"""
ChemBERTa embedding generator.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from molsearch.config import BATCH_SIZE, MODEL_NAME, VECTOR_DIM

logger = logging.getLogger(__name__)


def _get_device() -> torch.device:
    """Pick best available torch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class MoleculeEmbedder:
    """
    Generates L2-normalized ChemBERTa embeddings from SMILES.

    Attributes:
        tokenizer: HuggingFace tokenizer for ChemBERTa.
        model: HuggingFace model for ChemBERTa.
        vector_dim: Dimension of the output embeddings (derived from the model).
        device: Torch device used for inference.
    """

    def __init__(self, model_name: str = MODEL_NAME):
        """
        Initialize the embedder by loading the tokenizer and model.

        Args:
            model_name: HuggingFace model identifier.
                        Default: seyonec/ChemBERTa-zinc-base-v1

        Raises:
            ValueError: If the model's hidden size does not match VECTOR_DIM.
            OSError: If the model cannot be downloaded (network error, invalid
                     model name, etc.).
        """
        self.device = _get_device()
        logger.info("Using device: %s", self.device)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
        except OSError as exc:
            logger.error("failed to load model %s: %s", model_name, exc)
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

        logger.info("loaded %s on %s", model_name, self.device)

    def _mean_pool(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Average hidden states across non-padding tokens."""
        mask_expanded = (
            attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        )
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

        Raises:
            ValueError: If any SMILES exceeds token length limit.
            RuntimeError: If model inference fails.
        """
        n = len(smiles_list)
        if n == 0:
            return np.empty((0, self.vector_dim), dtype=np.float32)

        # pre-allocate to avoid mem spikes
        result = np.empty((n, self.vector_dim), dtype=np.float32)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = smiles_list[start:end]

            # bail on absurdly long smiles
            if any(len(s) > 400 for s in batch):
                logger.error("batch %d-%d: smiles too long, skipping", start, end)
                result[start:end] = np.nan
                continue

            try:
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=False,
                    return_tensors="pt",
                )
            except Exception as exc:
                logger.error("tokenization failed %d-%d: %s", start, end, exc)
                result[start:end] = np.nan
                continue

            if encoded["input_ids"].shape[1] > 512:
                logger.error("batch %d-%d: context limit exceeded", start, end)
                result[start:end] = np.nan
                continue

            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            try:
                with torch.no_grad():
                    outputs = self.model(**encoded)

                embeddings = self._mean_pool(
                    outputs.last_hidden_state,
                    encoded["attention_mask"],
                )

                # L2 normalize so dot product == cosine
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                batch_result = embeddings.cpu().numpy()
                # TODO: benchmark whether batch_size=64 is faster on GPU
                if np.any(np.isnan(batch_result)) or np.any(np.isinf(batch_result)):
                    logger.error("batch %d-%d: generated nan/inf vectors", start, end)
                    result[start:end] = np.nan
                    continue

                result[start:end] = batch_result

            except torch.cuda.OutOfMemoryError:
                logger.error("batch %d-%d: OOM, recovering", start, end)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                result[start:end] = np.nan
                continue
            except Exception as exc:
                logger.error("inference failed %d-%d: %s", start, end, exc)
                result[start:end] = np.nan
                continue

        return result
