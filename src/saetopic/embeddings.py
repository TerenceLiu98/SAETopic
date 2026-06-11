"""
Embedding backend for document encoding.

Supports Jina embeddings, SentenceTransformers, and custom callables.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    import numpy as np


class EmbeddingBackend:
    """
    Backend for computing document embeddings.

    Parameters
    ----------
    model : str or callable
        Model identifier (Hugging Face ID) or callable
    task : str, default="clustering"
        Task type for Jina embeddings
    device : str, default="auto"
        Device for computation
    batch_size : int, default=32
        Batch size for embedding computation
    """

    def __init__(
        self,
        model: str | Callable,
        task: str = "clustering",
        device: str = "auto",
        batch_size: int = 32,
    ):
        self.model = model
        self.task = task
        self.device = device
        self.batch_size = batch_size

    def embed(
        self,
        docs: list[str],
    ) -> np.ndarray:
        """
        Compute embeddings for documents.

        Parameters
        ----------
        docs : list of str
            Documents to embed

        Returns
        -------
        np.ndarray
            Document embeddings (n_docs x embedding_dim)
        """
        # TODO: Implement embedding computation
        raise NotImplementedError("EmbeddingBackend.embed is not implemented yet")
