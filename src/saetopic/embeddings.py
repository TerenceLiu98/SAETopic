"""
Embedding backend for document encoding.

Supports Jina embeddings, SentenceTransformers, and custom callables.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


class EmbeddingBackend:
    """
    Backend for computing document embeddings.

    Accepts a Hugging Face model id (loaded via SentenceTransformers), an
    already-instantiated object exposing ``encode`` (e.g. a
    ``SentenceTransformer``), or a plain callable mapping ``list[str]`` to a
    ``(n_docs, dim)`` array.

    Parameters
    ----------
    model : str or callable
        Model identifier (Hugging Face id) or a callable/encoder instance
    task : str, default="clustering"
        Task type forwarded to ``encode`` for Jina v5 embeddings
    device : str, default="auto"
        Device for computation ("auto", "cpu", "cuda", "mps")
    batch_size : int, default=32
        Batch size for embedding computation
    truncate_dim : int or None, default=None
        Matryoshka truncation dimension. Must match the SAE ``input_dim``.
    normalize : bool, default=True
        Whether to L2-normalize the resulting embeddings
    model_kwargs : dict or None, default=None
        Extra kwargs passed to ``SentenceTransformer`` (e.g. dtype)
    trust_remote_code : bool, default=True
        Forwarded to ``SentenceTransformer`` for remote-code models (Jina)
    """

    def __init__(
        self,
        model: str | Callable,
        task: str = "clustering",
        device: str = "auto",
        batch_size: int = 32,
        truncate_dim: int | None = None,
        normalize: bool = True,
        model_kwargs: dict | None = None,
        trust_remote_code: bool = True,
        max_seq_length: int | None = None,
    ):
        self.model = model
        self.task = task
        self.device = device
        self.batch_size = batch_size
        self.truncate_dim = truncate_dim
        self.normalize = normalize
        self.model_kwargs = dict(model_kwargs) if model_kwargs else {}
        self.trust_remote_code = trust_remote_code
        self.max_seq_length = max_seq_length

        self._backend = None
        self.embedding_dim_: int | None = None

    def _resolve_device(self) -> "torch.device":
        import torch

        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def _get_backend(self):
        """Lazily build / return the underlying encoder (str or .encode object)."""
        if self._backend is not None:
            return self._backend

        model = self.model
        if isinstance(model, str):
            import torch
            from sentence_transformers import SentenceTransformer

            device = self._resolve_device()
            model_kwargs = dict(self.model_kwargs)
            if device.type == "cuda" and "dtype" not in model_kwargs:
                model_kwargs["dtype"] = torch.bfloat16

            self._backend = SentenceTransformer(
                model,
                device=device,
                model_kwargs=model_kwargs,
                truncate_dim=self.truncate_dim,
                trust_remote_code=self.trust_remote_code,
            )
            if self.max_seq_length is not None:
                self._backend.max_seq_length = self.max_seq_length
            logger.info(
                "Loaded embedding model %s on %s (truncate_dim=%s, max_seq_length=%s)",
                model,
                device,
                self.truncate_dim,
                self.max_seq_length,
            )
        elif hasattr(model, "encode"):
            # Pre-instantiated encoder (e.g. a SentenceTransformer)
            self._backend = model
        else:
            # Will be treated as a plain callable in embed()
            self._backend = None
        return self._backend

    def embed(self, docs: list[str]) -> np.ndarray:
        """
        Compute embeddings for documents.

        Parameters
        ----------
        docs : list of str
            Documents to embed

        Returns
        -------
        np.ndarray
            Document embeddings (n_docs x embedding_dim), float32
        """
        if not docs:
            return np.zeros((0, 0), dtype=np.float32)

        backend = self._get_backend()
        if backend is not None:
            encode_kwargs = {"batch_size": self.batch_size, "show_progress_bar": False}
            try:
                embs = backend.encode(docs, task=self.task, **encode_kwargs)
            except TypeError:
                # Backend does not accept task= (non-Jina models)
                embs = backend.encode(docs, **encode_kwargs)
        elif callable(self.model):
            embs = self.model(docs)
        else:
            raise TypeError(
                f"EmbeddingBackend.model must be a str, an object with .encode, "
                f"or a callable; got {type(self.model)!r}"
            )

        embs = np.asarray(embs, dtype=np.float32)

        if embs.ndim != 2:
            raise ValueError(f"Embeddings must be 2D, got shape {embs.shape}")

        if self.normalize:
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            embs = embs / norms

        self.embedding_dim_ = embs.shape[1]
        return embs
