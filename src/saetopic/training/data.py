"""
Data loading utilities for SAE training.

This module provides utilities for loading and preparing embedding datasets
for training sparse autoencoders.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from torch import Tensor


class EmbeddingDataset(Dataset):
    """
    PyTorch Dataset for pre-computed embeddings.

    Parameters
    ----------
    embeddings : np.ndarray or Tensor
        Pre-computed embeddings (n_samples x embedding_dim)
    normalize : bool, default=True
        Whether to L2-normalize embeddings
    """

    def __init__(
        self,
        embeddings: np.ndarray | Tensor,
        normalize: bool = True,
    ):
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.from_numpy(embeddings).float()

        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        self.embeddings = embeddings
        self.n_samples = embeddings.shape[0]
        self.embedding_dim = embeddings.shape[1]

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tensor:
        return self.embeddings[idx]

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        normalize: bool = True,
    ) -> "EmbeddingDataset":
        """
        Load embeddings from a .npy or .pt file.

        Parameters
        ----------
        path : str or Path
            Path to embeddings file (.npy or .pt)
        normalize : bool, default=True
            Whether to L2-normalize embeddings

        Returns
        -------
        EmbeddingDataset
            Dataset instance
        """
        path = Path(path)
        if path.suffix == ".npy":
            embeddings = np.load(path)
        elif path.suffix == ".pt":
            embeddings = torch.load(path)
        else:
            raise ValueError(f"Unknown file type: {path.suffix}")

        return cls(embeddings, normalize=normalize)


def load_embeddings_from_hf(
    dataset_name: str = "HuggingFaceFW/finewiki",
    split: str = "train",
    embedding_column: str = "embedding",
    max_samples: int | None = None,
    normalize: bool = True,
) -> EmbeddingDataset:
    """
    Load embeddings from a Hugging Face dataset.

    This function assumes the dataset contains pre-computed embeddings.
    For text datasets, you'll need to compute embeddings first.

    Parameters
    ----------
    dataset_name : str, default="HuggingFaceFW/finewiki"
        Hugging Face dataset name
    split : str, default="train"
        Dataset split to load
    embedding_column : str, default="embedding"
        Column name containing embeddings
    max_samples : int or None, default=None
        Maximum number of samples to load (None for all)
    normalize : bool, default=True
        Whether to L2-normalize embeddings

    Returns
    -------
    EmbeddingDataset
        Dataset with loaded embeddings

    Examples
    --------
    >>> from saetopic.training import load_embeddings_from_hf
    >>> dataset = load_embeddings_from_hf("saetopic/finewiki-embeddings", max_samples=100000)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "datasets package is required. Install with: pip install datasets"
        )

    # Load dataset
    hf_dataset = load_dataset(dataset_name, split=split)

    # Extract embeddings
    if embedding_column in hf_dataset.column_names:
        embeddings = np.array(hf_dataset[embedding_column])
    else:
        # If no pre-computed embeddings, this won't work
        # User needs to compute embeddings first
        raise ValueError(
            f"Dataset does not contain '{embedding_column}' column. "
            "Please compute embeddings first or use a dataset with pre-computed embeddings."
        )

    # Subsample if needed
    if max_samples is not None and len(embeddings) > max_samples:
        indices = np.random.choice(len(embeddings), max_samples, replace=False)
        embeddings = embeddings[indices]

    return EmbeddingDataset(embeddings, normalize=normalize)


# Import F.normalize for use in this module
import torch.nn.functional as F


class StreamingEmbeddingDataset:
    """
    Streaming dataset that computes embeddings on-the-fly from HF dataset.

    This class enables training on large datasets without pre-computing
    all embeddings to disk. It uses HuggingFace Datasets streaming mode
    and computes embeddings in batches during training.

    Parameters
    ----------
    hf_dataset : datasets.Dataset or IterableDataset
        HuggingFace dataset (use streaming=True for large datasets)
    embedder : callable
        Embedding function (e.g., SentenceTransformer.encode)
    text_column : str, default="text"
        Column name containing text to embed
    buffer_size : int, default=10000
        Number of samples to buffer for shuffling
    embedding_batch_size : int, default=256
        Batch size for embedding computation
    normalize : bool, default=True
        Whether to L2-normalize embeddings
    seed : int, default=42
        Random seed for shuffling
    max_samples : int or None, default=None
        Maximum number of samples to stream (None for unlimited)
    task : str, default="clustering"
        Task type for Jina embeddings (e.g., "clustering", "retrieval")
        Passed to embedder.encode() as task parameter

    Examples
    --------
    >>> from datasets import load_dataset
    >>> from sentence_transformers import SentenceTransformer
    >>>
    >>> # Load dataset in streaming mode
    >>> hf_ds = load_dataset("HuggingFaceFW/finewiki", streaming=True, split="train")
    >>>
    >>> # Create embedder
    >>> embedder = SentenceTransformer("jinaai/jina-embeddings-v5-text-small")
    >>>
    >>> # Create streaming dataset
    >>> dataset = StreamingEmbeddingDataset(
    ...     hf_ds,
    ...     embedder,
    ...     buffer_size=50000,
    ...     task="clustering",  # Required for Jina v5
    ... )
    >>>
    >>> # Use in training (iterator mode)
    >>> for batch_embeddings in dataset:
    ...     # train on batch_embeddings
    ...     pass
    """

    def __init__(
        self,
        hf_dataset,
        embedder,
        text_column: str = "text",
        buffer_size: int = 10000,
        embedding_batch_size: int = 256,
        normalize: bool = True,
        seed: int = 42,
        max_samples: int | None = None,
        task: str = "clustering",
    ):
        self.base_dataset = hf_dataset
        self.embedder = embedder
        self.text_column = text_column
        self.buffer_size = buffer_size
        self.embedding_batch_size = embedding_batch_size
        self.normalize = normalize
        self.max_samples = max_samples
        self.seed = seed
        self.task = task

        # Detect embedding dimension from first batch
        self._embedding_dim: int | None = None
        self._n_samples_yielded: int = 0

    def _get_embedding_dim(self) -> int:
        """Get embedding dimension by encoding a sample."""
        sample = next(iter(self.base_dataset))
        text = sample[self.text_column] if isinstance(sample, dict) else sample
        if isinstance(text, list):
            text = text[0] if text else "test"
        emb = self.embedder.encode([text], task=self.task)
        return emb.shape[1]

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        if self._embedding_dim is None:
            self._embedding_dim = self._get_embedding_dim()
        return self._embedding_dim

    def __iter__(self):
        """Iterate over batches of embeddings."""
        import random

        random.seed(self.seed)
        buffer = []
        texts_buffer = []

        for item in self.base_dataset:
            # Check max_samples limit
            if self.max_samples is not None and self._n_samples_yielded >= self.max_samples:
                break

            # Extract text
            if isinstance(item, dict):
                text = item.get(self.text_column, "")
            else:
                text = item

            # Skip empty texts
            if not text or (isinstance(text, str) and len(text.strip()) == 0):
                continue

            texts_buffer.append(text)

            # When buffer is full, encode and yield
            if len(texts_buffer) >= self.buffer_size:
                # Shuffle buffer
                random.shuffle(texts_buffer)

                # Encode in batches
                for i in range(0, len(texts_buffer), self.embedding_batch_size):
                    batch_texts = texts_buffer[i : i + self.embedding_batch_size]

                    # Encode
                    embeddings = self.embedder.encode(
                        batch_texts,
                        batch_size=len(batch_texts),
                        task=self.task,
                        show_progress_bar=False,
                    )

                    # Convert to tensor
                    embeddings = torch.from_numpy(embeddings).float()

                    # Normalize if needed
                    if self.normalize:
                        embeddings = F.normalize(embeddings, p=2, dim=-1)

                    yield embeddings
                    self._n_samples_yielded += embeddings.shape[0]

                texts_buffer = []

        # Yield remaining items
        if texts_buffer:
            for i in range(0, len(texts_buffer), self.embedding_batch_size):
                batch_texts = texts_buffer[i : i + self.embedding_batch_size]
                embeddings = self.embedder.encode(
                    batch_texts,
                    batch_size=len(batch_texts),
                    show_progress_bar=False,
                )
                embeddings = torch.from_numpy(embeddings).float()
                if self.normalize:
                    embeddings = F.normalize(embeddings, p=2, dim=-1)
                yield embeddings
                self._n_samples_yielded += embeddings.shape[1]


def create_streaming_dataset(
    dataset_name: str = "HuggingFaceFW/finewiki",
    split: str = "train",
    embedder=None,
    text_column: str = "text",
    buffer_size: int = 10000,
    embedding_batch_size: int = 256,
    streaming: bool = True,
    max_samples: int | None = None,
    task: str = "clustering",
    **hf_kwargs,
) -> StreamingEmbeddingDataset:
    """
    Create a streaming embedding dataset from HuggingFace.

    This is a convenience function for creating a StreamingEmbeddingDataset
    directly from a HuggingFace dataset name.

    Parameters
    ----------
    dataset_name : str, default="HuggingFaceFW/finewiki"
        HuggingFace dataset name
    split : str, default="train"
        Dataset split
    embedder : callable
        Embedding function (e.g., SentenceTransformer)
    text_column : str, default="text"
        Column name containing text
    buffer_size : int, default=10000
        Shuffle buffer size
    embedding_batch_size : int, default=256
        Batch size for embedding computation
    streaming : bool, default=True
        Use streaming mode (set to False for small datasets)
    max_samples : int or None, default=None
        Maximum samples to stream
    task : str, default="clustering"
        Task type for Jina embeddings (e.g., "clustering", "retrieval")
    **hf_kwargs
        Additional arguments for load_dataset

    Returns
    -------
    StreamingEmbeddingDataset
        Streaming dataset

    Examples
    --------
    >>> from sentence_transformers import SentenceTransformer
    >>> embedder = SentenceTransformer("jinaai/jina-embeddings-v5-text-small")
    >>>
    >>> dataset = create_streaming_dataset(
    ...     embedder=embedder,
    ...     buffer_size=50000,
    ...     max_samples=1000000,
    ...     task="clustering",  # Required for Jina v5
    ... )
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "datasets package is required. Install with: pip install datasets"
        )

    if embedder is None:
        raise ValueError("embedder must be provided")

    # Load HF dataset
    hf_dataset = load_dataset(
        dataset_name,
        split=split,
        streaming=streaming,
        **hf_kwargs,
    )

    return StreamingEmbeddingDataset(
        hf_dataset,
        embedder,
        text_column=text_column,
        buffer_size=buffer_size,
        embedding_batch_size=embedding_batch_size,
        max_samples=max_samples,
        task=task,
    )
