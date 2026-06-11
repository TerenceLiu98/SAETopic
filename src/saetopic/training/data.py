"""
Data loading utilities for SAE training.

This module provides utilities for loading and preparing embedding datasets
for training sparse autoencoders.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
import torch.nn.functional as functional
from torch import Tensor
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    """
    PyTorch Dataset for pre-computed embeddings.

    Parameters
    ----------
    embeddings : np.ndarray or Tensor
        Pre-computed embeddings (n_samples x embedding_dim)
    normalize : bool, default=True
        Whether to L2-normalize embeddings
    lazy : bool, default=False
        If True, keep embeddings in their original array/tensor container and
        convert/normalize individual samples in __getitem__. This is useful for
        memory-mapped .npy files.
    """

    def __init__(
        self,
        embeddings: np.ndarray | Tensor,
        normalize: bool = True,
        lazy: bool = False,
    ):
        self.normalize = normalize
        self.lazy = lazy

        if not lazy:
            if isinstance(embeddings, np.ndarray):
                embeddings = torch.from_numpy(embeddings).float()

            if normalize:
                embeddings = functional.normalize(embeddings, p=2, dim=-1)

        self.embeddings = embeddings
        self.n_samples = embeddings.shape[0]
        self.embedding_dim = embeddings.shape[1]

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tensor:
        embedding = self.embeddings[idx]
        if self.lazy:
            if isinstance(embedding, np.ndarray):
                embedding = torch.from_numpy(
                    np.asarray(embedding, dtype=np.float32).copy()
                )
            else:
                embedding = embedding.float()
            if self.normalize:
                embedding = functional.normalize(embedding, p=2, dim=-1)
        return embedding

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        normalize: bool = True,
        mmap_mode: Literal["r", "r+", "w+", "c"] | None = None,
    ) -> "EmbeddingDataset":
        """
        Load embeddings from a .npy or .pt file.

        Parameters
        ----------
        path : str or Path
            Path to embeddings file (.npy or .pt)
        normalize : bool, default=True
            Whether to L2-normalize embeddings
        mmap_mode : str or None, default=None
            Memory-map mode passed to np.load for .npy files. Use "r" for
            large embedding files to avoid loading all embeddings into RAM.

        Returns
        -------
        EmbeddingDataset
            Dataset instance
        """
        path = Path(path)
        if path.suffix == ".npy":
            embeddings = np.load(path, mmap_mode=mmap_mode)
            lazy = mmap_mode is not None
        elif path.suffix == ".pt":
            embeddings = torch.load(path)
            lazy = False
        else:
            raise ValueError(f"Unknown file type: {path.suffix}")

        return cls(cast(np.ndarray | Tensor, embeddings), normalize=normalize, lazy=lazy)


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
        Number of texts to accumulate before yielding embeddings
    encode_batch_size : int, default=32
        Internal batch size for embedder.encode() (to avoid OOM)
        Set lower if you get CUDA OOM errors
    encode_device : str, list[str], or None, default=None
        Device or devices to pass to embedder.encode(). A list enables
        SentenceTransformers multi-process / multi-GPU encoding.
    encode_chunk_size : int or None, default=None
        Chunk size passed to SentenceTransformers encode(). This controls
        work distribution for multi-process / multi-GPU encoding.
    text_chunk_size : int or None, default=None
        If set, split long documents into chunks of this many tokenizer tokens
        before embedding. This avoids truncating long FineWiki-style articles.
    text_chunk_overlap : int, default=0
        Number of tokenizer tokens to overlap between adjacent text chunks.
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
    ...     encode_batch_size=32,  # Lower if OOM
    ...     task="clustering",
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
        encode_batch_size: int = 32,
        encode_device: str | list[str] | torch.device | list[torch.device] | None = None,
        encode_chunk_size: int | None = None,
        text_chunk_size: int | None = None,
        text_chunk_overlap: int = 0,
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
        self.encode_batch_size = encode_batch_size
        self.encode_device = encode_device
        self.encode_chunk_size = encode_chunk_size
        self.text_chunk_size = text_chunk_size
        self.text_chunk_overlap = text_chunk_overlap
        self.normalize = normalize
        self.max_samples = max_samples
        self.seed = seed
        self.task = task

        if self.text_chunk_size is not None and self.text_chunk_size <= 0:
            raise ValueError("text_chunk_size must be greater than 0")
        if self.encode_chunk_size is not None and self.encode_chunk_size <= 0:
            raise ValueError("encode_chunk_size must be greater than 0")
        if self.text_chunk_overlap < 0:
            raise ValueError("text_chunk_overlap must be non-negative")
        if (
            self.text_chunk_size is not None
            and self.text_chunk_overlap >= self.text_chunk_size
        ):
            raise ValueError("text_chunk_overlap must be smaller than text_chunk_size")

        # Detect embedding dimension from first batch
        self._embedding_dim: int | None = None

    def _get_embedding_dim(self) -> int:
        """Get embedding dimension by encoding a sample."""
        sample = next(iter(self.base_dataset))
        text = sample[self.text_column] if isinstance(sample, dict) else sample
        if isinstance(text, list):
            text = text[0] if text else "test"
        chunks = self._split_text(text)
        emb = self._encode_texts([chunks[0] if chunks else "test"])
        return int(emb.shape[1])

    def _split_text(self, text: Any) -> list[str]:
        """Split one dataset text into one or more embedding inputs."""
        if not isinstance(text, str):
            text = str(text)

        text = text.strip()
        if not text:
            return []

        if self.text_chunk_size is None:
            return [text]

        tokenizer = getattr(self.embedder, "tokenizer", None)
        if tokenizer is None:
            return self._split_text_by_words(text)

        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) <= self.text_chunk_size:
            return [text]

        chunks = []
        step = self.text_chunk_size - self.text_chunk_overlap
        for start in range(0, len(token_ids), step):
            chunk_ids = token_ids[start : start + self.text_chunk_size]
            if not chunk_ids:
                continue
            chunk = tokenizer.decode(
                chunk_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            ).strip()
            if chunk:
                chunks.append(chunk)

        return chunks

    def _split_text_by_words(self, text: str) -> list[str]:
        """Fallback chunking when the embedder does not expose a tokenizer."""
        if self.text_chunk_size is None:
            return [text]

        words = text.split()
        if len(words) <= self.text_chunk_size:
            return [text]

        chunks = []
        step = self.text_chunk_size - self.text_chunk_overlap
        for start in range(0, len(words), step):
            chunk = " ".join(words[start : start + self.text_chunk_size]).strip()
            if chunk:
                chunks.append(chunk)
        return chunks

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        """Encode texts with the configured SentenceTransformers options."""
        encode_kwargs = {
            "batch_size": self.encode_batch_size,
            "task": self.task,
            "show_progress_bar": False,
        }
        if self.encode_device is not None:
            encode_kwargs["device"] = self.encode_device
        if self.encode_chunk_size is not None:
            encode_kwargs["chunk_size"] = self.encode_chunk_size

        return cast(np.ndarray, self.embedder.encode(texts, **encode_kwargs))

    def _embed_text_batch(self, texts: list[str]) -> Tensor:
        """Encode a text batch and return normalized CPU float tensors."""
        embedding_array = self._encode_texts(texts)
        embeddings = torch.from_numpy(embedding_array).float()

        if self.normalize:
            embeddings = functional.normalize(embeddings, p=2, dim=-1)

        return embeddings

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
        texts_buffer = []
        n_samples_yielded = 0

        for item in self.base_dataset:
            # Check max_samples limit
            if self.max_samples is not None and n_samples_yielded >= self.max_samples:
                break

            # Extract text
            if isinstance(item, dict):
                text = item.get(self.text_column, "")
            else:
                text = item

            # Skip empty texts
            if not text or (isinstance(text, str) and len(text.strip()) == 0):
                continue

            for text_chunk in self._split_text(text):
                if self.max_samples is not None and (
                    n_samples_yielded + len(texts_buffer) >= self.max_samples
                ):
                    break
                texts_buffer.append(text_chunk)

            # When buffer is full, encode and yield
            if len(texts_buffer) >= self.buffer_size:
                # Shuffle buffer
                random.shuffle(texts_buffer)

                # Encode in batches
                for i in range(0, len(texts_buffer), self.embedding_batch_size):
                    batch_texts = texts_buffer[i : i + self.embedding_batch_size]

                    embeddings = self._embed_text_batch(batch_texts)
                    yield embeddings
                    n_samples_yielded += embeddings.shape[0]

                texts_buffer = []

        # Yield remaining items
        if texts_buffer:
            for i in range(0, len(texts_buffer), self.embedding_batch_size):
                batch_texts = texts_buffer[i : i + self.embedding_batch_size]
                embeddings = self._embed_text_batch(batch_texts)
                yield embeddings
                n_samples_yielded += embeddings.shape[0]


def create_streaming_dataset(
    dataset_name: str = "HuggingFaceFW/finewiki",
    subset: str | None = None,
    split: str = "train",
    embedder=None,
    text_column: str = "text",
    buffer_size: int = 10000,
    embedding_batch_size: int = 256,
    encode_batch_size: int = 32,
    encode_device: str | list[str] | torch.device | list[torch.device] | None = None,
    encode_chunk_size: int | None = None,
    text_chunk_size: int | None = None,
    text_chunk_overlap: int = 0,
    normalize: bool = True,
    seed: int = 42,
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
    subset : str or None, default=None
        Dataset subset/config name (e.g., "en_us" for google/fleurs)
        See: https://huggingface.co/docs/datasets/loading#subset-selection
    split : str, default="train"
        Dataset split
    embedder : callable
        Embedding function (e.g., SentenceTransformer)
    text_column : str, default="text"
        Column name containing text
    buffer_size : int, default=10000
        Shuffle buffer size
    embedding_batch_size : int, default=256
        Number of texts to accumulate before yielding embeddings
    encode_batch_size : int, default=32
        Internal batch size for embedder.encode() (to avoid OOM)
        Set lower if you get CUDA OOM errors
    encode_device : str, list[str], or None, default=None
        Device or devices to pass to embedder.encode()
    encode_chunk_size : int or None, default=None
        Chunk size passed to SentenceTransformers encode()
    text_chunk_size : int or None, default=None
        Split long documents into chunks of this many tokenizer tokens
    text_chunk_overlap : int, default=0
        Number of tokenizer tokens to overlap between adjacent chunks
    normalize : bool, default=True
        Whether to L2-normalize embeddings before yielding/saving them
    seed : int, default=42
        Random seed for streaming buffer shuffling
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
    >>> # Basic usage
    >>> dataset = create_streaming_dataset(
    ...     embedder=embedder,
    ...     buffer_size=50000,
    ...     max_samples=1000000,
    ... )
    >>>
    >>> # With subset (for datasets with multiple configs)
    >>> dataset = create_streaming_dataset(
    ...     dataset_name="google/fleurs",
    ...     subset="en_us",
    ...     embedder=embedder,
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
    # Note: subset is passed as the second positional argument to load_dataset
    load_args = [dataset_name]
    if subset is not None:
        load_args.append(subset)

    hf_dataset = load_dataset(
        *load_args,
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
        encode_batch_size=encode_batch_size,
        encode_device=encode_device,
        encode_chunk_size=encode_chunk_size,
        text_chunk_size=text_chunk_size,
        text_chunk_overlap=text_chunk_overlap,
        normalize=normalize,
        seed=seed,
        max_samples=max_samples,
        task=task,
    )
