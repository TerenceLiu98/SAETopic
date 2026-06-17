"""
Data loading utilities for SAE training.

This module provides utilities for loading and preparing embedding datasets
for training sparse autoencoders.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
import torch.nn.functional as functional
from torch import Tensor
from torch.utils.data import Dataset


_TEXT_URL_PATTERN = re.compile(r"(?i)\b(?:https?://|www\.)\S+")


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
        memory-mapped .npy files. For sharded embedding directories, use
        ShardedEmbeddingDataset.
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
    ) -> "EmbeddingDataset | ShardedEmbeddingDataset":
        """
        Load embeddings from a .npy, .pt, or sharded embedding directory.

        Parameters
        ----------
        path : str or Path
            Path to embeddings file (.npy or .pt) or sharded embedding directory
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
        if path.is_dir():
            return ShardedEmbeddingDataset.from_directory(
                path,
                normalize=normalize,
                mmap_mode=mmap_mode,
            )
        elif path.suffix == ".npy":
            embeddings = np.load(path, mmap_mode=mmap_mode)
            lazy = mmap_mode is not None
        elif path.suffix == ".pt":
            embeddings = torch.load(path)
            lazy = False
        else:
            raise ValueError(f"Unknown file type: {path.suffix}")

        return cls(cast(np.ndarray | Tensor, embeddings), normalize=normalize, lazy=lazy)


class ShardedEmbeddingDataset(Dataset):
    """
    PyTorch Dataset for sharded `.npy` embedding directories.

    Sharded embeddings are stored as:

    - `manifest.json`: total shape, dtype, and shard metadata
    - `shard_000000.npy`, `shard_000001.npy`, ...: row-contiguous embedding shards

    Only the shard needed for the requested sample is memory-mapped.
    """

    def __init__(
        self,
        directory: str | Path,
        manifest: dict[str, Any],
        normalize: bool = True,
        mmap_mode: Literal["r", "r+", "w+", "c"] | None = "r",
    ):
        self.directory = Path(directory)
        self.manifest = manifest
        self.normalize = normalize
        self.mmap_mode = mmap_mode

        shape = manifest["shape"]
        self.n_samples = int(shape[0])
        self.embedding_dim = int(shape[1])
        self.shards = manifest["shards"]

        self._starts: list[int] = []
        offset = 0
        for shard in self.shards:
            self._starts.append(offset)
            offset += int(shard["shape"][0])

        if offset != self.n_samples:
            raise ValueError(
                f"Shard rows ({offset}) do not match manifest shape ({self.n_samples})"
            )

        self._active_shard_index: int | None = None
        self._active_shard: np.ndarray | None = None

    @classmethod
    def from_directory(
        cls,
        path: str | Path,
        normalize: bool = True,
        mmap_mode: Literal["r", "r+", "w+", "c"] | None = "r",
    ) -> "ShardedEmbeddingDataset":
        """Load a sharded embedding dataset from a directory."""
        path = Path(path)
        manifest_path = path / "manifest.json"
        if not manifest_path.exists():
            raise ValueError(f"Missing sharded embedding manifest: {manifest_path}")

        manifest = json.loads(manifest_path.read_text())
        if manifest.get("format") != "saetopic.sharded_embeddings.v1":
            raise ValueError(f"Unknown sharded embedding format in {manifest_path}")

        return cls(path, manifest, normalize=normalize, mmap_mode=mmap_mode)

    def __len__(self) -> int:
        return self.n_samples

    def _load_shard(self, shard_index: int) -> np.ndarray:
        if self._active_shard_index != shard_index or self._active_shard is None:
            shard_path = self.directory / self.shards[shard_index]["file"]
            self._active_shard = np.load(shard_path, mmap_mode=self.mmap_mode)
            self._active_shard_index = shard_index
        return self._active_shard

    def __getitem__(self, idx: int) -> Tensor:
        if idx < 0:
            idx += self.n_samples
        if idx < 0 or idx >= self.n_samples:
            raise IndexError(idx)

        shard_index = int(np.searchsorted(self._starts, idx, side="right") - 1)
        shard = self._load_shard(shard_index)
        local_idx = idx - self._starts[shard_index]
        embedding = torch.from_numpy(np.asarray(shard[local_idx], dtype=np.float32).copy())

        if self.normalize:
            embedding = functional.normalize(embedding, p=2, dim=-1)

        return embedding


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
    text_split_strategy : {"token", "paragraph"}, default="token"
        How to split text before embedding. ``"token"`` uses fixed-size
        tokenizer chunks. ``"paragraph"`` uses blank-line-separated
        paragraphs, matching the SAE-TM foundation-SAE preprocessing style.
    min_sentences_per_chunk : int, default=1
        Minimum number of sentence-like spans required for a chunk when using
        ``text_split_strategy="paragraph"``.
    normalize : bool, default=True
        Whether to L2-normalize embeddings
    seed : int, default=42
        Random seed for shuffling
    max_samples : int or None, default=None
        Maximum number of samples to stream (None for unlimited)
    skip_samples : int, default=0
        Number of text chunks to skip before encoding. This is used for
        resuming sharded embedding jobs without re-encoding saved chunks.
    task : str, default="clustering"
        Task type for Jina embeddings (e.g., "clustering", "retrieval")
        Passed to embedder.encode() as task parameter
    skip_source_rows : int, default=0
        Number of raw source rows to skip before chunking. Used with sharded
        embedding resume cursors to avoid re-scanning the entire source.
    skip_chunk_offset : int, default=0
        Number of text chunks known to exist before ``skip_source_rows``.
    encode_method : {"encode", "document", "query"}, default="encode"
        SentenceTransformer method to use for text inputs. Use ``"document"``
        with Jina v5 omni non-retrieval tasks so the model applies its
        document prompt instead of treating bare text as off-distribution.
    sanitize_urls : bool, default=False
        Replace URLs in text chunks with ``"[URL]"`` before encoding. This is
        useful for text-only corpora with Jina omni models because bare URLs
        are valid media inputs and may be downloaded by the model.
    num_chunk_workers : int, default=1
        Number of threads used to split source rows into text chunks.
    chunk_worker_batch_size : int, default=1024
        Source rows collected before dispatching one parallel chunking batch.

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
        text_split_strategy: str = "token",
        min_sentences_per_chunk: int = 1,
        normalize: bool = True,
        seed: int = 42,
        max_samples: int | None = None,
        skip_samples: int = 0,
        task: str = "clustering",
        skip_source_rows: int = 0,
        skip_chunk_offset: int = 0,
        encode_method: str = "encode",
        sanitize_urls: bool = False,
        num_chunk_workers: int = 1,
        chunk_worker_batch_size: int = 1024,
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
        self.text_split_strategy = text_split_strategy
        self.min_sentences_per_chunk = min_sentences_per_chunk
        self.normalize = normalize
        self.max_samples = max_samples
        self.skip_samples = skip_samples
        self.seed = seed
        self.task = task
        self.skip_source_rows = skip_source_rows
        self.skip_chunk_offset = skip_chunk_offset
        self.encode_method = encode_method
        self.sanitize_urls = sanitize_urls
        self.num_chunk_workers = num_chunk_workers
        self.chunk_worker_batch_size = chunk_worker_batch_size
        self.source_rows_seen = 0
        self.safe_resume_cursor = {
            "source_rows_seen": int(self.skip_source_rows),
            "chunks_seen": int(self.skip_chunk_offset),
        }
        self.source_total = self._infer_source_total()
        self._encode_pool: dict[str, Any] | None = None

        if self.skip_samples < 0:
            raise ValueError("skip_samples must be non-negative")
        if self.skip_source_rows < 0:
            raise ValueError("skip_source_rows must be non-negative")
        if self.skip_chunk_offset < 0:
            raise ValueError("skip_chunk_offset must be non-negative")
        if self.encode_method not in {"encode", "document", "query"}:
            raise ValueError("encode_method must be 'encode', 'document', or 'query'")
        if self.num_chunk_workers <= 0:
            raise ValueError("num_chunk_workers must be greater than 0")
        if self.chunk_worker_batch_size <= 0:
            raise ValueError("chunk_worker_batch_size must be greater than 0")
        if self.text_split_strategy not in {"token", "paragraph"}:
            raise ValueError("text_split_strategy must be 'token' or 'paragraph'")
        if self.text_chunk_size is not None and self.text_chunk_size <= 0:
            raise ValueError("text_chunk_size must be greater than 0")
        if self.min_sentences_per_chunk <= 0:
            raise ValueError("min_sentences_per_chunk must be greater than 0")
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

    def _infer_source_total(self) -> int | None:
        """Infer raw dataset rows when the source exposes a cheap length."""
        try:
            source_total = len(self.base_dataset)
        except TypeError:
            return None
        except NotImplementedError:
            return None

        return int(source_total)

    def _encode_device_list(self) -> list[str] | None:
        """Return encode devices as strings when multi-device encoding is requested."""
        if not isinstance(self.encode_device, list) or len(self.encode_device) <= 1:
            return None
        return [str(device) for device in self.encode_device]

    def _start_encode_pool(self) -> None:
        """Start a persistent SentenceTransformers pool for multi-device encoding."""
        devices = self._encode_device_list()
        if devices is None or self._encode_pool is not None:
            return
        if not hasattr(self.embedder, "start_multi_process_pool"):
            return

        print(
            "Starting SentenceTransformers encode pool on devices: "
            + ", ".join(devices)
        )
        self._encode_pool = self.embedder.start_multi_process_pool(devices)

    def _stop_encode_pool(self) -> None:
        """Stop the persistent SentenceTransformers pool if one was started."""
        if self._encode_pool is None:
            return
        if hasattr(self.embedder, "stop_multi_process_pool"):
            self.embedder.stop_multi_process_pool(self._encode_pool)
        self._encode_pool = None

    def _get_embedding_dim(self) -> int:
        """Get embedding dimension by encoding a sample."""
        sample = next(iter(self._iter_base_dataset_from_skip()))
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
        if self.sanitize_urls:
            text = _TEXT_URL_PATTERN.sub("[URL]", text)

        if self.text_split_strategy == "paragraph":
            return self._split_text_by_paragraphs(text)

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

    def _split_text_by_paragraphs(self, text: str) -> list[str]:
        """Split text into paragraph-like chunks with a sentence-count filter."""
        paragraphs = [
            paragraph.strip()
            for paragraph in re.split(r"\n\s*\n+", text)
            if paragraph.strip()
        ]
        if not paragraphs:
            paragraphs = [text]

        return [
            paragraph
            for paragraph in paragraphs
            if self._count_sentences(paragraph) >= self.min_sentences_per_chunk
        ]

    @staticmethod
    def _count_sentences(text: str) -> int:
        """Count sentence-like spans without requiring external NLP resources."""
        spans = re.findall(r"[^.!?\n]+[.!?]+", text)
        if spans:
            return len(spans)
        return 1 if text.strip() else 0

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
            "show_progress_bar": False,
        }
        if self._encode_pool is not None:
            encode_kwargs["pool"] = self._encode_pool
        elif self.encode_device is not None:
            encode_kwargs["device"] = self.encode_device
        if self.encode_chunk_size is not None:
            encode_kwargs["chunk_size"] = self.encode_chunk_size

        if self.encode_method == "document":
            encode_document = getattr(self.embedder, "encode_document", None)
            if callable(encode_document):
                return cast(np.ndarray, encode_document(texts, **encode_kwargs))
            return cast(
                np.ndarray,
                self.embedder.encode(texts, prompt_name="document", **encode_kwargs),
            )

        if self.encode_method == "query":
            encode_query = getattr(self.embedder, "encode_query", None)
            if callable(encode_query):
                return cast(np.ndarray, encode_query(texts, **encode_kwargs))
            return cast(
                np.ndarray,
                self.embedder.encode(texts, prompt_name="query", **encode_kwargs),
            )

        encode_kwargs["task"] = self.task
        return cast(np.ndarray, self.embedder.encode(texts, **encode_kwargs))

    def _embed_text_batch(self, texts: list[str]) -> Tensor:
        """Encode a text batch and return normalized CPU float tensors."""
        embedding_array = self._encode_texts(texts)
        embeddings = torch.from_numpy(embedding_array).float()

        if self.normalize:
            embeddings = functional.normalize(embeddings, p=2, dim=-1)

        return embeddings

    def _iter_base_dataset_from_skip(self) -> Iterator[Any]:
        """Iterate source rows, using dataset-native skip/select when available."""
        skip_rows = int(self.skip_source_rows)
        if skip_rows <= 0:
            return iter(self.base_dataset)

        if hasattr(self.base_dataset, "skip"):
            skipped_dataset = self.base_dataset.skip(skip_rows)
            return iter(skipped_dataset)

        if hasattr(self.base_dataset, "select"):
            try:
                source_total = len(self.base_dataset)
            except (TypeError, NotImplementedError):
                source_total = None
            if source_total is not None:
                selected_dataset = self.base_dataset.select(range(skip_rows, source_total))
                return iter(selected_dataset)

        def fallback_iter() -> Iterator[Any]:
            for row_index, item in enumerate(self.base_dataset):
                if row_index >= skip_rows:
                    yield item

        return fallback_iter()

    def _iter_source_rows(self) -> Iterator[tuple[int, Any]]:
        """Yield ``(absolute_source_rows_seen, item)`` pairs."""
        self.source_rows_seen = int(self.skip_source_rows)
        for item in self._iter_base_dataset_from_skip():
            self.source_rows_seen += 1
            yield self.source_rows_seen, item

    def _row_text_chunks(self, row: tuple[int, Any]) -> tuple[int, list[str]]:
        """Extract and split text for one source row."""
        source_row_index, item = row
        if isinstance(item, dict):
            text = item.get(self.text_column, "")
        else:
            text = item

        if not text or (isinstance(text, str) and len(text.strip()) == 0):
            return source_row_index, []

        return source_row_index, self._split_text(text)

    def _iter_chunked_rows(self) -> Iterator[tuple[int, list[str]]]:
        """Yield source-row-indexed chunk lists, optionally split in parallel."""
        source_rows = self._iter_source_rows()

        if self.num_chunk_workers == 1:
            for row in source_rows:
                yield self._row_text_chunks(row)
            return

        def flush_rows(
            executor: ThreadPoolExecutor,
            rows: list[tuple[int, Any]],
        ) -> Iterator[tuple[int, list[str]]]:
            yield from executor.map(self._row_text_chunks, rows)

        with ThreadPoolExecutor(max_workers=self.num_chunk_workers) as executor:
            pending_rows: list[tuple[int, Any]] = []
            for row in source_rows:
                pending_rows.append(row)
                if len(pending_rows) >= self.chunk_worker_batch_size:
                    yield from flush_rows(executor, pending_rows)
                    pending_rows = []

            if pending_rows:
                yield from flush_rows(executor, pending_rows)

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
        texts_buffer: list[str] = []
        n_samples_skipped = 0
        n_samples_yielded = 0
        self.source_rows_seen = int(self.skip_source_rows)
        self.safe_resume_cursor = {
            "source_rows_seen": int(self.skip_source_rows),
            "chunks_seen": int(self.skip_chunk_offset),
        }
        buffer_safe_source_rows_seen = int(self.skip_source_rows)
        buffer_safe_chunks_seen = int(self.skip_chunk_offset)
        self._start_encode_pool()

        try:
            stop_after_buffer = False
            for source_row_index, text_chunks in self._iter_chunked_rows():
                chunks_seen = self.skip_chunk_offset + n_samples_skipped + n_samples_yielded

                if self.max_samples is not None and chunks_seen >= self.max_samples:
                    break

                for text_chunk in text_chunks:
                    chunks_seen = (
                        self.skip_chunk_offset
                        + n_samples_skipped
                        + n_samples_yielded
                        + len(texts_buffer)
                    )

                    if self.max_samples is not None and chunks_seen >= self.max_samples:
                        stop_after_buffer = True
                        break

                    if n_samples_skipped < self.skip_samples:
                        n_samples_skipped += 1
                        continue

                    texts_buffer.append(text_chunk)

                if not stop_after_buffer:
                    buffer_safe_source_rows_seen = source_row_index
                    buffer_safe_chunks_seen = (
                        self.skip_chunk_offset
                        + n_samples_skipped
                        + n_samples_yielded
                        + len(texts_buffer)
                    )

                # When buffer is full, encode and yield
                if len(texts_buffer) >= self.buffer_size:
                    # Shuffle buffer
                    random.shuffle(texts_buffer)

                    # Encode in batches
                    for i in range(0, len(texts_buffer), self.embedding_batch_size):
                        batch_texts = texts_buffer[i : i + self.embedding_batch_size]

                        embeddings = self._embed_text_batch(batch_texts)
                        n_samples_yielded += embeddings.shape[0]
                        if i + self.embedding_batch_size >= len(texts_buffer):
                            self.safe_resume_cursor = {
                                "source_rows_seen": int(buffer_safe_source_rows_seen),
                                "chunks_seen": int(buffer_safe_chunks_seen),
                            }
                        yield embeddings

                    texts_buffer = []

                if stop_after_buffer:
                    break

            # Yield remaining items
            if texts_buffer:
                for i in range(0, len(texts_buffer), self.embedding_batch_size):
                    batch_texts = texts_buffer[i : i + self.embedding_batch_size]
                    embeddings = self._embed_text_batch(batch_texts)
                    n_samples_yielded += embeddings.shape[0]
                    if i + self.embedding_batch_size >= len(texts_buffer):
                        self.safe_resume_cursor = {
                            "source_rows_seen": int(buffer_safe_source_rows_seen),
                            "chunks_seen": int(buffer_safe_chunks_seen),
                        }
                    yield embeddings
        finally:
            self._stop_encode_pool()


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
    text_split_strategy: str = "token",
    min_sentences_per_chunk: int = 1,
    normalize: bool = True,
    seed: int = 42,
    streaming: bool = True,
    num_proc: int | None = 16,
    max_samples: int | None = None,
    skip_samples: int = 0,
    task: str = "clustering",
    skip_source_rows: int = 0,
    skip_chunk_offset: int = 0,
    encode_method: str = "encode",
    sanitize_urls: bool = False,
    num_chunk_workers: int = 1,
    chunk_worker_batch_size: int = 1024,
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
    text_split_strategy : {"token", "paragraph"}, default="token"
        How to split text before embedding. ``"paragraph"`` uses
        blank-line-separated paragraphs and filters by sentence count.
    min_sentences_per_chunk : int, default=1
        Minimum sentence-like spans per paragraph chunk.
    normalize : bool, default=True
        Whether to L2-normalize embeddings before yielding/saving them
    seed : int, default=42
        Random seed for streaming buffer shuffling
    streaming : bool, default=True
        Use streaming mode (set to False for small datasets)
    num_proc : int or None, default=16
        Number of dataset-loading processes for non-streaming HuggingFace
        datasets. Ignored when ``streaming=True`` because HuggingFace Datasets
        does not support streaming with ``num_proc``.
    max_samples : int or None, default=None
        Maximum samples to stream
    skip_samples : int, default=0
        Number of text chunks to skip before encoding
    task : str, default="clustering"
        Task type for Jina embeddings (e.g., "clustering", "retrieval")
    skip_source_rows : int, default=0
        Number of raw source rows to skip before chunking.
    skip_chunk_offset : int, default=0
        Number of text chunks known before ``skip_source_rows``.
    encode_method : {"encode", "document", "query"}, default="encode"
        SentenceTransformer method to use for text inputs.
    sanitize_urls : bool, default=False
        Replace URLs in text chunks with ``"[URL]"`` before encoding.
    num_chunk_workers : int, default=1
        Number of threads used to split source rows into text chunks.
    chunk_worker_batch_size : int, default=1024
        Source rows collected before dispatching one parallel chunking batch.
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

    load_kwargs = {
        "split": split,
        "streaming": streaming,
        **hf_kwargs,
    }
    if not streaming and num_proc is not None:
        load_kwargs["num_proc"] = num_proc

    hf_dataset = load_dataset(*load_args, **load_kwargs)

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
        text_split_strategy=text_split_strategy,
        min_sentences_per_chunk=min_sentences_per_chunk,
        normalize=normalize,
        seed=seed,
        max_samples=max_samples,
        skip_samples=skip_samples,
        task=task,
        skip_source_rows=skip_source_rows,
        skip_chunk_offset=skip_chunk_offset,
        encode_method=encode_method,
        sanitize_urls=sanitize_urls,
        num_chunk_workers=num_chunk_workers,
        chunk_worker_batch_size=chunk_worker_batch_size,
    )
