#!/usr/bin/env python3
"""Run the SAETopic pretraining workflow from a YAML config.

This script is intentionally independent from ``examples/``. It only imports
library code from ``src/saetopic`` and keeps pretraining-specific orchestration
under ``pretrain/``.
"""

from __future__ import annotations

import argparse
import csv
import gc
import html
import json
import os
import re
import shutil
import time
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from scipy.sparse import csr_matrix, load_npz, save_npz
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from torch.utils.data import random_split

from saetopic import SAETopicModel
from saetopic.evaluation import (
    compute_coherence_rating,
    compute_intruder_detection,
    compute_wmd_diversity,
    iter_top_words_files,
    load_saetm_word2vec_cache,
    load_top_words_file,
    summarize_metric,
)
from saetopic.merging import preload_word_embedding_model
from saetopic.sae.loaders import SAECheckpoint
from saetopic.training import (
    StreamingEmbeddingDataset,
    compute_and_save_embeddings,
    create_streaming_dataset,
    train_sae,
)
from saetopic.training.data import EmbeddingDataset
from saetopic.training.train_sae import TrainingConfig

console = Console()
TEXT_URL_PATTERN = re.compile(r"(?i)\b(?:https?://|www\.)\S+")


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping config in {path}")
    return data


def resolve_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else Path.cwd() / path


def get_seed(config: dict[str, Any]) -> int:
    return int(config.get("project", {}).get("seed", 42))


def torch_dtype(name: str | None):
    if name is None:
        return None
    normalized = str(name).lower()
    if normalized in {"none", "auto"}:
        return None
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"Unknown torch dtype: {name}")
    return mapping[normalized]


def model_device(setting: str | None) -> str:
    if setting in {None, "auto"}:
        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(setting)


def encode_device(setting: Any) -> str | list[str] | None:
    if setting is None:
        return None
    if isinstance(setting, list):
        return [str(device) for device in setting]
    if setting == "auto_multi_cuda":
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            return [f"cuda:{idx}" for idx in range(torch.cuda.device_count())]
        return None
    if setting == "auto":
        return None
    return str(setting)


def build_embedder(config: dict[str, Any]):
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from sentence_transformers import SentenceTransformer

    model_cfg = config["embedding_model"]
    device = model_device(model_cfg.get("device"))
    dtype = torch_dtype(model_cfg.get("dtype"))

    model_kwargs = dict(model_cfg.get("model_kwargs") or {})
    if dtype is not None and device.startswith("cuda"):
        model_kwargs["dtype"] = dtype

    config_kwargs = dict(model_cfg.get("config_kwargs") or {})
    attn_implementation = model_cfg.get("attn_implementation")
    if attn_implementation:
        config_kwargs["_attn_implementation"] = attn_implementation

    kwargs: dict[str, Any] = {
        "trust_remote_code": bool(model_cfg.get("trust_remote_code", True)),
        "device": device,
    }
    if model_kwargs:
        kwargs["model_kwargs"] = model_kwargs
    if config_kwargs:
        kwargs["config_kwargs"] = config_kwargs
    if model_cfg.get("truncate_dim") is not None:
        kwargs["truncate_dim"] = int(model_cfg["truncate_dim"])

    embedder = SentenceTransformer(model_cfg["name"], **kwargs)
    if model_cfg.get("max_seq_length") is not None:
        embedder.max_seq_length = int(model_cfg["max_seq_length"])
    return embedder


def split_text_by_words(text: str, chunk_size: int | None, overlap: int) -> list[str]:
    """Split text into whitespace-word chunks for the offline chunk stage."""
    text = text.strip()
    if not text:
        return []
    if chunk_size is None:
        return [text]

    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def split_text_by_paragraphs(text: str, min_sentences: int) -> list[str]:
    """Split text into paragraph chunks with a sentence-count filter."""
    text = text.strip()
    if not text:
        return []

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
        if count_sentences(paragraph) >= min_sentences
    ]


def count_sentences(text: str) -> int:
    """Count sentence-like spans without external NLP resources."""
    spans = re.findall(r"[^.!?\n]+[.!?]+", text)
    if spans:
        return len(spans)
    return 1 if text.strip() else 0


def run_chunks(config: dict[str, Any]) -> tuple[int, Path]:
    """Create a flat, saved-to-disk text chunk dataset from the configured source."""
    try:
        from datasets import disable_progress_bar, load_dataset, load_from_disk
    except ImportError as exc:
        raise ImportError(
            "datasets package is required for the chunks stage. "
            "Install with the training extras."
        ) from exc

    disable_progress_bar()

    dataset_cfg = config["dataset"]
    chunks_cfg = config.get("chunks", {})
    chunks_path = resolve_path(chunks_cfg.get("path"))
    if chunks_path is None:
        raise ValueError("chunks.path is required for the chunks stage")

    if chunks_path.exists():
        try:
            chunk_dataset = load_from_disk(str(chunks_path))
        except Exception as exc:
            if not bool(chunks_cfg.get("overwrite", False)):
                raise ValueError(
                    f"chunks.path exists but is not a readable saved dataset: {chunks_path}. "
                    "Set chunks.overwrite: true to rebuild it."
                ) from exc
        else:
            if bool(chunks_cfg.get("resume", True)) and not bool(
                chunks_cfg.get("overwrite", False)
            ):
                console.print(
                    f"Found existing chunk dataset at {chunks_path}: "
                    f"{len(chunk_dataset):,} chunks"
                )
                return int(len(chunk_dataset)), chunks_path

    temp_path = chunks_path.with_name(f"{chunks_path.name}.building")
    if temp_path.exists():
        if not bool(chunks_cfg.get("overwrite", False)):
            raise ValueError(
                f"Temporary chunk build directory exists: {temp_path}. "
                "Remove it or set chunks.overwrite: true."
            )
        shutil.rmtree(temp_path)
    if chunks_path.exists() and bool(chunks_cfg.get("overwrite", False)):
        shutil.rmtree(chunks_path)

    strategy = str(chunks_cfg.get("strategy", dataset_cfg.get("text_split_strategy", "word")))
    if strategy not in {"word", "paragraph"}:
        raise ValueError(
            "chunks.strategy currently supports 'word' or 'paragraph'. "
            "Use word for fast FineWiki preprocessing."
        )

    text_column = chunks_cfg.get("text_column", dataset_cfg.get("text_column", "text"))
    chunk_size = chunks_cfg.get("chunk_size", dataset_cfg.get("text_chunk_size", 384))
    chunk_size = None if chunk_size is None else int(chunk_size)
    overlap = int(chunks_cfg.get("chunk_overlap", dataset_cfg.get("text_chunk_overlap", 0)))
    min_words = int(chunks_cfg.get("min_words", 1))
    min_sentences = int(
        chunks_cfg.get(
            "min_sentences_per_chunk",
            dataset_cfg.get("min_sentences_per_chunk", 1),
        )
    )
    sanitize_urls = bool(chunks_cfg.get("sanitize_urls", dataset_cfg.get("sanitize_urls", True)))
    num_proc = chunks_cfg.get("num_proc", dataset_cfg.get("num_proc", 16))
    map_batch_size = int(chunks_cfg.get("map_batch_size", 1000))
    max_source_rows = chunks_cfg.get("max_source_rows")

    load_args = [dataset_cfg["name"]]
    if dataset_cfg.get("subset") is not None:
        load_args.append(dataset_cfg.get("subset"))
    load_kwargs = {
        "split": dataset_cfg.get("split", "train"),
        "streaming": False,
    }
    if num_proc is not None:
        load_kwargs["num_proc"] = int(num_proc)

    console.print(
        f"Loading source dataset {dataset_cfg['name']} for chunking "
        f"with strategy={strategy}"
    )
    source_dataset = load_dataset(*load_args, **load_kwargs)
    if max_source_rows is not None:
        source_dataset = source_dataset.select(range(min(int(max_source_rows), len(source_dataset))))

    remove_columns = list(source_dataset.column_names)

    def chunk_batch(batch: dict[str, list[Any]], indices: list[int]) -> dict[str, list[Any]]:
        out_source_row: list[int] = []
        out_chunk_index: list[int] = []
        out_text: list[str] = []
        out_n_words: list[int] = []

        for row_index, raw_text in zip(indices, batch[text_column]):
            text = "" if raw_text is None else str(raw_text)
            if sanitize_urls:
                text = TEXT_URL_PATTERN.sub("[URL]", text)

            if strategy == "paragraph":
                chunks = split_text_by_paragraphs(text, min_sentences=min_sentences)
            else:
                chunks = split_text_by_words(text, chunk_size=chunk_size, overlap=overlap)

            for chunk_index, chunk in enumerate(chunks):
                n_words = len(chunk.split())
                if n_words < min_words:
                    continue
                out_source_row.append(int(row_index))
                out_chunk_index.append(int(chunk_index))
                out_text.append(chunk)
                out_n_words.append(int(n_words))

        return {
            "source_row": out_source_row,
            "chunk_index": out_chunk_index,
            "text": out_text,
            "n_words": out_n_words,
        }

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        progress.add_task("[cyan]Chunking FineWiki", total=None)
        chunk_dataset = source_dataset.map(
            chunk_batch,
            batched=True,
            batch_size=map_batch_size,
            with_indices=True,
            num_proc=None if num_proc is None else int(num_proc),
            remove_columns=remove_columns,
        )
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        progress.add_task("[cyan]Saving chunk dataset", total=None)
        chunk_dataset.save_to_disk(
            str(temp_path),
            num_proc=None if num_proc is None else int(num_proc),
        )

    manifest = {
        "format": "saetopic.text_chunks.v1",
        "source_dataset": dataset_cfg["name"],
        "subset": dataset_cfg.get("subset"),
        "split": dataset_cfg.get("split", "train"),
        "text_column": text_column,
        "strategy": strategy,
        "chunk_size": chunk_size,
        "chunk_overlap": overlap,
        "min_words": min_words,
        "min_sentences_per_chunk": min_sentences,
        "sanitize_urls": sanitize_urls,
        "n_chunks": int(len(chunk_dataset)),
    }
    (temp_path / "chunk_manifest.json").write_text(json.dumps(manifest, indent=2))
    temp_path.replace(chunks_path)

    console.print(f"Saved {len(chunk_dataset):,} text chunks to {chunks_path}")
    return int(len(chunk_dataset)), chunks_path


def run_embeddings(config: dict[str, Any]) -> tuple[int, int]:
    dataset_cfg = config["dataset"]
    embedding_cfg = config["embeddings"]

    embedder = build_embedder(config)
    if dataset_cfg.get("source") == "chunks":
        try:
            from datasets import load_from_disk
        except ImportError as exc:
            raise ImportError(
                "datasets package is required to load precomputed chunks."
            ) from exc

        chunks_cfg = config.get("chunks", {})
        chunks_path = resolve_path(dataset_cfg.get("chunks_path") or chunks_cfg.get("path"))
        if chunks_path is None:
            raise ValueError("dataset.chunks_path or chunks.path is required")
        chunk_dataset = load_from_disk(str(chunks_path))
        dataset = StreamingEmbeddingDataset(
            chunk_dataset,
            embedder,
            text_column=dataset_cfg.get("text_column", "text"),
            buffer_size=int(dataset_cfg.get("buffer_size", 10000)),
            embedding_batch_size=int(dataset_cfg.get("embedding_batch_size", 256)),
            encode_batch_size=int(dataset_cfg.get("encode_batch_size", 32)),
            encode_device=encode_device(dataset_cfg.get("encode_device")),
            encode_chunk_size=dataset_cfg.get("encode_chunk_size"),
            text_chunk_size=None,
            text_split_strategy="word",
            normalize=bool(dataset_cfg.get("normalize", True)),
            seed=get_seed(config),
            max_samples=dataset_cfg.get("max_samples"),
            task=dataset_cfg.get("task", "clustering"),
            encode_method=dataset_cfg.get("encode_method", "encode"),
            sanitize_urls=bool(dataset_cfg.get("sanitize_urls", False)),
            num_chunk_workers=1,
            prefetch_buffers=int(dataset_cfg.get("prefetch_buffers", 2)),
        )
    else:
        dataset = create_streaming_dataset(
            dataset_name=dataset_cfg["name"],
            subset=dataset_cfg.get("subset"),
            split=dataset_cfg.get("split", "train"),
            text_column=dataset_cfg.get("text_column", "text"),
            embedder=embedder,
            buffer_size=int(dataset_cfg.get("buffer_size", 10000)),
            embedding_batch_size=int(dataset_cfg.get("embedding_batch_size", 256)),
            encode_batch_size=int(dataset_cfg.get("encode_batch_size", 32)),
            encode_device=encode_device(dataset_cfg.get("encode_device")),
            encode_chunk_size=dataset_cfg.get("encode_chunk_size"),
            text_chunk_size=dataset_cfg.get("text_chunk_size"),
            text_chunk_overlap=int(dataset_cfg.get("text_chunk_overlap", 0)),
            text_split_strategy=dataset_cfg.get("text_split_strategy", "token"),
            min_sentences_per_chunk=int(dataset_cfg.get("min_sentences_per_chunk", 1)),
            normalize=bool(dataset_cfg.get("normalize", True)),
            seed=get_seed(config),
            streaming=bool(dataset_cfg.get("streaming", True)),
            num_proc=dataset_cfg.get("num_proc", 16),
            max_samples=dataset_cfg.get("max_samples"),
            task=dataset_cfg.get("task", "clustering"),
            encode_method=dataset_cfg.get("encode_method", "encode"),
            sanitize_urls=bool(dataset_cfg.get("sanitize_urls", False)),
            num_chunk_workers=int(dataset_cfg.get("num_chunk_workers", 1)),
            chunk_worker_batch_size=int(dataset_cfg.get("chunk_worker_batch_size", 1024)),
            prefetch_buffers=int(dataset_cfg.get("prefetch_buffers", 2)),
        )

    return compute_and_save_embeddings(
        dataset=dataset,
        output_path=resolve_path(embedding_cfg["path"]),
        chunk_size=int(embedding_cfg.get("chunk_size", 10000)),
        resume=bool(embedding_cfg.get("resume", True)),
    )


def training_config_from_yaml(config: dict[str, Any], input_dim: int) -> TrainingConfig:
    training_cfg = dict(config["sae"].get("training") or {})
    training_cfg["input_dim"] = input_dim
    if training_cfg.get("output_dir") is not None:
        training_cfg["output_dir"] = str(resolve_path(training_cfg["output_dir"]))

    valid_fields = {field.name for field in fields(TrainingConfig)}
    filtered = {key: value for key, value in training_cfg.items() if key in valid_fields}
    return TrainingConfig(**filtered)


def run_train_sae(config: dict[str, Any]):
    sae_cfg = config["sae"]
    embeddings_path = resolve_path(config["embeddings"]["path"])
    if embeddings_path is None:
        raise ValueError("embeddings.path is required")

    full_dataset = EmbeddingDataset.from_file(
        embeddings_path,
        normalize=bool(sae_cfg.get("normalize_embeddings", True)),
        mmap_mode=sae_cfg.get("mmap_mode", "r"),
    )
    input_dim = int(full_dataset.embedding_dim)
    training_cfg = training_config_from_yaml(config, input_dim=input_dim)

    val_fraction = float(sae_cfg.get("val_fraction", 0.0) or 0.0)
    val_dataset = None
    train_dataset = full_dataset
    if val_fraction > 0 and len(full_dataset) > 1:
        val_size = max(1, int(val_fraction * len(full_dataset)))
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(training_cfg.seed),
        )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return train_sae(
        dataset=train_dataset,
        val_dataset=val_dataset,
        config=training_cfg,
    )


def checkpoint_path(config: dict[str, Any]) -> Path:
    topics_cfg = config["topics"]
    explicit = resolve_path(topics_cfg.get("checkpoint_path"))
    if explicit is not None:
        return explicit

    train_output = resolve_path(config["sae"]["training"]["output_dir"])
    if train_output is None:
        raise ValueError("topics.checkpoint_path or sae.training.output_dir is required")
    best = train_output / "best"
    return best if best.exists() else train_output / "final"


def load_vision_probe_inputs(probe_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Load image probe records from HF datasets, inline config, JSONL, CSV, or TXT."""
    records: list[dict[str, Any]] = []
    records.extend(load_vision_probe_hf_inputs(probe_cfg))

    inline_inputs = probe_cfg.get("inputs") or []
    for idx, item in enumerate(inline_inputs):
        if isinstance(item, str):
            records.append({"id": str(idx), "image": item})
        elif isinstance(item, dict):
            record = dict(item)
            record.setdefault("id", str(idx))
            records.append(record)
        else:
            raise TypeError("vision_probe.inputs entries must be strings or mappings")

    input_file = resolve_path(probe_cfg.get("input_file"))
    if input_file is None:
        return records
    if not input_file.exists():
        raise FileNotFoundError(f"vision_probe.input_file does not exist: {input_file}")

    suffix = input_file.suffix.lower()
    if suffix == ".jsonl":
        with input_file.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"Expected JSON object on line {line_idx + 1} of {input_file}")
                record.setdefault("id", str(len(records)))
                records.append(record)
    elif suffix == ".csv":
        with input_file.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                record = {key: value for key, value in row.items() if value not in {None, ""}}
                record.setdefault("id", str(len(records)))
                records.append(record)
    else:
        with input_file.open("r", encoding="utf-8") as f:
            for line in f:
                image = line.strip()
                if image:
                    records.append({"id": str(len(records)), "image": image})

    return records


def load_vision_probe_hf_inputs(probe_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Load image probe records directly from a Hugging Face dataset cache."""
    dataset_id = probe_cfg.get("hf_dataset")
    if not dataset_id:
        return []

    try:
        from datasets import DownloadConfig, load_dataset
    except ImportError as exc:
        raise ImportError("datasets is required for vision_probe.hf_dataset") from exc

    subset = probe_cfg.get("hf_subset")
    split = str(probe_cfg.get("hf_split", "test"))
    image_column = str(probe_cfg.get("image_column", "image"))
    label_column = probe_cfg.get("label_column", "label")
    max_samples = probe_cfg.get("max_samples")
    max_samples = None if max_samples in {None, 0} else int(max_samples)
    max_per_label = probe_cfg.get("max_per_label")
    max_per_label = None if max_per_label in {None, 0} else int(max_per_label)

    load_args = [dataset_id]
    if subset is not None:
        load_args.append(subset)
    load_kwargs: dict[str, Any] = {"split": split}
    if probe_cfg.get("hf_cache_dir") is not None:
        load_kwargs["cache_dir"] = str(resolve_path(probe_cfg["hf_cache_dir"]))
    if bool(probe_cfg.get("hf_local_files_only", False)):
        load_kwargs["download_config"] = DownloadConfig(local_files_only=True)

    dataset = load_dataset(*load_args, **load_kwargs)
    label_names = None
    if label_column and label_column in dataset.features:
        label_names = getattr(dataset.features[label_column], "names", None)

    counts_by_label: dict[str, int] = {}
    records: list[dict[str, Any]] = []
    for row_idx, row in enumerate(dataset):
        if image_column not in row:
            raise ValueError(
                f"vision_probe.image_column={image_column!r} not found in {dataset_id}"
            )

        raw_label = row.get(label_column) if label_column else None
        label_id = int(raw_label) if isinstance(raw_label, int) else raw_label
        if label_names is not None and isinstance(label_id, int):
            label = label_names[label_id]
        elif raw_label is None:
            label = None
        else:
            label = str(raw_label)

        count_key = label if label is not None else "__none__"
        if max_per_label is not None and counts_by_label.get(count_key, 0) >= max_per_label:
            continue

        image = row[image_column]
        if hasattr(image, "convert"):
            image = image.convert("RGB")

        local_idx = counts_by_label.get(count_key, 0)
        record_id = f"{label or 'sample'}_{local_idx:04d}"
        records.append(
            {
                "id": record_id,
                "image": f"hf://{dataset_id}/{split}/{row_idx}",
                "_image": image,
                "label": label,
                "label_id": label_id,
            }
        )
        counts_by_label[count_key] = local_idx + 1
        if max_samples is not None and len(records) >= max_samples:
            break

    console.print(
        f"Loaded {len(records):,} vision probe images from {dataset_id} "
        f"split={split} via Hugging Face datasets cache"
    )
    return records


def build_vision_probe_embedder(config: dict[str, Any]):
    """Build a Jina/SentenceTransformer embedder configured for vision probing."""
    probe_cfg = config.get("vision_probe", {})
    vision_config = dict(config)
    model_cfg = dict(config["embedding_model"])
    model_kwargs = dict(model_cfg.get("model_kwargs") or {})
    model_kwargs["modality"] = probe_cfg.get("modality", "vision")
    model_kwargs["default_task"] = probe_cfg.get(
        "task",
        model_kwargs.get("default_task", model_cfg.get("task", "clustering")),
    )
    model_cfg["model_kwargs"] = model_kwargs
    if probe_cfg.get("device") is not None:
        model_cfg["device"] = probe_cfg["device"]
    if probe_cfg.get("truncate_dim") is not None:
        model_cfg["truncate_dim"] = int(probe_cfg["truncate_dim"])
    vision_config["embedding_model"] = model_cfg
    return build_embedder(vision_config)


def _vision_probe_payload(record: dict[str, Any]) -> Any:
    if "_image" in record:
        image = record["_image"]
    else:
        image = record.get("image") or record.get("path") or record.get("url")
    text = record.get("text") or record.get("caption")
    if image is None:
        raise ValueError(f"Vision probe record is missing image/path/url: {record}")
    if text:
        return (str(text), image)
    return image


def _encode_probe_payloads(
    embedder,
    payloads: list[Any],
    *,
    encode_method: str,
    batch_size: int,
) -> np.ndarray:
    encode_kwargs = {"batch_size": batch_size, "show_progress_bar": True}
    if encode_method == "document":
        encode_document = getattr(embedder, "encode_document", None)
        if callable(encode_document):
            embeddings = encode_document(payloads, **encode_kwargs)
        else:
            embeddings = embedder.encode(payloads, prompt_name="document", **encode_kwargs)
    elif encode_method == "query":
        encode_query = getattr(embedder, "encode_query", None)
        if callable(encode_query):
            embeddings = encode_query(payloads, **encode_kwargs)
        else:
            embeddings = embedder.encode(payloads, prompt_name="query", **encode_kwargs)
    elif encode_method == "encode":
        embeddings = embedder.encode(payloads, **encode_kwargs)
    else:
        raise ValueError("vision_probe.encode_method must be 'document', 'query', or 'encode'")
    return np.asarray(embeddings, dtype=np.float32)


def _write_vision_probe_visual_bow(
    out_dir: Path,
    sample_results: list[dict[str, Any]],
    n_features: int,
) -> dict[str, str]:
    """Write image-level and class-level visual-word artifacts from SAE activations."""
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    row_ids: list[str] = []
    row_labels: list[str | None] = []

    class_image_counts: dict[str, int] = {}
    class_occurrence_counts: dict[str, int] = {}
    class_activation_sums: dict[str, float] = {}
    class_feature_stats: dict[str, dict[int, dict[str, float]]] = {}

    for row_idx, sample in enumerate(sample_results):
        row_ids.append(str(sample.get("id", row_idx)))
        label = sample.get("label")
        label_key = str(label) if label is not None else "__unlabeled__"
        row_labels.append(None if label is None else str(label))
        class_image_counts[label_key] = class_image_counts.get(label_key, 0) + 1

        for rank, item in enumerate(sample.get("top_features", []), start=1):
            feature = int(item["feature"])
            activation = float(item["activation"])
            rows.append(row_idx)
            cols.append(feature)
            data.append(activation)

            class_occurrence_counts[label_key] = class_occurrence_counts.get(label_key, 0) + 1
            class_activation_sums[label_key] = (
                class_activation_sums.get(label_key, 0.0) + activation
            )
            stats = class_feature_stats.setdefault(label_key, {}).setdefault(
                feature,
                {"image_count": 0.0, "activation_sum": 0.0, "best_rank": float(rank)},
            )
            stats["image_count"] += 1.0
            stats["activation_sum"] += activation
            stats["best_rank"] = min(stats["best_rank"], float(rank))

    matrix_width = max(n_features, max(cols) + 1 if cols else 0)
    visual_bow = csr_matrix(
        (data, (rows, cols)),
        shape=(len(sample_results), matrix_width),
        dtype=np.float32,
    )
    visual_bow_path = out_dir / "visual_bow.npz"
    save_npz(visual_bow_path, visual_bow)

    visual_bow_meta = {
        "matrix": str(visual_bow_path),
        "n_images": len(sample_results),
        "n_features": matrix_width,
        "row_ids": row_ids,
        "labels": row_labels,
        "weighting": "sae_activation",
        "source": "vision_probe.top_features",
    }
    visual_bow_meta_path = out_dir / "visual_bow_meta.json"
    visual_bow_meta_path.write_text(
        json.dumps(visual_bow_meta, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    class_summary_path = out_dir / "class_summary.csv"
    with class_summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "label",
                "n_images",
                "unique_features",
                "total_feature_occurrences",
                "mean_active_features_per_image",
                "activation_sum",
            ],
        )
        writer.writeheader()
        for label_key in sorted(class_image_counts):
            n_images = class_image_counts[label_key]
            total_occurrences = class_occurrence_counts.get(label_key, 0)
            writer.writerow(
                {
                    "label": label_key,
                    "n_images": n_images,
                    "unique_features": len(class_feature_stats.get(label_key, {})),
                    "total_feature_occurrences": total_occurrences,
                    "mean_active_features_per_image": total_occurrences / max(n_images, 1),
                    "activation_sum": class_activation_sums.get(label_key, 0.0),
                }
            )

    class_distribution_path = out_dir / "class_feature_distribution.csv"
    with class_distribution_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "label",
                "n_images",
                "feature",
                "image_count",
                "image_fraction",
                "activation_sum",
                "mean_activation",
                "activation_share",
                "best_rank",
            ],
        )
        writer.writeheader()
        for label_key in sorted(class_feature_stats):
            n_images = class_image_counts[label_key]
            total_activation = class_activation_sums.get(label_key, 0.0)
            for feature, stats in sorted(
                class_feature_stats[label_key].items(),
                key=lambda item: (
                    -item[1]["image_count"],
                    -item[1]["activation_sum"],
                    item[0],
                ),
            ):
                image_count = int(stats["image_count"])
                activation_sum = stats["activation_sum"]
                writer.writerow(
                    {
                        "label": label_key,
                        "n_images": n_images,
                        "feature": feature,
                        "image_count": image_count,
                        "image_fraction": image_count / max(n_images, 1),
                        "activation_sum": activation_sum,
                        "mean_activation": activation_sum / max(image_count, 1),
                        "activation_share": (
                            activation_sum / total_activation if total_activation > 0 else 0.0
                        ),
                        "best_rank": int(stats["best_rank"]),
                    }
                )

    return {
        "visual_bow": str(visual_bow_path),
        "visual_bow_meta": str(visual_bow_meta_path),
        "class_summary": str(class_summary_path),
        "class_feature_distribution": str(class_distribution_path),
    }


def vision_config(config: dict[str, Any]) -> dict[str, Any]:
    cfg = config.get("vision")
    if not isinstance(cfg, dict):
        raise ValueError("vision config is required for vision_vocab, vision_bow, and vision_emission")
    return cfg


def vision_out_dir(config: dict[str, Any]) -> Path:
    cfg = vision_config(config)
    out_dir = resolve_path(cfg.get("out_dir", "results/vision_topics"))
    if out_dir is None:
        raise ValueError("vision.out_dir is required")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def vision_checkpoint_path(config: dict[str, Any]) -> Path:
    cfg = vision_config(config)
    explicit = resolve_path(cfg.get("checkpoint_path"))
    if explicit is not None:
        return explicit
    return checkpoint_path(config)


def build_vision_embedder(config: dict[str, Any]):
    """Build the image embedding model used for SAE theta in vision topic stages."""
    cfg = vision_config(config)
    probe_config = dict(config)
    probe_config["vision_probe"] = {
        "modality": cfg.get("modality", "vision"),
        "task": cfg.get("task", "clustering"),
        "device": cfg.get("device"),
        "truncate_dim": cfg.get("truncate_dim"),
    }
    return build_vision_probe_embedder(probe_config)


def _image_from_vision_record(record: dict[str, Any]):
    if "_image" in record:
        image = record["_image"]
        return image.convert("RGB") if hasattr(image, "convert") else image

    image_ref = record.get("image") or record.get("path") or record.get("url")
    if image_ref is None:
        raise ValueError(f"Vision record is missing image/path/url: {record}")
    image_ref = str(image_ref)
    if image_ref.startswith("hf://"):
        raise ValueError(
            f"{image_ref} cannot be reloaded without the HF image object. "
            "Load inputs from vision.hf_dataset in the same stage."
        )

    from PIL import Image

    if image_ref.startswith(("http://", "https://")):
        from urllib.request import urlopen

        with urlopen(image_ref) as response:  # noqa: S310 - user-provided image URL
            return Image.open(response).convert("RGB")
    return Image.open(resolve_path(image_ref) or image_ref).convert("RGB")


def _iter_dinov2_patch_batches(
    records: list[dict[str, Any]],
    tokenizer_cfg: dict[str, Any],
):
    """Yield normalized DINOv2 patch embeddings as ``(start_idx, batch_patches)``."""
    from transformers import AutoImageProcessor, AutoModel

    model_name = str(tokenizer_cfg.get("model", "facebook/dinov2-base"))
    device = torch.device(model_device(tokenizer_cfg.get("device", "auto")))
    dtype = torch_dtype(tokenizer_cfg.get("dtype"))
    model_dtype = dtype if dtype is not None and device.type == "cuda" else torch.float32
    local_files_only = bool(tokenizer_cfg.get("local_files_only", False))
    trust_remote_code = bool(tokenizer_cfg.get("trust_remote_code", False))
    image_batch_size = int(tokenizer_cfg.get("image_batch_size", 16))

    processor = AutoImageProcessor.from_pretrained(
        model_name,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    model = AutoModel.from_pretrained(
        model_name,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    model = model.to(device=device, dtype=model_dtype).eval()

    with torch.no_grad():
        for start in range(0, len(records), image_batch_size):
            batch_records = records[start : start + image_batch_size]
            images = [_image_from_vision_record(record) for record in batch_records]
            inputs = processor(images=images, return_tensors="pt")
            inputs = {
                key: value.to(device=device)
                for key, value in inputs.items()
                if isinstance(value, torch.Tensor)
            }
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(dtype=model_dtype)
            outputs = model(**inputs)
            hidden = outputs.last_hidden_state
            patches = hidden[:, 1:, :].float()
            patches = torch.nn.functional.normalize(patches, p=2, dim=-1)
            yield start, patches.cpu().numpy().astype(np.float32)


def _sample_patch_rows(
    patches: np.ndarray,
    *,
    max_rows: int,
    rng: np.random.Generator,
) -> np.ndarray:
    flat = patches.reshape(-1, patches.shape[-1])
    if max_rows <= 0 or flat.shape[0] <= max_rows:
        return flat
    indices = rng.choice(flat.shape[0], size=max_rows, replace=False)
    return flat[indices]


def run_vision_vocab(config: dict[str, Any]) -> None:
    """Build a DINOv2 patch-token KMeans visual vocabulary."""
    cfg = vision_config(config)
    tokenizer_cfg = dict(cfg.get("visual_tokenizer") or {})
    out_dir = vision_out_dir(config)
    records = load_vision_probe_inputs(cfg)
    if not records:
        raise ValueError("vision_vocab needs inputs. Set vision.hf_dataset, inputs, or input_file.")

    from sklearn.cluster import MiniBatchKMeans

    seed = int(tokenizer_cfg.get("seed", get_seed(config)))
    rng = np.random.default_rng(seed)
    n_clusters = int(tokenizer_cfg.get("codebook_size", 4096))
    max_patch_samples = tokenizer_cfg.get("max_patch_samples", 1_000_000)
    max_patch_samples = None if max_patch_samples in {None, 0} else int(max_patch_samples)
    kmeans_batch_size = int(tokenizer_cfg.get("kmeans_batch_size", 8192))
    per_image_patch_sample = int(tokenizer_cfg.get("patches_per_image_sample", 0))

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=kmeans_batch_size,
        random_state=seed,
        n_init=3,
        reassignment_ratio=0.01,
    )
    initialized = False
    pending: list[np.ndarray] = []
    pending_rows = 0
    seen_patch_samples = 0
    patch_dim: int | None = None

    for _, patches in _iter_dinov2_patch_batches(records, tokenizer_cfg):
        patch_dim = int(patches.shape[-1])
        max_rows = per_image_patch_sample * patches.shape[0] if per_image_patch_sample > 0 else 0
        sampled = _sample_patch_rows(patches, max_rows=max_rows, rng=rng)
        if max_patch_samples is not None:
            remaining = max_patch_samples - seen_patch_samples
            if remaining <= 0:
                break
            if sampled.shape[0] > remaining:
                sampled = sampled[:remaining]
        seen_patch_samples += int(sampled.shape[0])

        if not initialized:
            pending.append(sampled)
            pending_rows += int(sampled.shape[0])
            if pending_rows < n_clusters:
                continue
            init_batch = np.concatenate(pending, axis=0)
            kmeans.partial_fit(init_batch)
            initialized = True
            pending.clear()
            pending_rows = 0
        else:
            kmeans.partial_fit(sampled)

    if not initialized:
        if not pending:
            raise ValueError("No DINOv2 patch embeddings were produced for vision vocabulary.")
        init_batch = np.concatenate(pending, axis=0)
        if init_batch.shape[0] < n_clusters:
            raise ValueError(
                f"vision.visual_tokenizer.codebook_size={n_clusters} exceeds sampled "
                f"patches={init_batch.shape[0]}. Lower codebook_size or use more images."
            )
        kmeans.partial_fit(init_batch)
        patch_dim = int(init_batch.shape[-1])

    centroids = np.asarray(kmeans.cluster_centers_, dtype=np.float32)
    centroids_path = out_dir / "visual_vocab_centroids.npy"
    np.save(centroids_path, centroids)

    meta = {
        "model": str(tokenizer_cfg.get("model", "facebook/dinov2-base")),
        "n_visual_words": int(centroids.shape[0]),
        "patch_dim": int(patch_dim or centroids.shape[1]),
        "n_images": len(records),
        "sampled_patches": seen_patch_samples,
        "assignment": "hard_nearest_centroid",
        "centroids": str(centroids_path),
        "seed": seed,
    }
    (out_dir / "visual_vocab_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    console.print(f"Wrote visual vocabulary to {centroids_path}")


def _load_visual_centroids(out_dir: Path, cfg: dict[str, Any]) -> np.ndarray:
    explicit = resolve_path((cfg.get("visual_tokenizer") or {}).get("centroids_path"))
    path = explicit or out_dir / "visual_vocab_centroids.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"Visual vocabulary not found: {path}. Run --stages vision_vocab first."
        )
    return np.load(path).astype(np.float32)


def _write_class_visual_ctfidf(
    out_dir: Path,
    visual_bow: csr_matrix,
    labels: list[str | None],
) -> Path:
    df = np.asarray((visual_bow > 0).sum(axis=0)).ravel().astype(np.float32)
    idf = np.log((1.0 + visual_bow.shape[0]) / (1.0 + np.clip(df, 1.0, None))) + 1.0
    labels_normalized = [label if label is not None else "__unlabeled__" for label in labels]
    rows: list[dict[str, Any]] = []
    for label in sorted(set(labels_normalized)):
        image_indices = [idx for idx, value in enumerate(labels_normalized) if value == label]
        class_bow = visual_bow[image_indices]
        tf = np.asarray(class_bow.sum(axis=0)).ravel().astype(np.float32)
        total = float(tf.sum())
        if total <= 0:
            continue
        tf = tf / total
        score = tf * idf
        image_count = np.asarray((class_bow > 0).sum(axis=0)).ravel()
        top_features = np.flatnonzero(score)
        top_features = top_features[np.argsort(score[top_features])[-50:][::-1]]
        for rank, feature in enumerate(top_features, start=1):
            global_fraction = float(df[feature] / max(visual_bow.shape[0], 1))
            class_fraction = float(image_count[feature] / max(len(image_indices), 1))
            rows.append(
                {
                    "label": label,
                    "rank": rank,
                    "visual_word": int(feature),
                    "ctfidf_score": float(score[feature]),
                    "class_image_count": int(image_count[feature]),
                    "class_image_fraction": class_fraction,
                    "global_image_count": int(df[feature]),
                    "global_image_fraction": global_fraction,
                    "image_fraction_lift": (
                        class_fraction / global_fraction if global_fraction > 0 else 0.0
                    ),
                }
            )

    out_path = out_dir / "class_visual_ctfidf.csv"
    with out_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "label",
            "rank",
            "visual_word",
            "ctfidf_score",
            "class_image_count",
            "class_image_fraction",
            "global_image_count",
            "global_image_fraction",
            "image_fraction_lift",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def run_vision_bow(config: dict[str, Any]) -> None:
    """Encode images as a Bag-of-Visual-Words over the learned DINOv2 codebook."""
    cfg = vision_config(config)
    tokenizer_cfg = dict(cfg.get("visual_tokenizer") or {})
    out_dir = vision_out_dir(config)
    records = load_vision_probe_inputs(cfg)
    if not records:
        raise ValueError("vision_bow needs inputs. Set vision.hf_dataset, inputs, or input_file.")

    from sklearn.metrics import pairwise_distances_argmin

    centroids = _load_visual_centroids(out_dir, cfg)
    n_visual_words = int(centroids.shape[0])
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    row_ids: list[str] = []
    labels: list[str | None] = []

    for start, patches in _iter_dinov2_patch_batches(records, tokenizer_cfg):
        batch_size, patches_per_image, dim = patches.shape
        flat = patches.reshape(batch_size * patches_per_image, dim)
        assignments = pairwise_distances_argmin(flat, centroids, metric="euclidean")
        assignments = assignments.reshape(batch_size, patches_per_image)
        for local_idx in range(batch_size):
            row_idx = start + local_idx
            counts = np.bincount(assignments[local_idx], minlength=n_visual_words)
            nz = np.flatnonzero(counts)
            rows.extend([row_idx] * len(nz))
            cols.extend(int(value) for value in nz)
            data.extend(float(counts[value]) for value in nz)
            record = records[row_idx]
            row_ids.append(str(record.get("id", row_idx)))
            label = record.get("label")
            labels.append(None if label is None else str(label))

    visual_bow = csr_matrix(
        (data, (rows, cols)),
        shape=(len(records), n_visual_words),
        dtype=np.float32,
    )
    visual_bow_path = out_dir / "visual_bow.npz"
    save_npz(visual_bow_path, visual_bow)

    df = np.asarray((visual_bow > 0).sum(axis=0)).ravel().astype(np.float32)
    np.save(out_dir / "visual_word_df.npy", df)
    idf = np.log(visual_bow.shape[0] / np.clip(df, 1.0, None))
    if idf.max() > 0:
        idf = idf / idf.max()
    np.save(out_dir / "visual_word_idf.npy", idf.astype(np.float32))

    meta = {
        "matrix": str(visual_bow_path),
        "n_images": len(records),
        "n_visual_words": n_visual_words,
        "row_ids": row_ids,
        "labels": labels,
        "weighting": "patch_count",
        "visual_vocabulary": str(out_dir / "visual_vocab_centroids.npy"),
        "source": "dinov2_patch_kmeans",
    }
    (out_dir / "visual_bow_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    class_ctfidf_path = _write_class_visual_ctfidf(out_dir, visual_bow, labels)
    summary = {
        "n_images": len(records),
        "n_visual_words": n_visual_words,
        "nnz": int(visual_bow.nnz),
        "mean_visual_words_per_image": float(np.diff(visual_bow.indptr).mean()),
        "outputs": {
            "visual_bow": str(visual_bow_path),
            "visual_bow_meta": str(out_dir / "visual_bow_meta.json"),
            "visual_word_df": str(out_dir / "visual_word_df.npy"),
            "visual_word_idf": str(out_dir / "visual_word_idf.npy"),
            "class_visual_ctfidf": str(class_ctfidf_path),
        },
    }
    (out_dir / "vision_bow_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    console.print(f"Wrote visual BoW to {visual_bow_path}")


def _prepare_vision_embeddings_for_sae(
    config: dict[str, Any],
    records: list[dict[str, Any]],
    sae_checkpoint: SAECheckpoint,
) -> np.ndarray:
    cfg = vision_config(config)
    embedder = build_vision_embedder(config)
    payloads = [_vision_probe_payload(record) for record in records]
    embeddings = _encode_probe_payloads(
        embedder,
        payloads,
        encode_method=str(cfg.get("encode_method", "document")),
        batch_size=int(cfg.get("batch_size", config["embedding_model"].get("batch_size", 16))),
    )
    if embeddings.shape[1] > sae_checkpoint.embedding_dim:
        truncate_dim = int(cfg.get("truncate_dim", sae_checkpoint.embedding_dim))
        if truncate_dim == sae_checkpoint.embedding_dim:
            embeddings = embeddings[:, : sae_checkpoint.embedding_dim]
    if bool(cfg.get("normalize", True)):
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.where(norms == 0, 1.0, norms)
    if embeddings.shape[1] != sae_checkpoint.embedding_dim:
        raise ValueError(
            f"Vision embeddings have dimension {embeddings.shape[1]}, but SAE expects "
            f"{sae_checkpoint.embedding_dim}. Set embedding_model.truncate_dim or "
            "vision.truncate_dim to match the SAE input_dim."
        )
    return embeddings.astype(np.float32)


def _save_sae_theta_csr(
    embeddings: np.ndarray,
    sae,
    out_path: Path,
    *,
    batch_size: int,
    device: str,
    theta_mode: str,
    top_k: int,
) -> None:
    dev = torch.device(device)
    sae_dtype = torch.float32 if dev.type == "cpu" else torch.bfloat16
    sae = sae.to(dev, dtype=sae_dtype).eval()
    n_features = int(getattr(sae, "n_features", 0) or 0)
    data_chunks: list[np.ndarray] = []
    index_chunks: list[np.ndarray] = []
    indptr = [0]

    with torch.no_grad():
        for start in range(0, len(embeddings), batch_size):
            batch_np = embeddings[start : start + batch_size]
            batch = torch.from_numpy(batch_np).to(dev, dtype=sae_dtype)
            if theta_mode == "sparse_topk" and hasattr(sae, "activate"):
                h = sae.encode(batch)
                theta, _ = sae.activate(h)
            else:
                theta = sae.encode(batch)
            theta = theta.float().clamp_min(0)
            n_features = max(n_features, int(theta.shape[1]))
            row_sums = theta.sum(dim=1).clamp_min(1e-8)
            k = min(top_k, theta.shape[1]) if top_k > 0 else theta.shape[1]
            values, indices = torch.topk(theta, k=k, dim=1, sorted=True)
            mask = values > 0
            row_idx = mask.nonzero(as_tuple=False)[:, 0] if mask.any() else torch.empty(0, dtype=torch.long, device=dev)
            if row_idx.numel() > 0:
                selected_values = values[mask] / row_sums.index_select(0, row_idx)
                selected_indices = indices[mask]
                data_chunks.append(selected_values.cpu().numpy().astype(np.float32, copy=False))
                index_chunks.append(selected_indices.cpu().numpy().astype(np.int32, copy=False))
                counts = torch.bincount(row_idx, minlength=len(batch_np)).cpu().numpy()
            else:
                counts = np.zeros(len(batch_np), dtype=np.int64)
            base = indptr[-1]
            indptr.extend((base + np.cumsum(counts, dtype=np.int64)).tolist())

    data = np.concatenate(data_chunks) if data_chunks else np.array([], dtype=np.float32)
    indices = np.concatenate(index_chunks) if index_chunks else np.array([], dtype=np.int32)
    theta_csr = csr_matrix(
        (data, indices, np.asarray(indptr, dtype=np.int64)),
        shape=(len(embeddings), n_features),
    )
    save_npz(out_path, theta_csr)


def run_vision_emission(config: dict[str, Any]) -> None:
    """Learn B_vis: SAE feature -> visual-word emission probabilities."""
    cfg = vision_config(config)
    emission_cfg = dict(cfg.get("emission") or {})
    out_dir = vision_out_dir(config)
    visual_bow_path = resolve_path(cfg.get("visual_bow_path")) or out_dir / "visual_bow.npz"
    if not visual_bow_path.exists():
        raise FileNotFoundError(f"Visual BoW not found: {visual_bow_path}. Run vision_bow first.")
    visual_bow = load_npz(visual_bow_path).tocsr().astype(np.float32)

    records = load_vision_probe_inputs(cfg)
    if not records:
        raise ValueError("vision_emission needs inputs. Set vision.hf_dataset, inputs, or input_file.")
    if len(records) != visual_bow.shape[0]:
        raise ValueError(
            f"Loaded {len(records)} images, but visual_bow has {visual_bow.shape[0]} rows. "
            "Use the same vision input config used for vision_bow."
        )

    checkpoint = resolve_path(cfg.get("checkpoint_path")) or vision_checkpoint_path(config)
    sae_checkpoint = SAECheckpoint.from_pretrained(checkpoint)
    sae = sae_checkpoint.get_model()
    embeddings = _prepare_vision_embeddings_for_sae(config, records, sae_checkpoint)

    from saetopic.interpretation import CorpusAdapter

    device = model_device(cfg.get("device", config.get("topics", {}).get("device", "auto")))
    adapter = CorpusAdapter(
        vocab_size=visual_bow.shape[1],
        n_features=int(getattr(sae, "n_features", 0) or 0),
        idf_weighting=bool(emission_cfg.get("idf_weighting", cfg.get("idf_weighting", True))),
        device=device,
        use_sparse_activation=(emission_cfg.get("theta_mode", cfg.get("theta_mode", "dense")) == "sparse_topk"),
        random_state=get_seed(config),
    )
    adapter.fit(
        embeddings=torch.from_numpy(embeddings),
        bow=visual_bow,
        sae=sae,
        n_epochs=int(emission_cfg.get("corpus_adapter_epochs", 30)),
        batch_size=int(emission_cfg.get("corpus_adapter_batch_size", 512)),
        num_workers=int(emission_cfg.get("num_workers", 0)),
        verbose=bool(emission_cfg.get("verbose", True)),
    )

    emission_path = out_dir / "visual_emission_probabilities.pt"
    torch.save(
        {
            "B": torch.from_numpy(adapter.feature_word_matrix_),
            "background_distribution": torch.from_numpy(adapter.background_distribution_),
            "pi": adapter.pi_,
            "vocab_type": "dinov2_kmeans_visual_words",
            "n_visual_words": visual_bow.shape[1],
        },
        emission_path,
    )
    feature_prob_path = out_dir / "visual_feature_probabilities.pt"
    torch.save({"theta_avg": torch.from_numpy(adapter.theta_avg_)}, feature_prob_path)

    theta_path = out_dir / "theta_sae_csr.npz"
    _save_sae_theta_csr(
        embeddings,
        sae,
        theta_path,
        batch_size=int(emission_cfg.get("theta_batch_size", cfg.get("activation_batch_size", 256))),
        device=device,
        theta_mode=str(emission_cfg.get("theta_mode", cfg.get("theta_mode", "dense"))),
        top_k=int(emission_cfg.get("theta_top_k", 32)),
    )

    top_words_path = out_dir / "feature_top_visual_words.csv"
    with top_words_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["feature", "rank", "visual_word", "probability"])
        writer.writeheader()
        top_n = int(emission_cfg.get("top_visual_words", 20))
        for feature_idx, row in enumerate(adapter.feature_word_matrix_):
            top_indices = np.argsort(row)[-top_n:][::-1]
            for rank, visual_word in enumerate(top_indices, start=1):
                writer.writerow(
                    {
                        "feature": feature_idx,
                        "rank": rank,
                        "visual_word": int(visual_word),
                        "probability": float(row[visual_word]),
                    }
                )

    summary = {
        "n_images": len(records),
        "n_visual_words": int(visual_bow.shape[1]),
        "checkpoint_path": str(checkpoint),
        "idf_weighting": adapter.idf_weighting,
        "outputs": {
            "visual_emission_probabilities": str(emission_path),
            "visual_feature_probabilities": str(feature_prob_path),
            "theta_sae_csr": str(theta_path),
            "feature_top_visual_words": str(top_words_path),
        },
    }
    (out_dir / "vision_emission_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    console.print(f"Wrote visual emission matrix to {emission_path}")


def _sparsify_and_renormalize_rows(matrix: np.ndarray, tau: float) -> np.ndarray:
    """Keep each row's largest probabilities up to cumulative mass ``tau``."""
    if tau <= 0 or tau >= 1:
        row_sums = matrix.sum(axis=1, keepdims=True)
        return matrix / np.clip(row_sums, 1e-12, None)

    sparse = np.zeros_like(matrix, dtype=np.float32)
    for row_idx, row in enumerate(matrix):
        order = np.argsort(row)[::-1]
        sorted_values = row[order]
        total = float(sorted_values.sum())
        if total <= 0:
            continue
        cumsum = np.cumsum(sorted_values) / total
        keep_count = int(np.searchsorted(cumsum, tau, side="left")) + 1
        keep = order[:keep_count]
        sparse[row_idx, keep] = row[keep]
    row_sums = sparse.sum(axis=1, keepdims=True)
    return sparse / np.clip(row_sums, 1e-12, None)


def run_vision_topics(config: dict[str, Any]) -> None:
    """Merge SAE visual atoms into final visual topics using B_vis and visual centroids."""
    cfg = vision_config(config)
    out_dir = vision_out_dir(config)
    emission_path = resolve_path(cfg.get("visual_emission_path")) or (
        out_dir / "visual_emission_probabilities.pt"
    )
    feature_prob_path = resolve_path(cfg.get("visual_feature_probabilities_path")) or (
        out_dir / "visual_feature_probabilities.pt"
    )
    theta_path = resolve_path(cfg.get("theta_sae_path")) or out_dir / "theta_sae_csr.npz"
    if not emission_path.exists():
        raise FileNotFoundError(f"Visual emission file not found: {emission_path}")
    if not feature_prob_path.exists():
        raise FileNotFoundError(f"Visual feature probabilities not found: {feature_prob_path}")
    if not theta_path.exists():
        raise FileNotFoundError(f"SAE theta CSR not found: {theta_path}")

    from sklearn.cluster import KMeans

    emission = torch.load(emission_path, map_location="cpu")
    feature_probabilities = torch.load(feature_prob_path, map_location="cpu")
    b_matrix = emission["B"].float().cpu().numpy()
    theta_avg = feature_probabilities["theta_avg"].float().cpu().numpy()
    theta = load_npz(theta_path).tocsr().astype(np.float32)
    centroids = _load_visual_centroids(out_dir, cfg)

    max_topic_features_cfg = cfg.get("max_topic_features", 2000)
    max_topic_features = (
        None if max_topic_features_cfg in {None, 0} else int(max_topic_features_cfg)
    )
    min_theta_avg = float(cfg.get("min_theta_avg", 0.0))
    min_emission_entropy_gap = float(cfg.get("min_emission_entropy_gap", 0.5))
    emission_entropy = -(b_matrix * np.log(np.clip(b_matrix, 1e-12, None))).sum(axis=1)
    uniform_entropy = float(np.log(b_matrix.shape[1]))
    emission_entropy_gap = uniform_entropy - emission_entropy

    valid_mask = theta_avg > 0
    if min_theta_avg > 0:
        valid_mask &= theta_avg >= min_theta_avg
    if min_emission_entropy_gap > 0:
        valid_mask &= emission_entropy_gap >= min_emission_entropy_gap
    valid_feature_idx = np.flatnonzero(valid_mask)
    if len(valid_feature_idx) == 0:
        raise ValueError(
            "No SAE features passed vision topic filtering. Lower "
            "vision.min_theta_avg or vision.min_emission_entropy_gap."
        )
    if max_topic_features is not None and len(valid_feature_idx) > max_topic_features:
        order = np.argsort(theta_avg[valid_feature_idx])[-max_topic_features:][::-1]
        valid_feature_idx = valid_feature_idx[order]

    tau = float(cfg.get("topic_embedding_sparsity", 0.9))
    b_valid = b_matrix[valid_feature_idx]
    feature_embeddings = _sparsify_and_renormalize_rows(b_valid, tau=tau) @ centroids
    norms = np.linalg.norm(feature_embeddings, axis=1, keepdims=True)
    feature_embeddings = feature_embeddings / np.clip(norms, 1e-12, None)
    weights = theta_avg[valid_feature_idx]

    n_topics_cfg = cfg.get("n_topics", [100])
    topic_counts = [int(n_topics_cfg)] if isinstance(n_topics_cfg, int) else [int(n) for n in n_topics_cfg]
    top_n = int(cfg.get("top_visual_words", 50))
    seed = get_seed(config)

    for n_topics in topic_counts:
        if n_topics > len(valid_feature_idx):
            raise ValueError(
                f"vision.n_topics={n_topics} exceeds active SAE features={len(valid_feature_idx)}"
            )
        topic_dir = out_dir / f"topics_{n_topics}"
        topic_dir.mkdir(parents=True, exist_ok=True)
        kmeans = KMeans(n_clusters=n_topics, random_state=seed, n_init=10)
        kmeans.fit(feature_embeddings, sample_weight=weights)
        labels = kmeans.labels_

        mapping = csr_matrix(
            (
                np.ones_like(labels, dtype=np.float32),
                (valid_feature_idx, labels),
            ),
            shape=(b_matrix.shape[0], n_topics),
        )
        theta_topic = theta.dot(mapping)
        save_npz(topic_dir / "theta_topic_csr.npz", theta_topic)
        cluster_counts = np.asarray((theta_topic > 0).sum(axis=0)).ravel()

        cluster_to_features: dict[int, list[int]] = {idx: [] for idx in range(n_topics)}
        for local_idx, label in enumerate(labels):
            cluster_to_features[int(label)].append(int(valid_feature_idx[local_idx]))

        records = []
        top_words_lines: list[str] = []
        for cluster_id, features in cluster_to_features.items():
            feature_weights = theta_avg[features]
            cluster_b = b_matrix[features]
            avg_probs = (cluster_b * feature_weights[:, None]).sum(axis=0)
            avg_probs = avg_probs / max(float(feature_weights.sum()), 1e-12)
            top_visual_words = np.argsort(avg_probs)[-top_n:][::-1]
            top_words_text = ", ".join(str(int(word)) for word in top_visual_words)
            top_words_lines.append(top_words_text)
            records.append(
                {
                    "cluster_id": cluster_id,
                    "cluster_size": len(features),
                    "cluster_prob": float(feature_weights.sum()),
                    "cluster_ratio": float(cluster_counts[cluster_id] / max(theta.shape[0], 1)),
                    "top_visual_words": top_words_text,
                }
            )

        records = sorted(records, key=lambda row: (-row["cluster_size"], row["cluster_id"]))
        with (topic_dir / "clusters.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "cluster_id",
                    "cluster_size",
                    "cluster_prob",
                    "cluster_ratio",
                    "top_visual_words",
                ],
            )
            writer.writeheader()
            writer.writerows(records)
        (topic_dir / "top_visual_words.txt").write_text(
            "\n".join(top_words_lines) + "\n",
            encoding="utf-8",
        )
        (topic_dir / "cluster_to_feature_indices.json").write_text(
            json.dumps({str(k): v for k, v in cluster_to_features.items()}, sort_keys=True),
            encoding="utf-8",
        )
        summary = {
            "n_topics": n_topics,
            "n_active_features": int((theta_avg > 0).sum()),
            "n_topic_features": int(len(valid_feature_idx)),
            "feature_filtering": {
                "max_topic_features": max_topic_features,
                "min_theta_avg": min_theta_avg,
                "min_emission_entropy_gap": min_emission_entropy_gap,
                "uniform_entropy": uniform_entropy,
                "selected_theta_mass": float(theta_avg[valid_feature_idx].sum()),
                "selected_entropy_gap_mean": float(emission_entropy_gap[valid_feature_idx].mean()),
                "selected_entropy_gap_min": float(emission_entropy_gap[valid_feature_idx].min()),
            },
            "topic_embedding_sparsity": tau,
            "outputs": {
                "clusters": str(topic_dir / "clusters.csv"),
                "top_visual_words": str(topic_dir / "top_visual_words.txt"),
                "cluster_to_feature_indices": str(topic_dir / "cluster_to_feature_indices.json"),
                "theta_topic_csr": str(topic_dir / "theta_topic_csr.npz"),
            },
        }
        (topic_dir / "vision_topics_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    console.print(f"Wrote visual topics to {out_dir}")


def _parse_visual_word_ids(value: Any, limit: int | None = None) -> list[int]:
    words = [
        int(part.strip())
        for part in str(value).split(",")
        if part.strip()
    ]
    return words if limit is None else words[:limit]


def _make_contact_sheet(
    cells: list[tuple[Any, str]],
    *,
    thumb_size: int,
    columns: int,
    title: str,
):
    from PIL import Image, ImageDraw

    if not cells:
        cells = [(Image.new("RGB", (thumb_size, thumb_size), "white"), "no examples")]
    label_height = 34
    title_height = 34
    rows = int(np.ceil(len(cells) / columns))
    width = columns * thumb_size
    height = title_height + rows * (thumb_size + label_height)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), title[:160], fill="black")
    for idx, (image, label) in enumerate(cells):
        col = idx % columns
        row = idx // columns
        x = col * thumb_size
        y = title_height + row * (thumb_size + label_height)
        image = image.convert("RGB")
        image.thumbnail((thumb_size, thumb_size))
        x_offset = x + (thumb_size - image.width) // 2
        y_offset = y + (thumb_size - image.height) // 2
        sheet.paste(image, (x_offset, y_offset))
        draw.rectangle((x, y, x + thumb_size - 1, y + thumb_size - 1), outline="gray")
        draw.text((x + 4, y + thumb_size + 4), label[:42], fill="black")
    return sheet


def _top_csr_column_rows(matrix: csr_matrix, column: int, limit: int) -> list[tuple[int, float]]:
    col = matrix.getcol(column).tocoo()
    if col.nnz == 0:
        return []
    order = np.argsort(col.data)[-limit:][::-1]
    return [(int(col.row[idx]), float(col.data[idx])) for idx in order]


def _write_visual_word_image_sheets(
    out_dir: Path,
    records: list[dict[str, Any]],
    labels: list[Any],
    visual_bow: csr_matrix,
    visual_words: list[int],
    *,
    examples_per_word: int,
    thumb_size: int,
    columns: int,
) -> dict[int, str]:
    visual_word_dir = out_dir / "visual_words"
    visual_word_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[int, str] = {}
    for visual_word in visual_words:
        cells = []
        for row_idx, count in _top_csr_column_rows(visual_bow, visual_word, examples_per_word):
            image = _image_from_vision_record(records[row_idx])
            label = labels[row_idx] if row_idx < len(labels) else None
            cells.append((image, f"{row_idx} {label or ''} c={count:g}"))
        sheet = _make_contact_sheet(
            cells,
            thumb_size=thumb_size,
            columns=columns,
            title=f"visual_word {visual_word} top images",
        )
        path = visual_word_dir / f"visual_word_{visual_word}.jpg"
        sheet.save(path, quality=90)
        paths[visual_word] = str(path)
    return paths


def _write_visual_word_patch_sheets(
    out_dir: Path,
    records: list[dict[str, Any]],
    labels: list[Any],
    tokenizer_cfg: dict[str, Any],
    centroids: np.ndarray,
    visual_words: list[int],
    *,
    examples_per_word: int,
    thumb_size: int,
    columns: int,
    image_size: int,
) -> dict[int, str]:
    from sklearn.metrics import pairwise_distances_argmin

    visual_word_set = set(visual_words)
    examples: dict[int, list[tuple[float, int, int]]] = {word: [] for word in visual_words}
    for start, patches in _iter_dinov2_patch_batches(records, tokenizer_cfg):
        batch_size, patches_per_image, dim = patches.shape
        flat = patches.reshape(batch_size * patches_per_image, dim)
        assignments = pairwise_distances_argmin(flat, centroids, metric="euclidean")
        for flat_idx, word in enumerate(assignments):
            word = int(word)
            if word not in visual_word_set:
                continue
            patch_vec = flat[flat_idx]
            distance = float(np.linalg.norm(patch_vec - centroids[word]))
            local_idx = flat_idx // patches_per_image
            patch_idx = flat_idx % patches_per_image
            bucket = examples[word]
            bucket.append((distance, start + local_idx, patch_idx))
            bucket.sort(key=lambda item: item[0])
            del bucket[examples_per_word:]

    patch_dir = out_dir / "visual_word_patches"
    patch_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[int, str] = {}
    for word, bucket in examples.items():
        cells = []
        for distance, row_idx, patch_idx in bucket:
            image = _image_from_vision_record(records[row_idx]).resize((image_size, image_size))
            grid = int(round(np.sqrt(max(1, len(bucket)))))
            # The DINO output patch count is not stored per bucket, infer from current tokenizer default.
            # Recompute a square grid from image_size/14 when possible; DINOv2 ViT-B/14 at 224 is 16x16.
            grid = int(tokenizer_cfg.get("patch_grid") or max(1, image_size // 14))
            patch_size = image_size // grid
            x = (patch_idx % grid) * patch_size
            y = (patch_idx // grid) * patch_size
            patch = image.crop((x, y, min(x + patch_size, image_size), min(y + patch_size, image_size)))
            label = labels[row_idx] if row_idx < len(labels) else None
            cells.append((patch, f"{row_idx} {label or ''} d={distance:.2f}"))
        sheet = _make_contact_sheet(
            cells,
            thumb_size=thumb_size,
            columns=columns,
            title=f"visual_word {word} top patches",
        )
        path = patch_dir / f"visual_word_{word}_patches.jpg"
        sheet.save(path, quality=90)
        paths[word] = str(path)
    return paths


def run_vision_visualize(config: dict[str, Any]) -> None:
    """Create human-readable visual topic/contact-sheet artifacts."""
    cfg = vision_config(config)
    vis_cfg = dict(cfg.get("visualize") or {})
    out_dir = vision_out_dir(config)
    records = load_vision_probe_inputs(cfg)
    if not records:
        raise ValueError("vision_visualize needs inputs. Set vision.hf_dataset, inputs, or input_file.")

    visual_bow_path = resolve_path(cfg.get("visual_bow_path")) or out_dir / "visual_bow.npz"
    visual_bow_meta_path = resolve_path(cfg.get("visual_bow_meta_path")) or (
        out_dir / "visual_bow_meta.json"
    )
    if not visual_bow_path.exists():
        raise FileNotFoundError(f"Visual BoW not found: {visual_bow_path}")
    if not visual_bow_meta_path.exists():
        raise FileNotFoundError(f"Visual BoW metadata not found: {visual_bow_meta_path}")
    visual_bow = load_npz(visual_bow_path).tocsr()
    visual_bow_meta = json.loads(visual_bow_meta_path.read_text(encoding="utf-8"))
    labels = visual_bow_meta.get("labels") or [record.get("label") for record in records]

    default_topic_counts = cfg.get("n_topics") or [50]
    default_n_topics = (
        int(default_topic_counts[0])
        if isinstance(default_topic_counts, list)
        else int(default_topic_counts)
    )
    n_topics = int(vis_cfg.get("n_topics", default_n_topics))
    topic_dir = out_dir / f"topics_{n_topics}"
    clusters_path = topic_dir / "clusters.csv"
    theta_topic_path = topic_dir / "theta_topic_csr.npz"
    if not clusters_path.exists():
        raise FileNotFoundError(f"Topic clusters not found: {clusters_path}")
    if not theta_topic_path.exists():
        raise FileNotFoundError(f"Topic theta matrix not found: {theta_topic_path}")
    clusters = list(csv.DictReader(clusters_path.open("r", encoding="utf-8")))
    theta_topic = load_npz(theta_topic_path).tocsr()

    top_topics = int(vis_cfg.get("top_topics", 20))
    top_visual_words_per_topic = int(vis_cfg.get("top_visual_words_per_topic", 8))
    visual_word_examples = int(vis_cfg.get("visual_word_examples", 8))
    topic_image_examples = int(vis_cfg.get("topic_image_examples", 16))
    thumb_size = int(vis_cfg.get("thumb_size", 160))
    columns = int(vis_cfg.get("columns", 4))
    image_size = int(vis_cfg.get("image_size", 224))
    patch_representatives = bool(vis_cfg.get("patch_representatives", False))

    selected_clusters = clusters[:top_topics]
    visual_words: list[int] = []
    for row in selected_clusters:
        for visual_word in _parse_visual_word_ids(
            row.get("top_visual_words", ""),
            limit=top_visual_words_per_topic,
        ):
            if visual_word not in visual_words:
                visual_words.append(visual_word)

    viz_dir = out_dir / "visualizations" / f"topics_{n_topics}"
    viz_dir.mkdir(parents=True, exist_ok=True)
    visual_word_paths = _write_visual_word_image_sheets(
        viz_dir,
        records,
        labels,
        visual_bow,
        visual_words,
        examples_per_word=visual_word_examples,
        thumb_size=thumb_size,
        columns=columns,
    )

    patch_paths: dict[int, str] = {}
    if patch_representatives:
        tokenizer_cfg = dict(cfg.get("visual_tokenizer") or {})
        centroids = _load_visual_centroids(out_dir, cfg)
        patch_paths = _write_visual_word_patch_sheets(
            viz_dir,
            records,
            labels,
            tokenizer_cfg,
            centroids,
            visual_words,
            examples_per_word=visual_word_examples,
            thumb_size=thumb_size,
            columns=columns,
            image_size=image_size,
        )

    topic_rows = []
    for row in selected_clusters:
        cluster_id = int(row["cluster_id"])
        cells = []
        for row_idx, weight in _top_csr_column_rows(theta_topic, cluster_id, topic_image_examples):
            image = _image_from_vision_record(records[row_idx])
            label = labels[row_idx] if row_idx < len(labels) else None
            cells.append((image, f"{row_idx} {label or ''} w={weight:.3f}"))
        title = (
            f"topic {cluster_id} size={row.get('cluster_size')} "
            f"ratio={float(row.get('cluster_ratio', 0.0)):.3f}"
        )
        sheet = _make_contact_sheet(cells, thumb_size=thumb_size, columns=columns, title=title)
        topic_path = viz_dir / f"topic_{cluster_id}_images.jpg"
        sheet.save(topic_path, quality=90)
        topic_rows.append((row, topic_path))

    html_lines = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>Vision Topics</title>",
        "<style>body{font-family:sans-serif} img{max-width:640px;border:1px solid #ccc} "
        "table{border-collapse:collapse} td,th{border:1px solid #ccc;padding:6px;vertical-align:top}</style>",
        "</head><body>",
        f"<h1>Vision Topics {n_topics}</h1>",
        f"<p>Top topics shown: {len(topic_rows)}. Patch representatives: {patch_representatives}.</p>",
        "<table><tr><th>Topic</th><th>Stats</th><th>Top Visual Words</th><th>Images</th></tr>",
    ]
    for row, topic_path in topic_rows:
        words = _parse_visual_word_ids(row.get("top_visual_words", ""), top_visual_words_per_topic)
        word_links = []
        for word in words:
            image_path = Path(visual_word_paths[word]).relative_to(viz_dir)
            link = f"<a href='{html.escape(str(image_path))}'>vw {word}</a>"
            if word in patch_paths:
                patch_path = Path(patch_paths[word]).relative_to(viz_dir)
                link += f" (<a href='{html.escape(str(patch_path))}'>patches</a>)"
            word_links.append(link)
        topic_rel = topic_path.relative_to(viz_dir)
        html_lines.extend(
            [
                "<tr>",
                f"<td>{html.escape(str(row['cluster_id']))}</td>",
                "<td>"
                f"size={html.escape(str(row.get('cluster_size')))}<br>"
                f"prob={float(row.get('cluster_prob', 0.0)):.4f}<br>"
                f"ratio={float(row.get('cluster_ratio', 0.0)):.4f}"
                "</td>",
                f"<td>{'<br>'.join(word_links)}</td>",
                f"<td><a href='{html.escape(str(topic_rel))}'><img src='{html.escape(str(topic_rel))}'></a></td>",
                "</tr>",
            ]
        )
    html_lines.extend(["</table>", "</body></html>"])
    index_path = viz_dir / "index.html"
    index_path.write_text("\n".join(html_lines), encoding="utf-8")

    summary = {
        "n_topics": n_topics,
        "top_topics": len(topic_rows),
        "top_visual_words": len(visual_words),
        "patch_representatives": patch_representatives,
        "outputs": {
            "index": str(index_path),
            "visual_word_image_dir": str(viz_dir / "visual_words"),
            "visual_word_patch_dir": str(viz_dir / "visual_word_patches"),
        },
    }
    (viz_dir / "vision_visualize_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    console.print(f"Wrote visual topic index to {index_path}")


def run_vision_probe(config: dict[str, Any]) -> None:
    """Probe how image embeddings activate a trained text/omni SAE."""
    probe_cfg = config.get("vision_probe", {})
    records = load_vision_probe_inputs(probe_cfg)
    if not records:
        raise ValueError(
            "vision_probe needs inputs. Set vision_probe.hf_dataset, "
            "vision_probe.inputs, or vision_probe.input_file."
        )

    out_dir = resolve_path(probe_cfg.get("out_dir", "results/vision_probe"))
    if out_dir is None:
        raise ValueError("vision_probe.out_dir is required")
    out_dir.mkdir(parents=True, exist_ok=True)

    embedder = build_vision_probe_embedder(config)
    payloads = [_vision_probe_payload(record) for record in records]
    embeddings = _encode_probe_payloads(
        embedder,
        payloads,
        encode_method=str(probe_cfg.get("encode_method", "document")),
        batch_size=int(
            probe_cfg.get("batch_size", config["embedding_model"].get("batch_size", 16))
        ),
    )
    checkpoint = resolve_path(probe_cfg.get("checkpoint_path")) or checkpoint_path(config)
    sae_checkpoint = SAECheckpoint.from_pretrained(checkpoint)
    sae = sae_checkpoint.get_model()
    if embeddings.shape[1] > sae_checkpoint.embedding_dim:
        original_dim = embeddings.shape[1]
        truncate_dim = int(probe_cfg.get("truncate_dim", sae_checkpoint.embedding_dim))
        if truncate_dim == sae_checkpoint.embedding_dim:
            embeddings = embeddings[:, : sae_checkpoint.embedding_dim]
            console.print(
                "Truncated vision embeddings "
                f"from {original_dim} to SAE input_dim={sae_checkpoint.embedding_dim}"
            )
    if bool(probe_cfg.get("normalize", True)):
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.where(norms == 0, 1.0, norms)
    if embeddings.shape[1] != sae_checkpoint.embedding_dim:
        raise ValueError(
            f"Vision embeddings have dimension {embeddings.shape[1]}, but SAE expects "
            f"{sae_checkpoint.embedding_dim}. Set embedding_model.truncate_dim or "
            "vision_probe.truncate_dim to match the SAE input_dim."
        )

    device = model_device(probe_cfg.get("device", config.get("topics", {}).get("device", "auto")))
    dev = torch.device(device)
    sae = sae.to(dev).eval()
    sae_dtype = torch.float32 if dev.type == "cpu" else torch.bfloat16
    top_k = int(probe_cfg.get("top_k", 10))
    activation_mode = str(probe_cfg.get("activation_mode", "dense"))
    if activation_mode not in {"dense", "sparse"}:
        raise ValueError("vision_probe.activation_mode must be 'dense' or 'sparse'")

    sample_results: list[dict[str, Any]] = []
    feature_stats: dict[int, dict[str, float]] = {}
    recon_mse_values: list[float] = []
    recon_cosine_values: list[float] = []
    n_features = int(getattr(sae, "n_features", 0) or 0)

    activation_batch_size = int(probe_cfg.get("activation_batch_size", 128))
    with torch.no_grad():
        for start in range(0, len(embeddings), activation_batch_size):
            batch_np = embeddings[start : start + activation_batch_size]
            batch = torch.from_numpy(batch_np).to(dev, dtype=sae_dtype)
            x_recon, h, f, _ = sae(batch)
            feature_values = h.float() if activation_mode == "dense" else f.float()
            n_features = max(n_features, int(feature_values.shape[1]))
            k = min(top_k, feature_values.shape[1])
            top_values, top_indices = torch.topk(feature_values, k=k, dim=1, sorted=True)
            mse = (batch.float() - x_recon.float()).pow(2).mean(dim=1)
            cosine = torch.nn.functional.cosine_similarity(batch.float(), x_recon.float(), dim=1)

            for local_idx in range(len(batch_np)):
                record_idx = start + local_idx
                indices = [int(value) for value in top_indices[local_idx].cpu().tolist()]
                values = [float(value) for value in top_values[local_idx].cpu().tolist()]
                top_features = [
                    {"feature": feature, "activation": activation}
                    for feature, activation in zip(indices, values, strict=True)
                    if activation > 0.0
                ]
                for rank, (feature, activation) in enumerate(
                    (
                        (item["feature"], item["activation"])
                        for item in top_features
                    ),
                    start=1,
                ):
                    stats = feature_stats.setdefault(
                        feature,
                        {"count": 0.0, "activation_sum": 0.0, "best_rank": float(rank)},
                    )
                    stats["count"] += 1.0
                    stats["activation_sum"] += activation
                    stats["best_rank"] = min(stats["best_rank"], float(rank))

                recon_mse = float(mse[local_idx].cpu())
                recon_cosine = float(cosine[local_idx].cpu())
                recon_mse_values.append(recon_mse)
                recon_cosine_values.append(recon_cosine)

                source = records[record_idx]
                sample_results.append(
                    {
                        "id": source.get("id", str(record_idx)),
                        "image": source.get("image") or source.get("path") or source.get("url"),
                        "label": source.get("label"),
                        "text": source.get("text") or source.get("caption"),
                        "reconstruction_mse": recon_mse,
                        "reconstruction_cosine": recon_cosine,
                        "top_features": top_features,
                    }
                )

    with (out_dir / "samples.jsonl").open("w", encoding="utf-8") as f:
        for row in sample_results:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    with (out_dir / "feature_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["feature", "count", "mean_activation", "best_rank"],
        )
        writer.writeheader()
        for feature, stats in sorted(
            feature_stats.items(),
            key=lambda item: (-item[1]["count"], -item[1]["activation_sum"], item[0]),
        ):
            writer.writerow(
                {
                    "feature": feature,
                    "count": int(stats["count"]),
                    "mean_activation": stats["activation_sum"] / max(stats["count"], 1.0),
                    "best_rank": int(stats["best_rank"]),
                }
            )

    visual_bow_outputs = _write_vision_probe_visual_bow(out_dir, sample_results, n_features)

    summary = {
        "n_images": len(records),
        "embedding_dim": int(embeddings.shape[1]),
        "checkpoint_path": str(checkpoint),
        "activation_mode": activation_mode,
        "top_k": top_k,
        "mean_reconstruction_mse": float(np.mean(recon_mse_values)) if recon_mse_values else 0.0,
        "mean_reconstruction_cosine": (
            float(np.mean(recon_cosine_values)) if recon_cosine_values else 0.0
        ),
        "outputs": {
            "samples": str(out_dir / "samples.jsonl"),
            "feature_summary": str(out_dir / "feature_summary.csv"),
            **visual_bow_outputs,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    console.print(f"Wrote vision probe outputs to {out_dir}")


def load_news20k(
    dataset_cfg: dict[str, Any],
    n_docs: int | None,
    seed: int,
) -> tuple[list[str], np.ndarray, list[str] | None]:
    source = dataset_cfg.get("source", "hf")
    categories = dataset_cfg.get("categories")
    remove_metadata = bool(dataset_cfg.get("remove_metadata", True))

    if source == "hf":
        docs, labels, target_names = load_news20k_from_hf(
            dataset_name=dataset_cfg.get("hf_dataset", "SetFit/20_newsgroups"),
            split=dataset_cfg.get("hf_split", "train+test"),
            categories=categories,
            remove_metadata=remove_metadata,
        )
    elif source == "sklearn":
        docs, labels, target_names = load_news20k_from_sklearn(
            seed=seed,
            categories=categories,
            data_home=dataset_cfg.get("sklearn_data_home"),
            download_if_missing=bool(dataset_cfg.get("download_sklearn", False)),
            remove_metadata=remove_metadata,
        )
    else:
        raise ValueError(f"Unknown news20k source: {source}")

    if n_docs is not None and n_docs < len(docs):
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(docs), size=n_docs, replace=False)
        docs = [docs[idx] for idx in indices]
        labels = labels[indices]
    return docs, labels, target_names


def load_news20k_from_sklearn(
    seed: int,
    categories: list[str] | None,
    data_home: str | None,
    download_if_missing: bool,
    remove_metadata: bool,
) -> tuple[list[str], np.ndarray, list[str]]:
    dataset = fetch_20newsgroups(
        subset="all",
        categories=categories,
        data_home=data_home,
        remove=("headers", "footers", "quotes") if remove_metadata else (),
        shuffle=True,
        random_state=seed,
        download_if_missing=download_if_missing,
    )
    docs = [doc.strip() for doc in dataset.data]
    labels = np.asarray(dataset.target)
    non_empty = np.asarray([bool(doc) for doc in docs])
    return [doc for doc, keep in zip(docs, non_empty) if keep], labels[non_empty], list(dataset.target_names)


def load_news20k_from_hf(
    dataset_name: str,
    split: str,
    categories: list[str] | None,
    remove_metadata: bool,
) -> tuple[list[str], np.ndarray, list[str] | None]:
    from datasets import ClassLabel, concatenate_datasets, load_dataset

    if "+" in split:
        parts = [part.strip() for part in split.split("+") if part.strip()]
        dataset = concatenate_datasets([load_dataset(dataset_name, split=part) for part in parts])
    else:
        dataset = load_dataset(dataset_name, split=split)

    text_column = pick_column(dataset, ("text", "data", "content", "document"))
    label_column = pick_column(dataset, ("label", "target", "class"))
    label_name_column = pick_column(
        dataset,
        ("label_text", "label_name", "target_name", "category"),
        required=False,
    )

    label_feature = dataset.features.get(label_column)
    if isinstance(label_feature, ClassLabel):
        target_names = list(label_feature.names)
        label_to_id = {name: idx for idx, name in enumerate(target_names)}
    elif label_name_column is not None:
        target_names = sorted({str(value) for value in dataset[label_name_column]})
        label_to_id = {name: idx for idx, name in enumerate(target_names)}
    else:
        labels_seen = sorted({int(value) for value in dataset[label_column]})
        target_names = [str(label) for label in labels_seen]
        label_to_id = {str(label): idx for idx, label in enumerate(labels_seen)}

    docs: list[str] = []
    labels: list[int] = []
    category_filter = set(categories) if categories else None
    for row in dataset:
        text = str(row[text_column]).strip()
        if remove_metadata:
            text = strip_20newsgroups_metadata(text)
        if not text:
            continue

        label_value = row[label_column]
        if isinstance(label_feature, ClassLabel):
            label_id = int(label_value)
            label_name = target_names[label_id]
        elif label_name_column is not None:
            label_name = str(row[label_name_column])
            label_id = label_to_id[label_name]
        else:
            label_name = str(label_value)
            label_id = label_to_id[label_name]

        if category_filter is not None and label_name not in category_filter:
            continue
        docs.append(text)
        labels.append(label_id)

    return docs, np.asarray(labels), target_names


def strip_20newsgroups_metadata(text: str) -> str:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    for idx, line in enumerate(lines):
        if not line.strip():
            lines = lines[idx + 1 :]
            break

    cleaned = []
    for line in lines:
        stripped = line.strip()
        if stripped == "--":
            break
        if stripped.startswith((">", "|")):
            continue
        if re.match(r"^(writes|wrote|in article|article|from|subject):", stripped, re.I):
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def pick_column(dataset: Any, preferred: tuple[str, ...], required: bool = True) -> str | None:
    for column in preferred:
        if column in dataset.column_names:
            return column
    if required and dataset.column_names:
        return dataset.column_names[0]
    return None


def load_hf_text_dataset(
    dataset_cfg: dict[str, Any],
    n_docs: int | None,
    seed: int,
) -> tuple[list[str], np.ndarray | None, list[str] | None]:
    from datasets import ClassLabel, concatenate_datasets, load_dataset

    dataset_name = dataset_cfg["hf_dataset"]
    subset = dataset_cfg.get("hf_subset")
    split = dataset_cfg.get("hf_split", dataset_cfg.get("split", "train"))
    text_column = dataset_cfg.get("text_column", "text")
    label_column = dataset_cfg.get("label_column")

    if "+" in split:
        parts = [part.strip() for part in split.split("+") if part.strip()]
        datasets = [
            load_dataset(dataset_name, subset, split=part)
            if subset
            else load_dataset(dataset_name, split=part)
            for part in parts
        ]
        dataset = concatenate_datasets(datasets)
    else:
        dataset = (
            load_dataset(dataset_name, subset, split=split)
            if subset
            else load_dataset(dataset_name, split=split)
        )

    if n_docs is not None and n_docs < len(dataset):
        rng = np.random.default_rng(seed)
        dataset = dataset.select(rng.choice(len(dataset), size=n_docs, replace=False))

    docs: list[str] = []
    labels: list[int] = []
    for row in dataset:
        text = str(row[text_column]).strip()
        if not text:
            continue
        docs.append(text)
        if label_column:
            labels.append(int(row[label_column]))

    target_names = None
    if label_column:
        label_feature = dataset.features.get(label_column)
        if isinstance(label_feature, ClassLabel):
            target_names = list(label_feature.names)
        else:
            target_names = [str(label) for label in sorted(set(labels))]

    return docs, np.asarray(labels) if labels else None, target_names


def load_topic_dataset(
    dataset_cfg: dict[str, Any],
    n_docs: int | None,
    seed: int,
) -> tuple[list[str], np.ndarray | None, list[str] | None]:
    key = dataset_cfg["key"]
    if key == "news20k":
        return load_news20k(dataset_cfg, n_docs=n_docs, seed=seed)
    return load_hf_text_dataset(dataset_cfg, n_docs=n_docs, seed=seed)


def build_topic_model(config: dict[str, Any], n_topics: int) -> SAETopicModel:
    topic_cfg = config["topics"]
    vocabulary_size = topic_cfg.get("vocabulary_size")
    vocabulary_size = None if vocabulary_size in {None, 0} else int(vocabulary_size)
    merge_embedding_model = topic_cfg.get("merge_embedding_model")
    if isinstance(merge_embedding_model, str) and merge_embedding_model.lower() == "none":
        merge_embedding_model = None

    return SAETopicModel.from_pretrained(
        checkpoint_path(config),
        n_topics=n_topics,
        merge_embedding_model=merge_embedding_model,
        corpus_adapter_epochs=int(topic_cfg.get("corpus_adapter_epochs", 30)),
        corpus_adapter_batch_size=int(topic_cfg.get("corpus_adapter_batch_size", 512)),
        activation_batch_size=int(topic_cfg.get("activation_batch_size", 256)),
        embedding_batch_size=int(topic_cfg.get("embedding_batch_size", 64)),
        idf_weighting=bool(topic_cfg.get("idf_weighting", True)),
        vocabulary_size=vocabulary_size,
        min_df=int(topic_cfg.get("min_df", 5)),
        max_df=float(topic_cfg.get("max_df", 1.0)),
        stop_words=(None if str(topic_cfg.get("stop_words", "saetm")).lower() == "none" else topic_cfg.get("stop_words", "saetm")),
        theta_mode=topic_cfg.get("theta_mode", "dense"),
        max_seq_length=int(topic_cfg.get("max_seq_length", 512)),
        use_ctfidf=bool(topic_cfg.get("use_ctfidf", False)),
        drop_empty_topics=False,
        random_state=get_seed(config),
        device=topic_cfg.get("device", "auto"),
    )


def save_topic_outputs(
    model: SAETopicModel,
    docs: list[str],
    labels: np.ndarray | None,
    output_dir: Path,
    elapsed: float,
    save_theta_topic: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    topic_words = model.get_topics(top_n=20)
    artifact_topic_words = model.get_topics(top_n=50)

    cluster_info = model.get_cluster_info()
    if not cluster_info.empty:
        cluster_info = cluster_info.sort_values(
            ["cluster_size", "cluster_id"],
            ascending=[False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        ordered_topic_ids = [int(topic_id) for topic_id in cluster_info["cluster_id"]]
    else:
        ordered_topic_ids = sorted(artifact_topic_words)

    info = model.get_topic_info()
    info["Top_Words_20"] = [
        ", ".join(word for word, _ in topic_words[topic_id])
        for topic_id in info["Topic"]
    ]
    if ordered_topic_ids:
        info_by_topic = info.set_index("Topic", drop=False)
        available_topic_ids = [
            topic_id
            for topic_id in ordered_topic_ids
            if topic_id in info_by_topic.index
        ]
        if available_topic_ids:
            info = info_by_topic.loc[available_topic_ids].reset_index(drop=True)
    info.to_csv(output_dir / "topic_info.csv", index=False)
    with (output_dir / "top_words.txt").open("w", encoding="utf-8") as f:
        for topic_id in ordered_topic_ids:
            words = [word for word, _ in artifact_topic_words.get(topic_id, [])[:50]]
            if words:
                f.write(", ".join(words) + "\n")

    cluster_info.to_csv(output_dir / "clusters.csv", index=False)
    (output_dir / "cluster_to_feature_indices.json").write_text(
        json.dumps(model.get_cluster_to_feature_indices(), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    if save_theta_topic:
        save_npz(output_dir / "theta_topic_csr.npz", model.get_theta_topic_matrix(normalize=False, sparse=True))

    summary: dict[str, Any] = {
        "n_docs": len(docs),
        "n_topics": model.n_topics,
        "fit_or_retopic_seconds": elapsed,
        "vocab_size": len(model.vocab_ or []),
        "embedding_shape": list(model.embeddings_.shape) if model.embeddings_ is not None else None,
        "activation_shape": (
            list(model.feature_activations_.shape)
            if model.feature_activations_ is not None
            else None
        ),
        "theta_avg_shape": list(model.theta_avg_.shape) if model.theta_avg_ is not None else None,
        "merge_embedding_model": model.merge_embedding_model,
        "wrote_theta_topic_csr": save_theta_topic,
    }
    if labels is not None and model.topics_ is not None and len(labels) == len(model.topics_):
        summary["ARI"] = adjusted_rand_score(labels, model.topics_)
        summary["NMI"] = normalized_mutual_info_score(labels, model.topics_)

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def run_topics(config: dict[str, Any]) -> None:
    topic_cfg = config["topics"]
    topic_counts = list(dict.fromkeys(int(value) for value in topic_cfg.get("n_topics", [50, 100, 200, 300, 500])))
    n_docs_cfg = topic_cfg.get("n_docs", 0)
    default_n_docs = None if n_docs_cfg in {None, 0} else int(n_docs_cfg)
    out_dir = resolve_path(topic_cfg.get("out_dir", "results/text_topics"))
    if out_dir is None:
        raise ValueError("topics.out_dir is required")

    merge_embedding_model = topic_cfg.get("merge_embedding_model")
    if merge_embedding_model and str(merge_embedding_model).lower() != "none":
        console.print(f"Preloading merge word embedding model once: {merge_embedding_model}")
        preload_word_embedding_model(str(merge_embedding_model))

    model = build_topic_model(config, topic_counts[0])
    console.print(
        "Loaded SAE-TM once for all topic datasets "
        f"(input_dim={model.sae_input_dim_}, n_features={model.sae_n_features_})"
    )

    for dataset_cfg in topic_cfg.get("datasets", []):
        dataset_key = dataset_cfg["key"]
        n_docs = dataset_cfg.get("n_docs", default_n_docs)
        n_docs = None if n_docs in {None, 0} else int(n_docs)
        console.print(f"\n[bold]=== Topic dataset: {dataset_key} ===[/]")
        docs, labels, _ = load_topic_dataset(dataset_cfg, n_docs=n_docs, seed=get_seed(config))
        console.print(f"  docs={len(docs):,} | labels={'yes' if labels is not None else 'no'}")

        t0 = time.time()
        topics, probs = model.fit_transform(docs, n_topics=topic_counts[0])
        del topics, probs
        output_dir = out_dir / dataset_key / f"topics_{model.n_topics}"
        save_topic_outputs(
            model,
            docs,
            labels,
            output_dir,
            elapsed=time.time() - t0,
            save_theta_topic=bool(topic_cfg.get("save_theta_topic", False)),
        )
        console.print(f"  wrote {output_dir}")

        for n_topics in topic_counts[1:]:
            t0 = time.time()
            model.retopic(n_topics)
            output_dir = out_dir / dataset_key / f"topics_{n_topics}"
            save_topic_outputs(
                model,
                docs,
                labels,
                output_dir,
                elapsed=time.time() - t0,
                save_theta_topic=bool(topic_cfg.get("save_theta_topic", False)),
            )
            console.print(f"  wrote {output_dir}")

        gc.collect()


def build_vllm_callable(eval_cfg: dict[str, Any]):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(eval_cfg.get("llm_model", "microsoft/phi-4"))
    llm_kwargs = {
        "model": eval_cfg.get("llm_model", "microsoft/phi-4"),
        "trust_remote_code": True,
        "dtype": "auto",
        "tensor_parallel_size": int(eval_cfg.get("tensor_parallel_size", 1)),
        "max_model_len": int(eval_cfg.get("max_model_len", 4096)),
        "gpu_memory_utilization": float(eval_cfg.get("gpu_memory_utilization", 0.9)),
        "enforce_eager": bool(eval_cfg.get("enforce_eager", False)),
    }
    if eval_cfg.get("max_num_seqs") is not None:
        llm_kwargs["max_num_seqs"] = int(eval_cfg["max_num_seqs"])
    if eval_cfg.get("max_num_batched_tokens") is not None:
        llm_kwargs["max_num_batched_tokens"] = int(eval_cfg["max_num_batched_tokens"])
    llm = LLM(**llm_kwargs)

    class VLLMCallable:
        def __call__(self, prompt: str) -> str:
            return self.batch_intruder([prompt])[0]

        def _generate(self, prompts: list[str], *, max_tokens: int, temperature: float, top_p: float = 1.0) -> list[str]:
            formatted = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for prompt in prompts
            ]
            params = SamplingParams(max_tokens=max_tokens, temperature=temperature, top_p=top_p)
            try:
                outputs = llm.generate(formatted, params, use_tqdm=False)
            except TypeError:
                outputs = llm.generate(formatted, params)
            return [output.outputs[0].text for output in outputs]

        def batch_coherence(self, prompts: list[str]) -> list[str]:
            return self._generate(prompts, max_tokens=512, temperature=0.7, top_p=0.9)

        def batch_intruder(self, prompts: list[str]) -> list[str]:
            return self._generate(prompts, max_tokens=10, temperature=0.0)

    return VLLMCallable()


def n_coherence_prompts(topic_words: dict[int, list[str]], k: int, repetitions: int) -> int:
    return sum(1 for words in topic_words.values() if len(words) >= k) * repetitions


def n_intruder_prompts(topic_words: dict[int, list[str]], k: int, n: int, repetitions: int) -> int:
    valid_topics = [topic_id for topic_id, words in topic_words.items() if len(words) >= k]
    if len(topic_words) < 2:
        return 0
    return sum(
        1
        for topic_id in valid_topics
        if k >= n and any(other != topic_id and topic_words[other] for other in topic_words)
    ) * repetitions


def progress_batch_callable(llm_batch, progress: Progress, task_id: TaskID):
    if llm_batch is None:
        return None

    def call(prompts: list[str]) -> list[str]:
        responses = list(llm_batch(prompts))
        progress.update(task_id, advance=len(prompts))
        return responses

    return call


def run_evaluate(config: dict[str, Any]) -> None:
    eval_cfg = config["evaluation"]
    paths = eval_cfg.get("paths")
    if paths is None:
        paths = [config["topics"].get("out_dir", "results/text_topics")]
    top_words_files = iter_top_words_files(paths)
    if not top_words_files:
        raise FileNotFoundError(f"No top_words.txt files found from: {paths}")

    word_embeddings = None
    mean_embedding = None
    word_embeddings_dir = resolve_path(eval_cfg.get("word_embeddings_dir"))
    if word_embeddings_dir is not None and word_embeddings_dir.exists():
        word_embeddings, mean_embedding = load_saetm_word2vec_cache(word_embeddings_dir)
    elif eval_cfg.get("embedding_model") is None:
        raise FileNotFoundError(
            "SAE-TM word2vec cache not found. Set evaluation.word_embeddings_dir "
            "or evaluation.embedding_model."
        )

    llm = None
    if eval_cfg.get("llm_backend", "none") == "vllm":
        llm = build_vllm_callable(eval_cfg)

    out_path = resolve_path(eval_cfg.get("out", "results/text_topics/saetm_eval_reference.jsonl"))
    if out_path is None:
        raise ValueError("evaluation.out is required")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("elapsed •"),
        TimeRemainingColumn(),
        TextColumn("remaining"),
    )

    with out_path.open("w", encoding="utf-8") as f, progress:
        file_task = progress.add_task("[cyan]Evaluating topic files", total=len(top_words_files))
        for path in top_words_files:
            progress.update(file_task, description=f"[cyan]Evaluating {path.parent}")
            topic_words = load_top_words_file(path, top_n=int(eval_cfg.get("top_n", 20)))
            result: dict[str, Any] = {
                "path": str(path),
                "metrics": {
                    "D": compute_wmd_diversity(
                        topic_words,
                        top_n=int(eval_cfg.get("top_n", 20)),
                        word_embeddings=word_embeddings,
                        mean_embedding=mean_embedding,
                        embedding_model=eval_cfg.get("embedding_model"),
                    )
                },
            }

            if llm is not None:
                k = int(eval_cfg.get("k", 5))
                n = int(eval_cfg.get("n", 4))
                r = int(eval_cfg.get("r", 3))
                llm_batch_size = int(eval_cfg.get("llm_batch_size", 32))
                seed = eval_cfg.get("seed")

                cr_total = n_coherence_prompts(topic_words, k, r)
                cr_task = progress.add_task("[magenta]CR judge prompts", total=cr_total)
                cr = compute_coherence_rating(
                    topic_words,
                    llm=llm,
                    llm_batch=progress_batch_callable(getattr(llm, "batch_coherence", None), progress, cr_task),
                    llm_batch_size=llm_batch_size,
                    top_n=k,
                    sample_size=k,
                    repetitions=r,
                    seed=seed,
                )
                progress.update(cr_task, completed=cr_total)
                progress.remove_task(cr_task)

                ci_total = n_intruder_prompts(topic_words, k, n, r)
                ci_task = progress.add_task("[magenta]CI judge prompts", total=ci_total)
                ci = compute_intruder_detection(
                    topic_words,
                    llm=llm,
                    llm_batch=progress_batch_callable(getattr(llm, "batch_intruder", None), progress, ci_task),
                    llm_batch_size=llm_batch_size,
                    top_n=k,
                    sample_size=n,
                    repetitions=r,
                    seed=seed,
                )
                progress.update(ci_task, completed=ci_total)
                progress.remove_task(ci_task)

                ci = {topic_id: score * 100.0 for topic_id, score in ci.items()}
                result["metrics"]["CR"] = summarize_metric(cr)
                result["metrics"]["CI"] = summarize_metric(ci)
                result["CR_by_topic"] = cr
                result["CI_by_topic"] = ci

            f.write(json.dumps(result, sort_keys=True) + "\n")
            progress.update(file_task, advance=1)

    console.print(f"Wrote {out_path}")


def run_stage(stage: str, config: dict[str, Any]) -> None:
    if stage == "chunks":
        n_chunks, chunks_path = run_chunks(config)
        console.print(f"Saved {n_chunks:,} text chunks to {chunks_path}")
    elif stage == "embeddings":
        n_embeddings, embedding_dim = run_embeddings(config)
        console.print(f"Saved {n_embeddings:,} embeddings of dimension {embedding_dim}")
    elif stage == "train_sae":
        run_train_sae(config)
    elif stage == "topics":
        run_topics(config)
    elif stage == "evaluate":
        run_evaluate(config)
    elif stage == "vision_vocab":
        run_vision_vocab(config)
    elif stage == "vision_bow":
        run_vision_bow(config)
    elif stage == "vision_emission":
        run_vision_emission(config)
    elif stage == "vision_topics":
        run_vision_topics(config)
    elif stage == "vision_visualize":
        run_vision_visualize(config)
    elif stage == "vision_probe":
        run_vision_probe(config)
    else:
        raise ValueError(f"Unknown pretrain stage: {stage}")


def main(default_stages: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="pretrain/params.yaml.example")
    parser.add_argument(
        "--stages",
        nargs="+",
        default=None,
        help=(
            "Stages to run: chunks embeddings train_sae topics evaluate "
            "vision_vocab vision_bow vision_emission vision_topics vision_visualize vision_probe. "
            "Defaults to pipeline.stages."
        ),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    stages = args.stages or default_stages or config.get("pipeline", {}).get("stages")
    if not stages:
        raise ValueError("No stages provided. Set pipeline.stages or pass --stages.")

    for stage in stages:
        console.rule(f"pretrain: {stage}")
        run_stage(stage, config)


if __name__ == "__main__":
    main()
