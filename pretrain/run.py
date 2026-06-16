#!/usr/bin/env python3
"""Run the SAETopic pretraining workflow from a YAML config.

This script is intentionally independent from ``examples/``. It only imports
library code from ``src/saetopic`` and keeps pretraining-specific orchestration
under ``pretrain/``.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
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
from scipy.sparse import save_npz
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
    write_top_words_file,
)
from saetopic.merging import preload_word_embedding_model
from saetopic.training import compute_and_save_embeddings, create_streaming_dataset, train_sae
from saetopic.training.data import EmbeddingDataset
from saetopic.training.train_sae import TrainingConfig

console = Console()


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


def run_embeddings(config: dict[str, Any]) -> tuple[int, int]:
    dataset_cfg = config["dataset"]
    embedding_cfg = config["embeddings"]

    embedder = build_embedder(config)
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

    info = model.get_topic_info()
    info["Top_Words_20"] = [
        ", ".join(word for word, _ in topic_words[topic_id])
        for topic_id in info["Topic"]
    ]
    info.to_csv(output_dir / "topic_info.csv", index=False)
    write_top_words_file(artifact_topic_words, output_dir / "top_words.txt", top_n=50)

    model.get_cluster_info().to_csv(output_dir / "clusters.csv", index=False)
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
    if stage == "embeddings":
        n_embeddings, embedding_dim = run_embeddings(config)
        console.print(f"Saved {n_embeddings:,} embeddings of dimension {embedding_dim}")
    elif stage == "train_sae":
        run_train_sae(config)
    elif stage == "topics":
        run_topics(config)
    elif stage == "evaluate":
        run_evaluate(config)
    else:
        raise ValueError(f"Unknown pretrain stage: {stage}")


def main(default_stages: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="pretrain/params.yaml.example")
    parser.add_argument(
        "--stages",
        nargs="+",
        default=None,
        help="Stages to run: embeddings train_sae topics evaluate. Defaults to pipeline.stages.",
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
