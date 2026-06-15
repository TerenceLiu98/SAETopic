"""
Build SAE-TM topic models for text datasets used in SAE-TM-style evaluation.

Supported datasets:
    news20k, imdb, yelp, dailymail

For each dataset and topic count, this writes:
    <out-dir>/<dataset>/topics_<n>/topic_info.csv
    <out-dir>/<dataset>/topics_<n>/clusters.csv
    <out-dir>/<dataset>/topics_<n>/top_words.txt
    <out-dir>/<dataset>/topics_<n>/cluster_to_feature_indices.json
    <out-dir>/<dataset>/topics_<n>/summary.json
    <out-dir>/<dataset>/topics_<n>/theta_topic_csr.npz  (with --save-theta-topic)

Example:
    PYTHONPATH=src python examples/build_text_topic_models.py \
        --ckpt /path/to/checkpoint/best \
        --datasets news20k imdb yelp dailymail \
        --n-docs 2000 \
        --n-topics 50 100 \
        --out-dir results/text_topics
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import save_npz
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

try:
    from examples.build_news20k_topic_model import load_news20k
except ImportError:
    from build_news20k_topic_model import load_news20k

from saetopic import SAETopicModel
from saetopic.evaluation import write_top_words_file

HF_DATASET_DEFAULTS: dict[str, dict[str, Any]] = {
    "imdb": {
        "dataset_name": "stanfordnlp/imdb",
        "subset": None,
        "split": "train+test",
        "text_column": "text",
        "label_column": "label",
    },
    "yelp": {
        "dataset_name": "Yelp/yelp_review_full",
        "subset": None,
        "split": "train",
        "text_column": "text",
        "label_column": "label",
    },
    "dailymail": {
        "dataset_name": "ccdv/cnn_dailymail",
        "subset": "3.0.0",
        "split": "train",
        "text_column": "article",
        "label_column": None,
    },
}


def _load_hf_dataset(
    dataset_key: str,
    n_docs: int | None,
    seed: int,
    overrides: dict[str, str | None],
) -> tuple[list[str], np.ndarray | None, list[str] | None]:
    try:
        from datasets import ClassLabel, concatenate_datasets, load_dataset
    except ImportError as exc:
        raise ImportError("datasets is required. Install with `pip install datasets`.") from exc

    defaults = dict(HF_DATASET_DEFAULTS[dataset_key])
    defaults.update({key: value for key, value in overrides.items() if value is not None})

    dataset_name = defaults["dataset_name"]
    subset = defaults["subset"]
    split = defaults["split"]
    text_column = defaults["text_column"]
    label_column = defaults["label_column"]

    load_kwargs = {"split": split}
    if "+" in split:
        parts = [part.strip() for part in split.split("+") if part.strip()]
        dataset_parts = [
            load_dataset(dataset_name, subset, split=part)
            if subset
            else load_dataset(dataset_name, split=part)
            for part in parts
        ]
        dataset = concatenate_datasets(dataset_parts)
    else:
        dataset = (
            load_dataset(dataset_name, subset, **load_kwargs)
            if subset
            else load_dataset(dataset_name, **load_kwargs)
        )

    if text_column not in dataset.column_names:
        raise ValueError(
            f"{dataset_name!r} split {split!r} has no text column {text_column!r}. "
            f"Available columns: {dataset.column_names}"
        )
    if label_column is not None and label_column not in dataset.column_names:
        raise ValueError(
            f"{dataset_name!r} split {split!r} has no label column {label_column!r}. "
            f"Available columns: {dataset.column_names}"
        )

    if n_docs is not None and n_docs < len(dataset):
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(dataset), size=n_docs, replace=False)
        dataset = dataset.select(indices)

    docs: list[str] = []
    labels: list[int] = []
    for row in dataset:
        text = str(row[text_column]).strip()
        if not text:
            continue
        docs.append(text)
        if label_column is not None:
            labels.append(int(row[label_column]))

    target_names = None
    if label_column is not None:
        label_feature = dataset.features.get(label_column)
        if isinstance(label_feature, ClassLabel):
            target_names = list(label_feature.names)
        else:
            target_names = [str(label) for label in sorted(set(labels))]

    return docs, (np.asarray(labels) if labels else None), target_names


def load_text_dataset(
    dataset_key: str,
    n_docs: int | None,
    seed: int,
    args: argparse.Namespace,
) -> tuple[list[str], np.ndarray | None, list[str] | None]:
    """Load one configured text dataset."""
    if dataset_key == "news20k":
        docs, labels, target_names = load_news20k(
            n_docs=n_docs,
            seed=seed,
            categories=None,
            data_source=args.news20k_data_source,
            hf_dataset=args.news20k_hf_dataset,
            hf_split=args.news20k_hf_split,
            sklearn_data_home=args.sklearn_data_home,
            download_sklearn=args.download_sklearn,
            remove_metadata=not args.keep_news_metadata,
        )
        return docs, labels, target_names

    overrides = {}
    if args.hf_dataset:
        overrides["dataset_name"] = args.hf_dataset
    if args.hf_subset:
        overrides["subset"] = args.hf_subset
    if args.hf_split:
        overrides["split"] = args.hf_split
    if args.text_column:
        overrides["text_column"] = args.text_column
    if args.label_column:
        overrides["label_column"] = args.label_column
    return _load_hf_dataset(dataset_key, n_docs=n_docs, seed=seed, overrides=overrides)


def build_model(args: argparse.Namespace, n_topics: int) -> SAETopicModel:
    vocabulary_size = None if args.vocabulary_size == 0 else args.vocabulary_size
    merge_embedding_model = (
        None if args.merge_embedding_model.lower() == "none" else args.merge_embedding_model
    )
    return SAETopicModel.from_pretrained(
        args.ckpt,
        n_topics=n_topics,
        merge_embedding_model=merge_embedding_model,
        corpus_adapter_epochs=args.epochs,
        corpus_adapter_batch_size=args.corpus_adapter_batch_size,
        activation_batch_size=args.activation_batch_size,
        embedding_batch_size=args.embedding_batch_size,
        vocabulary_size=vocabulary_size,
        min_df=args.min_df,
        max_df=args.max_df,
        stop_words=(None if args.stop_words.lower() == "none" else args.stop_words),
        theta_mode=args.theta_mode,
        max_seq_length=args.max_seq_length,
        use_ctfidf=args.use_ctfidf,
        drop_empty_topics=False,
        random_state=args.seed,
        device=args.device,
    )


def save_topic_outputs(
    model: SAETopicModel,
    docs: list[str],
    labels: np.ndarray | None,
    output_dir: Path,
    elapsed: float,
    save_theta_topic: bool = False,
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

    clusters = model.get_cluster_info()
    clusters.to_csv(output_dir / "clusters.csv", index=False)
    with (output_dir / "cluster_to_feature_indices.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(model.get_cluster_to_feature_indices(), f, indent=2, sort_keys=True)

    if save_theta_topic:
        theta_topic = model.get_theta_topic_matrix(normalize=False, sparse=True)
        save_npz(output_dir / "theta_topic_csr.npz", theta_topic)

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
        "theta_avg_shape": (
            list(model.theta_avg_.shape) if model.theta_avg_ is not None else None
        ),
        "merge_embedding_model": model.merge_embedding_model,
        "wrote_clusters_csv": True,
        "wrote_theta_topic_csr": save_theta_topic,
    }
    if labels is not None and model.topics_ is not None and len(labels) == len(model.topics_):
        summary["ARI"] = adjusted_rand_score(labels, model.topics_)
        summary["NMI"] = normalized_mutual_info_score(labels, model.topics_)

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)


def run_dataset(dataset_key: str, args: argparse.Namespace, model: SAETopicModel) -> None:
    n_docs = None if args.n_docs == 0 else args.n_docs
    print(f"\n=== Dataset: {dataset_key} ===")
    docs, labels, target_names = load_text_dataset(
        dataset_key=dataset_key,
        n_docs=n_docs,
        seed=args.seed,
        args=args,
    )
    del target_names
    print(f"  docs={len(docs):,} | labels={'yes' if labels is not None else 'no'}")

    topic_counts = list(dict.fromkeys(args.n_topics))

    t0 = time.time()
    topics, probs = model.fit_transform(docs, n_topics=topic_counts[0])
    del topics, probs
    elapsed = time.time() - t0
    output_dir = Path(args.out_dir) / dataset_key / f"topics_{model.n_topics}"
    save_topic_outputs(
        model,
        docs,
        labels,
        output_dir,
        elapsed,
        save_theta_topic=args.save_theta_topic,
    )
    print(f"  wrote {output_dir}")

    for n_topics in topic_counts[1:]:
        t0 = time.time()
        model.retopic(n_topics)
        elapsed = time.time() - t0
        output_dir = Path(args.out_dir) / dataset_key / f"topics_{n_topics}"
        save_topic_outputs(
            model,
            docs,
            labels,
            output_dir,
            elapsed,
            save_theta_topic=args.save_theta_topic,
        )
        print(f"  wrote {output_dir}")

    gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, help="Trained SAE checkpoint directory.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["news20k", "imdb", "yelp", "dailymail"],
        choices=["news20k", "imdb", "yelp", "dailymail"],
        help="Datasets to run.",
    )
    parser.add_argument(
        "--n-docs",
        type=int,
        default=2000,
        help="Documents per dataset. Use 0 for all rows.",
    )
    parser.add_argument(
        "--n-topics",
        nargs="+",
        type=int,
        default=[50, 100, 200, 300, 500],
        help="Topic granularities to save. Later values use retopic.",
    )
    parser.add_argument("--out-dir", default="results/text_topics", help="Output root.")
    parser.add_argument("--epochs", type=int, default=30, help="CorpusAdapter epochs.")
    parser.add_argument("--vocabulary-size", type=int, default=5000, help="0 for unlimited.")
    parser.add_argument("--min-df", type=int, default=5)
    parser.add_argument("--max-df", type=float, default=1.0)
    parser.add_argument("--stop-words", default="saetm")
    parser.add_argument(
        "--merge-embedding-model",
        default="word2vec-google-news-300",
        help=(
            "Gensim word embedding model for SAE-TM feature merging. "
            "Use 'none' to reuse the document embedding backend for vocab words."
        ),
    )
    parser.add_argument("--theta-mode", default="dense", choices=["dense", "sparse_topk"])
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--use-ctfidf", action="store_true")
    parser.add_argument(
        "--save-theta-topic",
        action="store_true",
        help="Write SAE-TM-style theta_topic_csr.npz for each topic granularity.",
    )
    parser.add_argument("--corpus-adapter-batch-size", type=int, default=512)
    parser.add_argument("--activation-batch-size", type=int, default=256)
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--news20k-data-source", choices=["hf", "sklearn"], default="hf")
    parser.add_argument("--news20k-hf-dataset", default="SetFit/20_newsgroups")
    parser.add_argument("--news20k-hf-split", default="train+test")
    parser.add_argument("--sklearn-data-home", default=None)
    parser.add_argument("--download-sklearn", action="store_true")
    parser.add_argument("--keep-news-metadata", action="store_true")

    parser.add_argument(
        "--hf-dataset",
        default=None,
        help="Override HF dataset id for non-news20k runs; intended for one dataset at a time.",
    )
    parser.add_argument("--hf-subset", default=None, help="Override HF config/subset.")
    parser.add_argument("--hf-split", default=None, help="Override HF split.")
    parser.add_argument("--text-column", default=None, help="Override text column.")
    parser.add_argument("--label-column", default=None, help="Override label column.")
    args = parser.parse_args()

    if args.hf_dataset and len([d for d in args.datasets if d != "news20k"]) > 1:
        raise ValueError("--hf-dataset override is only safe when running one non-news20k dataset")

    topic_counts = list(dict.fromkeys(args.n_topics))
    model = build_model(args, n_topics=topic_counts[0])
    print(
        "Loaded SAE-TM once for all datasets "
        f"(input_dim={model.sae_input_dim_}, n_features={model.sae_n_features_})"
    )

    for dataset_key in args.datasets:
        run_dataset(dataset_key, args, model)


if __name__ == "__main__":
    main()
