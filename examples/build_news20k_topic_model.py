"""
Build downstream topics on news-20k with a trained SAE checkpoint.

This example uses the 20 Newsgroups corpus as a compact downstream clustering
benchmark. It does not train an SAE. Instead, it loads an existing SAE
checkpoint, adapts SAE features to the news vocabulary, merges feature atoms
into topics, and demonstrates fast retopic.

Example:
    python examples/build_news20k_topic_model.py \
        --ckpt /path/to/checkpoint/best \
        --n-docs 2000 \
        --n-topics 20 \
        --out news20k_topics.csv
"""

from __future__ import annotations

import argparse
import time
from collections import Counter, defaultdict
from typing import Any

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from saetopic import SAETopicModel


def load_news20k(
    n_docs: int | None,
    seed: int,
    categories: list[str] | None,
    data_source: str,
    hf_dataset: str,
    hf_split: str,
    sklearn_data_home: str | None,
    download_sklearn: bool,
) -> tuple[list[str], np.ndarray, list[str]]:
    """Load and optionally subsample the 20 Newsgroups corpus."""
    if data_source == "hf":
        docs, labels, target_names = load_news20k_from_hf(
            dataset_name=hf_dataset,
            split=hf_split,
            categories=categories,
        )
    elif data_source == "sklearn":
        docs, labels, target_names = load_news20k_from_sklearn(
            seed=seed,
            categories=categories,
            data_home=sklearn_data_home,
            download_if_missing=download_sklearn,
        )
    else:
        raise ValueError(f"Unknown data source: {data_source}")

    if n_docs is not None and n_docs < len(docs):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(docs), size=n_docs, replace=False)
        docs = [docs[i] for i in idx]
        labels = labels[idx]

    return docs, labels, target_names


def load_news20k_from_sklearn(
    seed: int,
    categories: list[str] | None,
    data_home: str | None,
    download_if_missing: bool,
) -> tuple[list[str], np.ndarray, list[str]]:
    """Load 20 Newsgroups through sklearn's local cache/downloader."""
    dataset = fetch_20newsgroups(
        subset="all",
        categories=categories,
        data_home=data_home,
        remove=("headers", "footers", "quotes"),
        shuffle=True,
        random_state=seed,
        download_if_missing=download_if_missing,
    )

    docs = [doc.strip() for doc in dataset.data]
    labels = np.asarray(dataset.target)
    target_names = list(dataset.target_names)

    non_empty = np.asarray([bool(doc) for doc in docs])
    docs = [doc for doc, keep in zip(docs, non_empty) if keep]
    labels = labels[non_empty]

    return docs, labels, target_names


def load_news20k_from_hf(
    dataset_name: str,
    split: str,
    categories: list[str] | None,
) -> tuple[list[str], np.ndarray, list[str]]:
    """Load 20 Newsgroups from a Hugging Face dataset cache or repo."""
    try:
        from datasets import ClassLabel, concatenate_datasets, load_dataset
    except ImportError as exc:
        raise ImportError(
            "datasets is required for --data-source hf. Install with "
            "`pip install datasets` or use --data-source sklearn."
        ) from exc

    try:
        if "+" in split:
            parts = [part.strip() for part in split.split("+") if part.strip()]
            dataset_parts = [load_dataset(dataset_name, split=part) for part in parts]
            dataset = concatenate_datasets(dataset_parts)
        else:
            dataset = load_dataset(dataset_name, split=split)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load Hugging Face dataset {dataset_name!r} split {split!r}. "
            "If HF_HUB_OFFLINE=1 is set, make sure the dataset is already cached. "
            "Otherwise unset HF_HUB_OFFLINE or use --data-source sklearn with a "
            "local sklearn cache."
        ) from exc

    text_column = _pick_column(dataset, preferred=("text", "data", "content", "document"))
    label_column = _pick_column(dataset, preferred=("label", "target", "class"))
    label_name_column = _pick_column(
        dataset,
        preferred=("label_text", "label_name", "target_name", "category"),
        required=False,
    )

    label_feature = dataset.features.get(label_column)
    if isinstance(label_feature, ClassLabel):
        target_names = list(label_feature.names)
        label_to_id = {name: idx for idx, name in enumerate(target_names)}
    elif label_name_column is not None:
        names = sorted({str(value) for value in dataset[label_name_column]})
        target_names = names
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


def _pick_column(dataset: Any, preferred: tuple[str, ...], required: bool = True) -> str | None:
    """Pick a dataset column by preferred names, with a simple fallback."""
    for column in preferred:
        if column in dataset.column_names:
            return column
    if required and dataset.column_names:
        return dataset.column_names[0]
    return None


def print_cluster_label_summary(
    topics: list[int],
    labels: np.ndarray,
    target_names: list[str],
    top_n: int = 3,
) -> None:
    """Print the dominant gold categories inside each learned topic."""
    topic_to_labels: dict[int, Counter] = defaultdict(Counter)
    for topic, label in zip(topics, labels):
        topic_to_labels[int(topic)][int(label)] += 1

    print("\n=== Topic/category overlap ===")
    for topic in sorted(topic_to_labels):
        total = sum(topic_to_labels[topic].values())
        common = topic_to_labels[topic].most_common(top_n)
        summary = ", ".join(
            f"{target_names[label]}={count / total:.0%}" for label, count in common
        )
        print(f"  Topic {topic:02d} ({total:4d} docs): {summary}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt",
        required=True,
        help="Path to a trained SAE checkpoint dir with config.json and weights.",
    )
    parser.add_argument(
        "--n-docs",
        type=int,
        default=2000,
        help="Number of downstream documents. Use 0 for all documents.",
    )
    parser.add_argument("--n-topics", type=int, default=20, help="Initial topic count.")
    parser.add_argument(
        "--retopic",
        type=int,
        default=10,
        help="Retopic target after fitting. Use 0 to skip retopic.",
    )
    parser.add_argument("--epochs", type=int, default=30, help="CorpusAdapter epochs.")
    parser.add_argument("--min-df", type=int, default=5, help="Vectorizer min document frequency.")
    parser.add_argument(
        "--max-df",
        type=float,
        default=0.8,
        help="Vectorizer max document frequency ratio.",
    )
    parser.add_argument(
        "--categories",
        nargs="*",
        default=None,
        help="Optional 20 Newsgroups categories. Omit to use all categories.",
    )
    parser.add_argument(
        "--data-source",
        choices=["hf", "sklearn"],
        default="hf",
        help="Dataset source. hf uses Hugging Face cache/repo; sklearn uses sklearn's cache.",
    )
    parser.add_argument(
        "--hf-dataset",
        default="SetFit/20_newsgroups",
        help="Hugging Face dataset id used when --data-source hf.",
    )
    parser.add_argument(
        "--hf-split",
        default="train+test",
        help="Hugging Face split expression used when --data-source hf.",
    )
    parser.add_argument(
        "--sklearn-data-home",
        default=None,
        help="Local sklearn dataset cache path used when --data-source sklearn.",
    )
    parser.add_argument(
        "--download-sklearn",
        action="store_true",
        help="Allow sklearn to download from its upstream URL if the cache is missing.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--use-ctfidf",
        action="store_true",
        help="Use distinctiveness-weighted display words instead of raw emission words.",
    )
    parser.add_argument(
        "--stop-words",
        default="news20k",
        help='Stop-word set: "news20k", "english", "wikipedia", or "none".',
    )
    parser.add_argument("--out", default="news20k_topics.csv", help="Topic info CSV output path.")
    args = parser.parse_args()

    n_docs = None if args.n_docs == 0 else args.n_docs
    print("Loading news-20k downstream corpus...")
    docs, labels, target_names = load_news20k(
        n_docs=n_docs,
        seed=args.seed,
        categories=args.categories,
        data_source=args.data_source,
        hf_dataset=args.hf_dataset,
        hf_split=args.hf_split,
        sklearn_data_home=args.sklearn_data_home,
        download_sklearn=args.download_sklearn,
    )
    print(
        f"  docs={len(docs):,} | categories={len(target_names)} | "
        f"example length={len(docs[0])} chars"
    )

    t0 = time.time()
    model = SAETopicModel.from_pretrained(
        args.ckpt,
        n_topics=args.n_topics,
        corpus_adapter_epochs=args.epochs,
        corpus_adapter_batch_size=1024,
        activation_batch_size=512,
        embedding_batch_size=64,
        min_df=args.min_df,
        max_df=args.max_df,
        stop_words=(None if args.stop_words.lower() == "none" else args.stop_words),
        theta_mode="dense",
        max_seq_length=512,
        use_ctfidf=args.use_ctfidf,
        drop_empty_topics=True,
        device="auto",
    )
    print(
        f"  SAE loaded in {time.time() - t0:.1f}s | "
        f"input_dim={model.sae_input_dim_}, n_features={model.sae_n_features_}, "
        f"top_k={model.top_k_features}"
    )

    t0 = time.time()
    topics, probs = model.fit_transform(docs)
    print(f"  fit_transform in {time.time() - t0:.1f}s")
    print(
        f"  embeddings={model.embeddings_.shape} | "
        f"activations={model.feature_activations_.shape} | "
        f"vocab={len(model.vocab_)} | topics={model.n_topics}"
    )
    print(
        f"  ARI={adjusted_rand_score(labels, topics):.4f} | "
        f"NMI={normalized_mutual_info_score(labels, topics):.4f} | "
        f"probs={probs.shape}"
    )

    info = model.get_topic_info()
    print("\n=== Topic info ===")
    print(info[["Topic", "Count", "Name"]].to_string(index=False))
    info.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}")

    print("\n=== Top words per topic ===")
    for topic_id in range(model.n_topics):
        words = ", ".join(word for word, _ in model.get_topic(topic_id, top_n=10))
        print(f"  Topic {topic_id:02d}: {words}")

    print_cluster_label_summary(topics, labels, target_names)

    if args.retopic:
        print(f"\n=== retopic(n_topics={args.retopic}) ===")
        t0 = time.time()
        model.retopic(args.retopic)
        retopic_topics = model.topics_
        print(
            f"  retopic in {time.time() - t0:.2f}s | topics={model.n_topics} | "
            f"ARI={adjusted_rand_score(labels, retopic_topics):.4f} | "
            f"NMI={normalized_mutual_info_score(labels, retopic_topics):.4f}"
        )
        for topic_id in range(model.n_topics):
            words = ", ".join(word for word, _ in model.get_topic(topic_id, top_n=8))
            print(f"  Topic {topic_id:02d}: {words}")


if __name__ == "__main__":
    main()
