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

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from saetopic import SAETopicModel


def load_news20k(
    n_docs: int | None,
    seed: int,
    categories: list[str] | None,
) -> tuple[list[str], np.ndarray, list[str]]:
    """Load and optionally subsample the 20 Newsgroups corpus."""
    dataset = fetch_20newsgroups(
        subset="all",
        categories=categories,
        remove=("headers", "footers", "quotes"),
        shuffle=True,
        random_state=seed,
    )

    docs = [doc.strip() for doc in dataset.data]
    labels = np.asarray(dataset.target)
    target_names = list(dataset.target_names)

    non_empty = np.asarray([bool(doc) for doc in docs])
    docs = [doc for doc, keep in zip(docs, non_empty) if keep]
    labels = labels[non_empty]

    if n_docs is not None and n_docs < len(docs):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(docs), size=n_docs, replace=False)
        docs = [docs[i] for i in idx]
        labels = labels[idx]

    return docs, labels, target_names


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
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--use-ctfidf",
        action="store_true",
        help="Use distinctiveness-weighted display words instead of raw emission words.",
    )
    parser.add_argument("--out", default="news20k_topics.csv", help="Topic info CSV output path.")
    args = parser.parse_args()

    n_docs = None if args.n_docs == 0 else args.n_docs
    print("Loading news-20k downstream corpus...")
    docs, labels, target_names = load_news20k(
        n_docs=n_docs,
        seed=args.seed,
        categories=args.categories,
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
        stop_words="english",
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
