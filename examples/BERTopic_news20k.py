"""
Run a BERTopic baseline on news-20k.

This is a standalone comparison script for the SAE-TM news-20k example. It
uses the same corpus loading and metadata stripping code, then reports topic
words, topic/category overlap, ARI/NMI, and writes BERTopic topic info to CSV.

Example:
    PYTHONPATH=src python examples/BERTopic_news20k.py \
        --data-source hf \
        --hf-dataset SetFit/20_newsgroups \
        --hf-split train+test \
        --n-docs 2000 \
        --n-topics 20 \
        --out bertopic_news20k_topics.csv
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

try:
    from examples.build_news20k_topic_model import load_news20k, print_cluster_label_summary
except ImportError:
    from build_news20k_topic_model import load_news20k, print_cluster_label_summary

from saetopic.evaluation import write_top_words_file


def _import_bertopic():
    try:
        from bertopic import BERTopic
    except ImportError as exc:
        raise ImportError(
            "BERTopic is required for this example. Install it with "
            "`pip install bertopic` or `uv pip install bertopic`."
        ) from exc
    return BERTopic


def _build_vectorizer(args: argparse.Namespace) -> CountVectorizer:
    stop_words = None
    if args.stop_words.lower() == "english":
        stop_words = "english"
    elif args.stop_words.lower() not in {"none", ""}:
        raise ValueError('BERTopic baseline supports --stop-words "english" or "none".')

    max_features = None if args.vocabulary_size == 0 else args.vocabulary_size
    return CountVectorizer(
        stop_words=stop_words,
        min_df=args.min_df,
        max_df=args.max_df,
        max_features=max_features,
        ngram_range=(1, args.max_ngram),
    )


def _print_top_words(model, top_n: int) -> None:
    topic_ids = sorted(topic for topic in model.get_topics() if topic != -1)
    print("\n=== Top words per topic ===")
    for topic_id in topic_ids:
        words = ", ".join(word for word, _ in model.get_topic(topic_id)[:top_n])
        print(f"  Topic {topic_id:02d}: {words}")


def _get_topic_words(model, top_n: int = 20) -> dict[int, list[tuple[str, float]]]:
    return {
        topic_id: model.get_topic(topic_id)[:top_n]
        for topic_id in sorted(topic for topic in model.get_topics() if topic != -1)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-docs",
        type=int,
        default=2000,
        help="Number of downstream documents. Use 0 for all documents.",
    )
    parser.add_argument(
        "--n-topics",
        type=int,
        default=20,
        help="Target number of topics after BERTopic reduction. Use 0 for automatic.",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="BERTopic embedding model id or local path.",
    )
    parser.add_argument(
        "--min-topic-size",
        type=int,
        default=10,
        help="Minimum cluster size used by BERTopic/HDBSCAN.",
    )
    parser.add_argument(
        "--calculate-probabilities",
        action="store_true",
        help="Compute BERTopic probability matrix. More informative but can use much more memory.",
    )
    parser.add_argument(
        "--vocabulary-size",
        type=int,
        default=5000,
        help="Maximum vocabulary size for topic representations. Use 0 for unlimited.",
    )
    parser.add_argument("--min-df", type=int, default=5, help="Vectorizer min document frequency.")
    parser.add_argument(
        "--max-df",
        type=float,
        default=0.8,
        help="Vectorizer max document frequency ratio.",
    )
    parser.add_argument(
        "--max-ngram",
        type=int,
        default=1,
        help="Maximum n-gram length for BERTopic topic words.",
    )
    parser.add_argument(
        "--stop-words",
        default="english",
        help='Stop-word mode for BERTopic vectorizer: "english" or "none".',
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
    parser.add_argument(
        "--keep-news-metadata",
        action="store_true",
        help="Keep 20 Newsgroups headers, footers, and quoted text.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--out", default="bertopic_news20k_topics.csv", help="CSV output path.")
    parser.add_argument(
        "--top-words-out",
        default=None,
        help="Optional top_words.txt output. Defaults to <out-dir>/top_words.txt.",
    )
    args = parser.parse_args()

    bertopic_cls = _import_bertopic()

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
        remove_metadata=not args.keep_news_metadata,
    )
    print(
        f"  docs={len(docs):,} | categories={len(target_names)} | "
        f"example length={len(docs[0])} chars"
    )

    vectorizer_model = _build_vectorizer(args)
    nr_topics = None if args.n_topics == 0 else args.n_topics

    model = bertopic_cls(
        embedding_model=args.embedding_model,
        vectorizer_model=vectorizer_model,
        nr_topics=nr_topics,
        min_topic_size=args.min_topic_size,
        calculate_probabilities=args.calculate_probabilities,
        verbose=True,
    )

    print("Fitting BERTopic...")
    t0 = time.time()
    topics, probs = model.fit_transform(docs)
    elapsed = time.time() - t0

    topics_array = np.asarray(topics)
    topic_ids = sorted(topic for topic in set(topics) if topic != -1)
    n_outliers = int(np.sum(topics_array == -1))
    probs_shape = None if probs is None else probs.shape

    print(f"  fit_transform in {elapsed:.1f}s")
    print(
        f"  topics={len(topic_ids)} | outliers={n_outliers:,} | "
        f"probs={probs_shape}"
    )
    print(
        f"  ARI={adjusted_rand_score(labels, topics):.4f} | "
        f"NMI={normalized_mutual_info_score(labels, topics):.4f}"
    )

    info = model.get_topic_info()
    print("\n=== Topic info ===")
    print(info[["Topic", "Count", "Name"]].to_string(index=False))
    info.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}")
    top_words_out = (
        Path(args.top_words_out)
        if args.top_words_out
        else Path(args.out).with_name("top_words.txt")
    )
    write_top_words_file(_get_topic_words(model, top_n=20), top_words_out, top_n=20)
    print(f"Wrote {top_words_out}")

    _print_top_words(model, top_n=10)
    print_cluster_label_summary(topics, labels, target_names)


if __name__ == "__main__":
    main()
