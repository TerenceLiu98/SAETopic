"""
Build a topic model from a trained SAE checkpoint + a corpus.

Example: load the FineWiki English articles (from the local HF cache, fully
offline) and turn them into topics using a pretrained SAE.

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    python examples/build_topic_model.py \
        --ckpt /home/jovyan/helloworld-datavol-1/SAETopic/checkpoints/jina-v5-sae-small/checkpoint_epoch_16 \
        --n-docs 2000 --n-topics 50

Pipeline (see SAETopicModel.fit):
    docs -> embeddings (jina, dim = SAE input_dim) -> SAE topic atoms
         -> CorpusAdapter (feature->word) -> TopicMerger (atoms -> n_topics)
"""
# ruff: noqa: E402

import argparse
import glob
import os
import time

# Load jina-v5 + FineWiki from local cache only (this sandbox resets HF connections).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np

from saetopic import SAETopicModel

# Default FineWiki en cache location (HF hub snapshot).
FINEWIKI_EN_GLOB = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--HuggingFaceFW--finewiki/"
    "snapshots/*/data/enwiki/*.parquet"
)

import re

# Titles of pure date/biography-list scaffolding that carry no topic signal.
_DATE_LIST_TITLE = re.compile(
    r"^\s*(Deaths in|Births in|Events in|List of|Timeline of|"
    r"\d{3,4}s?|Category:|Portal:|Outline of|Index of)",
    re.IGNORECASE,
)


def _is_date_list_article(title: str) -> bool:
    """True for date/year/list scaffolding articles with no real topic."""
    return bool(_DATE_LIST_TITLE.match(title or ""))


def load_finewiki_en(n_docs: int, seed: int = 42) -> list[str]:
    """Load `n_docs` English FineWiki articles from the local parquet cache.

    Date-list / biography-list scaffolding articles (e.g. "Deaths in 1980",
    "Events in the Reformation", bare years) are skipped, since they flood
    topic words with date boilerplate.
    """
    import pyarrow.parquet as pq

    files = sorted(glob.glob(FINEWIKI_EN_GLOB))
    if not files:
        raise FileNotFoundError(
            f"No FineWiki en parquet found at {FINEWIKI_EN_GLOB}. "
            "Adjust FINEWIKI_EN_GLOB or stream the dataset online."
        )

    # Read a bit more than needed so we can sample a varied subset.
    target = max(n_docs, 200)
    pool: list[str] = []
    skipped = 0
    for f in files:
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=2048, columns=["title", "text"]):
            for row in batch.to_pylist():
                title = (row.get("title") or "").strip()
                if _is_date_list_article(title):
                    skipped += 1
                    continue
                text = (row.get("text") or "").strip()
                if not text:
                    continue
                pool.append(f"{title}\n{text}" if title else text)
                if len(pool) >= target * 3:
                    break
            if len(pool) >= target * 3:
                break
        if len(pool) >= target * 3:
            break

    print(f"  filtered out {skipped} date/list scaffolding articles")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pool), size=min(n_docs, len(pool)), replace=False)
    return [pool[i] for i in idx]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--ckpt",
        default="/home/jovyan/helloworld-datavol-1/SAETopic/checkpoints/jina-v5-sae-small/best",
        help=(
            "Path to a trained SAE checkpoint dir (config.json + model.safetensors). "
            "Note: best/ is 768-dim (matches the finewiki_embeddings on disk); "
            "checkpoint_epoch_16/ is 512-dim (older run, its training embeddings were overwritten)."
        ),
    )
    ap.add_argument("--n-docs", type=int, default=2000, help="Number of corpus documents")
    ap.add_argument("--n-topics", type=int, default=100, help="Number of topics (SAE-TM default: 100)")
    ap.add_argument("--epochs", type=int, default=30, help="CorpusAdapter epochs")
    ap.add_argument("--min-df", type=int, default=3, help="Vectorizer min document frequency")
    ap.add_argument(
        "--stop-words",
        default="wikipedia",
        help='Stop-word set: "english", "wikipedia" (english + date/bio boilerplate), or none',
    )
    ap.add_argument(
        "--theta-mode",
        default="dense",
        choices=["dense", "sparse_topk"],
        help='SAE θ mode: "dense" (SAE-TM faithful, default) or "sparse_topk"',
    )
    ap.add_argument("--out", default="topics.csv", help="Where to write topic_info CSV")
    args = ap.parse_args()

    print(f"Loading {args.n_docs} FineWiki en articles (offline)...")
    docs = load_finewiki_en(args.n_docs)
    print(f"  got {len(docs)} docs; example length: {len(docs[0])} chars")

    t0 = time.time()
    model = SAETopicModel.from_pretrained(
        args.ckpt,
        n_topics=args.n_topics,
        corpus_adapter_epochs=args.epochs,
        corpus_adapter_batch_size=1024,
        activation_batch_size=512,
        embedding_batch_size=64,
        min_df=args.min_df,
        stop_words=(None if args.stop_words.lower() == "none" else args.stop_words),
        theta_mode=args.theta_mode,
        max_seq_length=512,  # match the SAE's 512-token training chunks
        device="auto",
    )
    print(f"  SAE loaded in {time.time()-t0:.1f}s | input_dim={model.sae_input_dim_}, "
          f"n_features={model.sae_n_features_}, top_k={model.top_k_features}")

    t0 = time.time()
    topics, probs = model.fit_transform(docs)
    print(f"  fit_transform in {time.time()-t0:.1f}s")
    print(f"  embeddings_ {model.embeddings_.shape} | feature_activations_ {model.feature_activations_.shape}")
    print(f"  vocab size {len(model.vocab_)} | topic_word_matrix_ {model.topic_word_matrix_.shape}")

    info = model.get_topic_info()
    print("\n=== Topic info ===")
    print(info[["Topic", "Count", "Name"]].to_string(index=False))
    info.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}")

    print("\n=== Top words per topic (first 12) ===")
    for t in range(min(12, model.n_topics)):
        words = ", ".join(w for w, _ in model.get_topic(t, top_n=10))
        print(f"  Topic {t}: {words}")

    # The differentiator: change granularity without retraining.
    print("\n=== retopic(n_topics=20) ===")
    t0 = time.time()
    model.retopic(20)
    print(f"  retopic in {time.time()-t0:.2f}s (reuses SAE + corpus adaptation)")
    for t in range(min(8, model.n_topics)):
        words = ", ".join(w for w, _ in model.get_topic(t, top_n=8))
        print(f"  Topic {t}: {words}")


if __name__ == "__main__":
    main()
