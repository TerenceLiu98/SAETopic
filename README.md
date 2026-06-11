# SAETopic

> **Sparse Autoencoder topic-atom training and planned topic inference**

SAETopic is a Python package for training sparse autoencoder (SAE) topic atoms, an **unofficial** implementation of [Sparse Autoencoders are Topic Models](https://arxiv.org/abs/2511.16309) with its official [GitHub repo](https://github.com/ExplainableML/SAE-TM/tree/main). 
The current implementation focuses on memory-aware SAE training; the
inference interface and rapid topic granularity exploration are planned.

## Core Features

- **SAE Training Pipeline** — Stream HF text datasets, pre-compute embeddings, and train sparse topic atoms
- **Memory-Aware Large-Corpus Training** — Long-text chunking, multi-GPU embedding, chunked `.npy` writes, mmap training, and sparse top-k SAE training
- **Pretrained Topic Atoms** — Planned downloadable SAE weights for no-training usage
- **Retopic Without Retraining** — Planned topic granularity changes (`retopic(n_topics=...)`) without retraining SAE or corpus adaptation
- **Topic Inference API** — Planned `fit_transform`, `get_topic_info`, `visualize_topics` interface
- **Interpretable Topics** — Planned corpus-specific word interpretation for each topic atom
- **Fast Exploration** — Planned exploration of 20, 50, 100, 200 topics on the same fitted model

## Installation

```bash
pip install saetopic
```

With optional dependencies:

```bash
pip install "saetopic[viz]"   # Visualization (plotly)
pip install "saetopic[train]" # Training utilities
pip install "saetopic[dev]"   # Development tools
pip install "saetopic[all]"   # All extras
```

## Quickstart

Current training workflow:

```bash
pip install "saetopic[train]"

saetopic-train embed \
  --dataset-name HuggingFaceFW/finewiki \
  --output data/finewiki_embeddings \
  --max-samples 100000 \
  --text-chunk-size 512 \
  --text-chunk-overlap 32 \
  --max-seq-length 512 \
  --encode-batch-size 8 \
  --embedding-batch-size 64 \
  --seed 42 \
  --truncate-dim 512

saetopic-train train \
  --embeddings data/finewiki_embeddings \
  --no-normalize-embeddings \
  --input-dim 512 \
  --expansion-factor 32 \
  --top-k 32 \
  --output checkpoints/jina-v5-sae-small
```

Planned pretrained-model workflow:

```python
from saetopic import SAETopicModel

# Load pretrained model (Jina v5 embeddings with task="clustering")
model = SAETopicModel.from_pretrained("saetopic/jina-v5-sae-small")

# Fit and get topics
docs = ["Your documents here...", "More documents..."]
topics, probs = model.fit_transform(docs, n_topics=50)

# Explore topics
model.get_topic_info()  # DataFrame with topic details
model.get_topic(0)      # Top words for topic
model.visualize_topics()

# Change granularity without retraining
model.retopic(n_topics=30)

# Search topics by query
model.find_topics("climate policy", top_n=5)
```

## Key Concepts

**Topic Atoms (SAE Features)** — Learned from large corpora, these sparse features capture reusable semantic concepts that can be adapted to any downstream corpus.

**Corpus Adaptation** — SAETopic learns how topic atoms map to words in your specific vocabulary, enabling interpretable topic representations.

**Retopic** — After fitting, change the number of topics by reclustering the same topic atoms — much faster than full retraining.

## Terminology

| Research Term | User-Facing Term |
|--------------|------------------|
| SAE feature | topic atom |
| downstream dataset | your corpus / documents |
| word emission matrix | corpus-specific word interpretation |
| SAE-to-topic clustering | retopic / topic merging |

## Planned API

```python
from saetopic import SAETopicModel

model = SAETopicModel.from_pretrained("saetopic/jina-v5-sae-small")
topics, probs = model.fit_transform(docs, n_topics=50)

# Explore
model.get_topic_info()
model.get_topic(0)
model.get_representative_docs(topic_id=0)

# Change granularity
model.retopic(n_topics=100)

# Search
model.find_topics("machine learning")

# Visualize
model.visualize_topics()
model.visualize_documents()

# Save/Load
model.save("my_model")
loaded = SAETopicModel.load("my_model")
```

## Default Embedding Model

**Default**: `jinaai/jina-embeddings-v5-text-small` with `task="clustering"`

- Dimension: 1024
- Recommended training examples use Matryoshka `--truncate-dim 512`, so pair
  those saved embeddings with `--input-dim 512`
- Optimized for clustering and downstream tasks
- Strong semantic representation with permissive licensing

## Pretraining Datasets

- **Text**: [HuggingFaceFW/finewiki](https://huggingface.co/datasets/HuggingFaceFW/finewiki) (with CC-BY-SA 4.0)
- **Vision** (planned): [ILSVRC/imagenet-1k](https://huggingface.co/datasets/ILSVRC/imagenet-1k) (with [LICENSE](https://huggingface.co/datasets/ILSVRC/imagenet-1k#licensing-information))

## Training Topic Atoms

For large text corpora such as FineWiki, pre-compute embeddings once and train
SAEs from the saved sharded embedding directory. This avoids keeping the
embedding model in GPU memory during SAE training, avoids a final single-file
merge step, and makes hyperparameter sweeps much faster.

```bash
# Step 1: stream FineWiki, split long articles, and save normalized embeddings.
saetopic-train embed \
  --dataset-name HuggingFaceFW/finewiki \
  --output data/finewiki_embeddings \
  --max-samples 100000 \
  --auto-multi-gpu \
  --text-chunk-size 512 \
  --text-chunk-overlap 32 \
  --max-seq-length 512 \
  --encode-batch-size 8 \
  --embedding-batch-size 64 \
  --encode-chunk-size 128 \
  --seed 42 \
  --truncate-dim 512

# Step 2: train from the saved embeddings.
# Sharded embeddings are memory-mapped by default; skip re-normalization
# because the embed step already normalized the embeddings before saving.
saetopic-train train \
  --embeddings data/finewiki_embeddings \
  --no-normalize-embeddings \
  --input-dim 512 \
  --expansion-factor 32 \
  --top-k 32 \
  --batch-size 256 \
  --n-epochs 100 \
  --output checkpoints/jina-v5-sae-small \
  --dataset-name HuggingFaceFW/finewiki \
  --dataset-license "CC-BY-SA 4.0 / Apache 2.0"

# Optional: upload the self-contained final checkpoint.
saetopic-train upload \
  --checkpoint-dir checkpoints/jina-v5-sae-small/final \
  --repo-id your-org/jina-v5-sae-small \
  --create-repo \
  --private
```

Notes:

- FineWiki articles are long; `--text-chunk-size` prevents silent truncation and
  keeps per-batch GPU memory predictable.
- Sharded embedding output writes `manifest.partial.json` after each shard.
  If a run is interrupted, rerun the same command to resume from existing
  shards; successful completion writes `manifest.json`.
- `--auto-multi-gpu` passes all visible CUDA devices to SentenceTransformers'
  multi-process encoder. If a single document chunk is too long, reduce
  `--text-chunk-size` or `--max-seq-length`; extra GPUs improve throughput but
  do not fix single-sample OOM.
- The embed command L2-normalizes embeddings by default. Keep
  `train --no-normalize-embeddings` for those files; if you use
  `embed --no-normalize-embeddings`, let `train` normalize them unless your
  embedder already returns the exact representation you want to train on.
- SAE training uses a sparse top-k path internally, so it does not materialize
  the dense `(batch_size, n_features)` activation tensor during training.
- The top-level `saetopic` inference CLI is planned. Current command-line
  training workflows use `saetopic-train` or `python -m saetopic.training.cli`.

## Development

```bash
# Install in development mode
pip install -e ".[dev]"

# Linting
ruff check src/
ruff format src/

# Type checking
mypy src/

# Testing
pytest tests/

# Build package
python -m build
```

## Status

This project is in early development (v0.1). The API is subject to change.

**Current Milestone**: SAE training infrastructure. The training path supports
FineWiki-style large text corpora with streaming embedding, chunked embedding
storage, mmap loading, and memory-efficient sparse top-k SAE training.

Pretrained Hub loading and the full topic inference API are still in progress.

See [docs/PRD.md](docs/PRD.md) for current requirements and planning.

## Citation

If you use SAETopic, please cite:

```bibtex
@software{saetopic2026,
  title = {SAETopic: Topic-Atom Training with Sparse Autoencoders},
  author = {SAETopic Contributors},
  year = {2026},
  url = {https://github.com/yourusername/saetopic}
}
```

## License

Apache-2.0

## Legal Notice

SAETopic is an independent, unofficial clean-room implementation. All SAE code and checkpoints are trained independently using permissively licensed datasets.
