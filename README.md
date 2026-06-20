<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="demo/logit-dark.svg">
    <img alt="SSAETopic" src="https://32cf906.webp.li/2026/06/SAETopic-removebg-preview.png" width="240">
  </picture>

  <h1>SAETopic</h1>

  ![visitors](https://visitor-badge.laobi.icu/badge?page_id=TerenceLiu98/SAETopic)
  
</div>


> **Sparse Autoencoder topic atoms for text and vision topic modeling**

SAETopic is a Python package for training sparse autoencoder (SAE) topic atoms, an **unofficial** implementation of [Sparse Autoencoders are Topic Models](https://arxiv.org/abs/2511.16309) with its official [GitHub repo](https://github.com/ExplainableML/SAE-TM/tree/main). 
The package provides a BERTopic-style interface for fitting topics from
pretrained SAE atoms, changing topic granularity without retraining the SAE,
and running memory-aware SAE pretraining pipelines.

## Core Features

- **SAE Training Pipeline** — Stream HF text datasets, pre-compute embeddings, and train sparse topic atoms
- **Memory-Aware Large-Corpus Training** — Long-text chunking, multi-GPU embedding, chunked `.npy` writes, mmap training, and sparse top-k SAE training
- **Pretrained Topic Atoms** — Load SAE checkpoints from local paths or the Hugging Face Hub
- **Retopic Without Retraining** — Change topic granularity with `retopic(n_topics=...)` without retraining the SAE or corpus adapter
- **Topic Inference API** — `fit`, `fit_transform`, `transform`, `get_topic_info`, `get_topic`, and `get_document_info`
- **Interpretable Topics** — Learn corpus-specific word emissions for each SAE topic atom
- **Text and Vision Research Pipelines** — Text BoW and vision BoVW experiments through the pretrain config pipeline

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

## Quickstart: Topic Modeling

Use a pretrained SAE checkpoint or a local checkpoint produced by
`saetopic-train`:

```python
from saetopic import SAETopicModel

docs = [
    "The rover collected images from the surface of Mars.",
    "The satellite entered orbit after a successful launch.",
    "The team won after scoring in the final minute.",
    "The coach changed tactics before the championship game.",
]

model = SAETopicModel.from_pretrained(
    "path/to/sae-checkpoint-or-hf-repo",
    n_topics=2,
    idf_weighting=True,
    stop_words="english",
)

topics, probs = model.fit_transform(docs)

print(model.get_topic_info())
print(model.get_topic(0))
print(model.get_document_info())

# Recluster the same fitted topic atoms at a different granularity.
model.retopic(n_topics=3)
```

See [examples/quickstart_text.py](examples/quickstart_text.py) for a runnable
script template.

## Command Line Usage

The top-level `saetopic` command handles fitted topic models:

```bash
saetopic fit \
  --input docs.csv \
  --text-column text \
  --model path/to/sae-checkpoint \
  --output models/my-saetopic \
  --n-topics 50

saetopic topics \
  --model models/my-saetopic \
  --output topics.csv

saetopic retopic \
  --model models/my-saetopic \
  --n-topics 100 \
  --output models/my-saetopic-100
```

`saetopic-train` remains the advanced command for embedding datasets, training
SAEs, and uploading checkpoints.

## Training Topic Atoms

For large text corpora such as FineWiki, pre-compute embeddings once and train
SAEs from the saved sharded embedding directory. This avoids keeping the
embedding model in GPU memory during SAE training, avoids a final single-file
merge step, and makes hyperparameter sweeps much faster.

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

## API Status

Implemented:

- `SAETopicModel.from_pretrained(...)`
- `fit(...)`, `fit_transform(...)`, `transform(...)`
- `retopic(...)` / `reduce_topics(...)`
- `get_topic_info()`, `get_topic(...)`, `get_topics(...)`
- `get_document_info(...)`, `get_representative_docs(...)`
- `find_topics(...)`
- `get_cluster_info()`, `get_cluster_to_feature_indices()`, `get_theta_topic_matrix(...)`
- `save(...)` / `load(...)`

Planned:

- interactive visualizations such as `visualize_topics()` and `visualize_documents()`
- built-in model evaluation wrappers
- first-party pretrained checkpoint releases

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
- **Vision**: DINOv2 patch-token BoVW experiments support Hugging Face image datasets such as `timm/mini-imagenet` and `clip-benchmark/wds_flickr8k`

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
  --steps 800000 \
  --warmup-ratio 0.1 \
  --aux-loss-weight 0.03125 \
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
- SAE training uses sparse top-k paths internally, including BatchTopK and
  Matryoshka BatchTopK, so it does not materialize the dense
  `(batch_size, n_features)` activation tensor during training.
- Interrupted SAE training can resume from saved training checkpoints with
  `train --resume` or `train --resume-from-checkpoint path/to/checkpoint`.
- The top-level `saetopic` CLI is for fitted topic models. Training workflows
  use `saetopic-train` or `python -m saetopic.training.cli`.

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

**Current Milestone**: Package surface stabilization. The model API supports
fitting topics, transforming documents, retopic granularity changes, and
SAE-TM-style topic artifacts. The training path supports FineWiki-style large
text corpora with streaming embedding, chunked embedding storage, mmap loading,
and memory-efficient sparse top-k / Matryoshka SAE training.

Interactive visualizations, built-in evaluation wrappers, and first-party
pretrained checkpoint releases are still in progress.

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
