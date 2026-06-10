# SAETopic

> **Sparse Autoencoder Topic Modeling with a BERTopic-like API**

SAETopic is a Python package for topic modeling using sparse autoencoder (SAE) topic atoms. It provides a BERTopic-style interface while enabling rapid topic granularity exploration without retraining the core model.

**Unofficial clean-room implementation** inspired by [SAE-TM](https://github.com/ExplainableML/SAE-TM).

## Core Features

- **Pretrained Topic Atoms** — Use downloadable SAE weights, no training required
- **Retopic Without Retraining** — Change topic granularity (`retopic(n_topics=...)`) without retraining SAE or corpus adaptation
- **BERTopic-style API** — Familiar `fit_transform`, `get_topic_info`, `visualize_topics` interface
- **Interpretable Topics** — Corpus-specific word interpretation for each topic atom
- **Fast Exploration** — Try 20, 50, 100, 200 topics on the same fitted model

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

## Why SAETopic?

| Feature | BERTopic | SAE-TM | SAETopic |
|---------|----------|-------|----------|
| BERTopic-style API | ✅ | ❌ | ✅ |
| Pretrained weights | Partial | ❌ | ✅ |
| Retopic without retraining | Partial | ✅ | ✅ |
| Interpretable topic atoms | ❌ | ✅ | ✅ |
| Quickstart in 5 minutes | ✅ | ❌ | ✅ |

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
- Optimized for clustering and downstream tasks
- Strong semantic representation with permissive licensing

## Pretraining Datasets

- **Text**: [HuggingFaceFW/finewiki](https://huggingface.co/datasets/HuggingFaceFW/finewiki) (CC-BY-SA 4.0 / Apache 2.0)
- **Vision** (v0.3+): [ILSVRC/imagenet-1k](https://huggingface.co/datasets/ILSVRC/imagenet-1k)

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

**Current Milestone**: Week 1 — Repo skeleton, API stubs

See [CLAUDE.md](CLAUDE.md) for development guidance and [docs/PRD.md](docs/PRD.md) for complete planning.

## Citation

If you use SAETopic, please cite:

```bibtex
@software{saetopic2026,
  title = {SAETopic: BERTopic-style Topic Modeling with Sparse Autoencoders},
  author = {SAETopic Contributors},
  year = {2026},
  url = {https://github.com/yourusername/saetopic}
}
```

## Inspired By

- **SAE-TM**: [Sparse Autoencoders are Topic Models](https://github.com/ExplainableML/SAE-TM)
- **BERTopic**: [https://github.com/maartengr/bertopic](https://github.com/maartengr/bertopic)
- **Concept**: [https://github.com/MaartenGr/Concept](https://github.com/MaartenGr/Concept)

## License

Apache-2.0

## Legal Notice

SAETopic is an independent, unofficial clean-room implementation. It is not affiliated with or endorsed by the authors of SAE-TM. All SAE code and checkpoints are trained independently using permissively licensed datasets.
