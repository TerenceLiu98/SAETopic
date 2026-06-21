<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="demo/logit-dark.svg">
    <img alt="SAETopic" src="https://32cf906.webp.li/2026/06/SAETopic-removebg-preview.png" width="240">
  </picture>

  <h1>SAETopic</h1>

  ![visitors](https://visitor-badge.laobi.icu/badge?page_id=TerenceLiu98/SAETopic)
</div>

> **Sparse Autoencoder topic atoms for text and vision topic modeling**

SAETopic is a Python package for topic modeling with sparse autoencoder
(SAE) topic atoms. It provides a BERTopic-style interface for fitting topics,
inspecting topic words, transforming documents, saving/loading fitted models,
and changing topic granularity without retraining the SAE. It is an
independent implementation inspired by
[Sparse Autoencoders are Topic Models](https://arxiv.org/abs/2511.16309).

## Installation

From source:

```bash
git clone https://github.com/TerenceLiu98/SAETopic.git
cd SAETopic
pip install -e ".[dev]"
```

When published to PyPI:

```bash
pip install saetopic
pip install "saetopic[train]"  # training utilities
pip install "saetopic[all]"    # optional training/viz dependencies
```

## Quick Start

Run a fully offline demo. It creates a tiny in-memory SAE and deterministic
toy embeddings, then exercises `fit_transform`, `get_topic_info`, `retopic`,
`save`, and `load`:

```bash
python examples/quickstart_local.py
```

Use a real SAE checkpoint when you have one:

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
model.get_topic_info()
model.get_topic(0)
model.get_document_info()

model.retopic(n_topics=3)
model.save("my-saetopic-model")
loaded = SAETopicModel.load("my-saetopic-model")
```

More detail: [docs/quickstart.md](docs/quickstart.md).

## Command Line

The top-level `saetopic` command is for fitted topic models:

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

Use `saetopic-train` or the config-driven `pretrain/run.py` workflow for
embedding datasets, training SAE checkpoints, and research-scale text/vision
experiments.

## Common Methods

| Task | Code |
|---|---|
| Fit the model | `.fit(docs)` |
| Fit and return assignments | `.fit_transform(docs)` |
| Transform new documents | `.transform([new_doc])` |
| Access one topic | `.get_topic(0)` |
| Access all topics | `.get_topics()` |
| Get topic table | `.get_topic_info()` |
| Get document table | `.get_document_info()` |
| Representative documents | `.get_representative_docs(topic_id=0)` |
| Search topics | `.find_topics("space mission")` |
| Change granularity | `.retopic(n_topics=100)` |
| Save model | `.save("my_model")` |
| Load model | `SAETopicModel.load("my_model")` |

## Public Attributes

| Attribute | Description |
|---|---|
| `.topics_` | Topic assignment per fitted document |
| `.document_topic_matrix_` | Document-topic probabilities |
| `.topic_word_matrix_` | Topic-word emission distributions |
| `.feature_word_matrix_` | SAE-feature-to-word emission matrix |
| `.topic_atom_clusters_` | SAE feature cluster assignment per topic atom |
| `.vocab_` | Corpus vocabulary |
| `.embeddings_` | Fitted document embeddings |

## How It Relates

- **BERTopic** clusters document embeddings and represents clusters with
  token-weighting methods such as c-TF-IDF.
- **SAE-TM** treats sparse autoencoder features as reusable topic atoms and
  learns corpus-specific word emissions for those atoms.
- **Concept** applies topic-modeling ideas to images and names the resulting
  image clusters concepts.
- **SAETopic** exposes SAE topic atoms through a package API and research
  pipelines for text BoW and vision BoVW emissions.

## Documentation

- [Quickstart](docs/quickstart.md)
- [API Guide](docs/api.md)
- [Pretraining](docs/pretraining.md)
- [Vision Topics](docs/vision.md)
- [Model Format](docs/model_format.md)
- [Current Project State](docs/PRD.md)

## Status

Implemented:

- Text topic modeling API: `fit`, `fit_transform`, `transform`, `retopic`
- Topic inspection: `get_topic_info`, `get_topic`, `get_document_info`
- SAE-TM-style artifacts: cluster info, feature clusters, theta-topic matrix
- Save/load for fitted `SAETopicModel`
- Training and pretraining utilities for SAE checkpoints
- Config-driven vision BoVW research pipeline

Planned:

- Interactive visualizations such as `visualize_topics()`
- Built-in evaluation wrappers
- First-party pretrained checkpoint releases

## Citation

If you use SAETopic, please cite:

```bibtex
@software{saetopic2026,
  title = {SAETopic: Topic-Atom Modeling with Sparse Autoencoders},
  author = {SAETopic Contributors},
  year = {2026},
  url = {https://github.com/TerenceLiu98/SAETopic}
}

# The original research paper is 
@article{girrbach2025sparse,
  title={Sparse Autoencoders are Topic Models},
  author={Girrbach, Leander and Akata, Zeynep},
  journal={arXiv preprint arXiv:2511.16309},
  year={2025}
}
```



## License

Apache-2.0

