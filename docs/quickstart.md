# Quickstart

This guide shows two paths:

1. A fully offline demo that runs without downloads.
2. A real checkpoint workflow for fitted topic modeling.

## Offline Demo

The quickest smoke test is:

```bash
python examples/quickstart_local.py
```

This example builds:

- a tiny in-memory `BatchTopKSAE`
- deterministic toy embeddings
- a small nine-document corpus

It then runs:

```python
topics, probs = model.fit_transform(docs)
model.get_topic_info()
model.get_topic(0)
model.retopic(n_topics=2)
model.save(path)
loaded = SAETopicModel.load(path)
```

The output is not meant to be a high-quality topic model. It is a no-network
API check that proves the package surface works end to end.

## Real Checkpoint Workflow

Use a local SAE checkpoint or a Hugging Face repo containing an SAE checkpoint:

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
    min_df=1,
    idf_weighting=True,
    stop_words="english",
)

topics, probs = model.fit_transform(docs)
```

Inspect the fitted model:

```python
model.get_topic_info()
model.get_topic(0, top_n=10)
model.get_document_info()
model.get_representative_docs(topic_id=0)
```

Change the number of topics without retraining the SAE or corpus adapter:

```python
model.retopic(n_topics=3)
```

Save and load the fitted model:

```python
model.save("my-saetopic-model")
loaded = SAETopicModel.load("my-saetopic-model")
```

## CLI Workflow

Fit from a text file, JSONL, JSON, CSV, or TSV:

```bash
saetopic fit \
  --input docs.csv \
  --text-column text \
  --model path/to/sae-checkpoint \
  --output models/my-saetopic \
  --n-topics 50
```

Export topics:

```bash
saetopic topics \
  --model models/my-saetopic \
  --output topics.csv
```

Retopic an existing fitted model:

```bash
saetopic retopic \
  --model models/my-saetopic \
  --n-topics 100 \
  --output models/my-saetopic-100
```

## Notes

- `SAETopicModel.from_pretrained(...)` loads the SAE checkpoint. It does not
  fit topics until `fit` or `fit_transform` is called.
- `embedding_model` can be a Hugging Face/SentenceTransformers model id, an
  object exposing `encode`, or a callable that maps `list[str]` to a 2D numpy
  array.
- The embedding dimension must match the SAE input dimension.
- There are no first-party pretrained checkpoints in this repository yet; use a
  local checkpoint from `saetopic-train` or the config-driven pretraining
  workflow.
