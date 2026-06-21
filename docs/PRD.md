# SAETopic Current Project State

This document records the current product state and near-term requirements. It
replaces the older training-only milestone notes.

## Goal

SAETopic provides topic modeling on top of sparse autoencoder topic atoms:

1. Train reusable SAE topic atoms from large embedding corpora.
2. Adapt those atoms to a user corpus through observable token emissions.
3. Merge topic atoms into different topic granularities without retraining the
   SAE.
4. Expose the workflow through a Python package API and a lightweight CLI.

## Current Scope

### Implemented

Package API:

- `SAETopicModel.from_pretrained`
- `fit`, `fit_transform`, and `transform`
- `retopic` / `reduce_topics`
- `get_topic_info`, `get_topic`, `get_topics`
- `get_document_info`, `get_representative_docs`
- `find_topics`
- `get_cluster_info`, `get_cluster_to_feature_indices`
- `get_theta_topic_matrix`
- `save` / `load`

CLI:

- `saetopic fit`
- `saetopic topics`
- `saetopic retopic`
- `saetopic-train embed`
- `saetopic-train train`
- `saetopic-train upload`

Training:

- Stream Hugging Face text datasets through embedding pipelines.
- Split long documents into bounded chunks before embedding.
- Configure SentenceTransformers/Jina encode batch size, max sequence length,
  task, truncation, dtype, and multi-GPU encode devices.
- Save sharded embedding directories with manifests and resume metadata.
- Train `standard`, `jumprelu`, `topk`, `batch_topk`,
  `matryoshka_batch_topk`, and `ort_batch_topk` SAE variants.
- Resume SAE training from latest or explicit checkpoints.
- Save checkpoints, training metadata, model cards, and file checksums.
- Upload trained checkpoints to Hugging Face Hub.

Text topic experiments:

- Official-style preprocessing order for SAE-TM document processing.
- Corpus adaptation from SAE theta and BoW to feature-word emissions.
- Topic merging with cluster-size-descending export.
- SAE-TM-style `theta_topic_csr.npz` export.
- D/CI/CR evaluation plumbing.

Vision research pipeline:

- DINOv2 patch-token KMeans visual vocabulary.
- Image BoVW construction from Hugging Face image datasets.
- Visual emission matrix `B_vis`.
- Visual topic merging.
- Full-image visual-word and topic contact-sheet visualization.
- Optional patch representatives.

### Planned

- Interactive package visualizations: `visualize_topics`,
  `visualize_documents`, `visualize_hierarchy`, `visualize_atoms`.
- Built-in model evaluation wrappers from the package API.
- First-party pretrained checkpoint releases.
- Public image API, for example `fit_images` or multimodal `fit`.
- Documentation site with rendered examples.

## Documentation Requirements

The repository documentation should keep these layers separate:

1. `README.md`: package-facing surface, short and runnable.
2. `docs/quickstart.md`: local demo and checkpoint workflow.
3. `docs/api.md`: methods, attributes, and key parameters.
4. `docs/pretraining.md`: SAE training and text research pipeline.
5. `docs/vision.md`: vision BoVW research workflow.
6. `docs/model_format.md`: save/load directory format.
7. `pretrain/README.md`: operational notes for `pretrain/run.py`.

README examples should not require unpublished checkpoints. If a checkpoint is
required, the example must say so explicitly.

## Near-Term Requirements

Package surface:

- Keep `SAETopicModel` API stable enough for examples and CLI usage.
- Maintain a fully offline smoke-test example.
- Keep save/load roundtrip tests passing.

Pretraining:

- Keep `pretrain/params.yaml.example` minimal and commented.
- Keep advanced parameters supported in code but out of the main example unless
  they are part of the common workflow.
- Keep text and vision defaults synchronized across README, docs, and
  `pretrain/README.md`.

Vision:

- Treat vision as a research pipeline until a public image API is implemented.
- Prefer patch-grounded visual-word explanations for paper-facing experiments.

## Verification Gates

Targeted package checks:

```bash
uv run ruff check src/saetopic/cli.py src/saetopic/model.py src/saetopic/serialization.py tests/test_cli.py tests/test_model.py examples/quickstart_local.py
uv run pytest tests/test_cli.py tests/test_imports.py tests/test_model.py
python examples/quickstart_local.py
```

Full repository checks before release:

```bash
uv run ruff check src tests examples
uv run pytest tests
```

Known caveat: full-suite failures should be triaged separately if they come
from unrelated streaming-order assumptions or legacy research tests.
