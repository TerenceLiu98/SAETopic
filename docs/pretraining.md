# Pretraining

SAETopic separates package inference from SAE pretraining.

- Use `saetopic` for fitted topic models.
- Use `saetopic-train` or `pretrain/run.py` for embedding generation,
  checkpoint training, text topic experiments, evaluation, and vision research
  stages.

## Command-Line Training

The lower-level training CLI is useful for embedding a dataset and training an
SAE checkpoint. This is a generic text example; the config-driven workflow
below currently defaults to Jina v5 omni small with 768-dimensional truncated
embeddings.

```bash
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

## Config-Driven Pretraining

The `pretrain/` workflow is the reproducible research pipeline:

```bash
cp pretrain/params.yaml.example pretrain/params.yaml
PYTHONPATH=src python pretrain/run.py --config pretrain/params.yaml
```

Common stages:

```bash
PYTHONPATH=src python pretrain/run.py --config pretrain/params.yaml --stages chunks
PYTHONPATH=src python pretrain/run.py --config pretrain/params.yaml --stages embeddings train_sae
PYTHONPATH=src python pretrain/run.py --config pretrain/params.yaml --stages topics
PYTHONPATH=src python pretrain/run.py --config pretrain/params.yaml --stages evaluate
```

## Recommended Text Defaults

The current example config uses:

- embedding model: `jinaai/jina-embeddings-v5-omni-small`
- embedding task: `clustering`
- modality: `text`
- embedding dimension: `truncate_dim: 768`
- chunking: `word`, `chunk_size: 384`, `min_words: 64`
- SAE architecture: `batch_topk`
- SAE expansion: `64`
- SAE top-k: `32`
- epochs: `10`
- checkpoint path: `checkpoints/jina-v5-omni-small-sae`

The actual useful epoch count depends on convergence. In large runs, inspect
reconstruction loss, R2, and downstream topic quality rather than treating
`n_epochs` as fixed.

## Architectures

Supported SAE architectures include:

- `standard`
- `jumprelu`
- `topk`
- `batch_topk`
- `matryoshka_batch_topk`
- `ort_batch_topk`

Architecture-specific options such as Matryoshka groups and OrtSAE
orthogonality weights are supported by the training code but are intentionally
not expanded in the minimal `params.yaml.example`.

## Evaluation

The config-driven evaluator can compute SAE-TM-style topic metrics from
`top_words.txt` artifacts. `evaluation.llm_backend: none` disables LLM-based
ratings. Set it to `vllm` when you want LLM-assisted metrics and have the
hardware available.

`evaluation.word_embeddings_dir` should point to an SAE-TM-compatible WMD
cache:

```text
w2v/
  embeddings.np.npy
  vocabulary.json
```

## Output Hygiene

Change output paths when you change:

- embedding model
- embedding dimension
- chunk size
- normalization
- SAE architecture
- training data

This avoids mixing incompatible chunks, embeddings, checkpoints, and topic
artifacts.
