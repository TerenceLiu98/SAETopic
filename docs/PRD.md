# SAETopic Product Requirements

## Goal

SAETopic provides topic inference on top of sparse autoencoder topic atoms. The
intended workflow is:

1. Train reusable SAE topic atoms from large embedding corpora.
2. Adapt those atoms to a user corpus for interpretable topic words.
3. Change topic granularity without retraining the SAE.

The current implementation focuses on step 1: memory-aware SAE training from
large text corpora such as FineWiki.

## Current Scope

### Implemented

- Stream Hugging Face text datasets through `create_streaming_dataset`.
- Split long documents into bounded tokenizer chunks before embedding.
- Configure SentenceTransformers encode batch size, max sequence length, task,
  Matryoshka truncation, bf16 loading, and multi-GPU encode devices.
- Save streamed embeddings to a single `.npy` file with chunked temporary
  writes, avoiding a full in-memory concatenation.
- Load `.npy` embeddings with memory mapping for SAE training.
- Train `topk` and `batch_topk` SAE variants.
- Use sparse top-k reconstruction during training to avoid materializing the
  dense `(batch_size, n_features)` activation tensor.
- Save checkpoints, training metadata, model cards, and file checksums.
- Upload trained checkpoints to Hugging Face Hub.

### Not Yet Implemented

- Loading pretrained SAETopic models through `SAETopicModel.from_pretrained`.
- End-to-end `fit`, `fit_transform`, and `transform` topic modeling.
- Corpus-specific word interpretation for topic atoms.
- Retopic/topic merging over fitted topic atoms.
- Visualization helpers.
- Evaluation metrics.
- Vision pretraining.

## Training Workflow Requirements

### Embedding Precomputation

Large text corpora should be embedded once and saved before SAE training.
This keeps the embedding model out of memory during SAE optimization and makes
hyperparameter sweeps repeatable.

Required behavior:

- The embedder must process bounded text chunks for long FineWiki-style
  articles.
- `encode_batch_size` must directly control SentenceTransformers' internal
  batch size.
- Multi-GPU encoding must be opt-in through an explicit device list or
  `--auto-multi-gpu`.
- The final embedding file must be a valid `.npy` array even when produced from
  many temporary chunks.
- Empty streams must fail explicitly instead of writing an invalid output file.
- Embedding normalization must be explicit: the embed CLI normalizes by default,
  while `--no-normalize-embeddings` saves raw embedder outputs.

### SAE Training

Training from saved embeddings is the recommended development path.

Required behavior:

- `.npy` files should be memory-mapped by default.
- Users should be able to skip re-normalization when embeddings were normalized
  during the embed step.
- The training output directory should follow `TrainingConfig.output_dir` unless
  explicitly overridden.
- Checkpoints should include model weights, optimizer state, training state,
  config, model card, and checksums.
- Generated model cards should describe the checkpoint as a training artifact
  until pretrained loading and inference APIs are implemented.
- The train CLI may upload immediately after training, and the upload CLI should
  upload an existing self-contained `final` checkpoint without retraining.
  Both paths should expose repository creation and private repository flags.
- Sparse training must preserve the same loss semantics as the dense forward
  path.

## User-Facing CLI

The `saetopic-train` console script and `saetopic.training.cli` module are the
supported interfaces for the current training milestone. The top-level
`saetopic` inference CLI should fail clearly until the inference commands are
implemented.

Primary commands:

```bash
saetopic-train embed \
  --dataset-name HuggingFaceFW/finewiki \
  --output data/finewiki_embeddings.npy \
  --text-chunk-size 512 \
  --max-seq-length 512 \
  --encode-batch-size 8 \
  --seed 42 \
  --truncate-dim 512

saetopic-train train \
  --embeddings data/finewiki_embeddings.npy \
  --no-normalize-embeddings \
  --input-dim 512 \
  --expansion-factor 32 \
  --top-k 32

saetopic-train upload \
  --checkpoint-dir checkpoints/jina-v5-sae-small/final \
  --repo-id your-org/jina-v5-sae-small \
  --create-repo
```

The CLI should remain aligned with the Python APIs so every memory-sensitive
training option exposed in examples can also be used from the shell.

## Documentation Policy

The README should present implemented training functionality first and label
pretrained/inference workflows as planned until the corresponding methods are
implemented and tested.

Examples should prefer FineWiki-safe defaults:

- `text_chunk_size=512`
- `text_chunk_overlap=32`
- `max_seq_length=512`
- low `encode_batch_size` values for initial runs
- explicit `seed` values for streaming buffer shuffling
- precompute embeddings before SAE training

## Verification Gates

Before treating the current training milestone as stable, these commands should
pass:

```bash
uv run --with mypy mypy src
uv run --with ruff ruff check src tests examples
uv run --with pytest pytest
```
