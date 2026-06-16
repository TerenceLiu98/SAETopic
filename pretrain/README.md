# Pretrain Workflow

This directory contains the reproducible pretraining workflow. It is separate
from `examples/`; scripts here should only depend on package code under `src/`.

## 1. Create a local config

`params.yaml.example` is a machine-specific template. Copy it before editing:

```bash
cp pretrain/params.yaml.example pretrain/params.yaml
```

Edit paths and hardware settings in `pretrain/params.yaml`, especially:

- `project.output_dir`
- `embeddings.path`
- `sae.training.output_dir`
- `evaluation.word_embeddings_dir`
- `dataset.encode_batch_size`
- `dataset.embedding_batch_size`
- `embeddings.chunk_size`

## 2. Prepare gensim / WMD assets

Topic merging uses the gensim model named in:

```yaml
topics:
  merge_embedding_model: word2vec-google-news-300
```

Set the gensim data directory when needed:

```bash
export GENSIM_DATA_DIR=/home/jovyan/gensim-data
```

Evaluation `D` expects the SAE-TM WMD cache:

```text
/home/jovyan/gensim-data/w2v/
  embeddings.np.npy
  vocabulary.json
```

If you only have the original gensim vector file, convert it once:

```bash
mkdir -p /home/jovyan/gensim-data/w2v

python - <<'PY'
from pathlib import Path
import json
import numpy as np
from gensim.models import KeyedVectors

src = Path("/home/jovyan/gensim-data/word2vec-google-news-300/word2vec-google-news-300.gz")
out = Path("/home/jovyan/gensim-data/w2v")
out.mkdir(parents=True, exist_ok=True)

kv = KeyedVectors.load_word2vec_format(src, binary=True)
np.save(out / "embeddings.np.npy", kv.vectors.astype(np.float32, copy=False))
(out / "vocabulary.json").write_text(json.dumps(kv.index_to_key), encoding="utf-8")
PY
```

## 3. Run stages

Run all stages configured in `pipeline.stages`:

```bash
PYTHONPATH=src python pretrain/run.py --config pretrain/params.yaml
```

Run embedding generation and SAE training:

```bash
PYTHONPATH=src python pretrain/run.py \
  --config pretrain/params.yaml \
  --stages embeddings train_sae
```

Run downstream topic construction:

```bash
PYTHONPATH=src python pretrain/run.py \
  --config pretrain/params.yaml \
  --stages topics
```

Run topic-word evaluation:

```bash
PYTHONPATH=src python pretrain/run.py \
  --config pretrain/params.yaml \
  --stages evaluate
```

## Notes

- Embedding generation can use multiple GPUs when `dataset.encode_device` is
  `auto_multi_cuda`. SAE training is currently single-device.
- `embeddings.chunk_size` controls how many embeddings are written per shard,
  not text splitting or encoder batch size.
- The example config uses paragraph splitting with `min_sentences_per_chunk: 5`
  and `sae.training.expansion_factor: 64` to match the SAE-TM foundation-SAE
  setup more closely.
- If you change the embedding model, also change `embeddings.path` and
  `sae.training.output_dir` to avoid mixing old embeddings/checkpoints.
- `evaluation.llm_backend: none` computes only `D`. Set it to `vllm` to compute
  `CR` and `CI`.
