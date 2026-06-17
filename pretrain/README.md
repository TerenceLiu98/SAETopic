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
- `chunks.path`
- `dataset.chunks_path`
- `embeddings.path`
- `sae.training.output_dir`
- `evaluation.word_embeddings_dir`
- `chunks.num_proc`
- `chunks.map_batch_size`
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

Build the offline FineWiki text chunks:

```bash
PYTHONPATH=src python pretrain/run.py \
  --config pretrain/params.yaml \
  --stages chunks
```

Run embedding generation from the saved chunks and then SAE training:

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

- The default workflow is two-stage preprocessing:
  `chunks -> embeddings`. The `chunks` stage flattens FineWiki articles into a
  saved Hugging Face dataset with one row per embedding input. The `embeddings`
  stage reads that saved chunk dataset and only runs Jina encoding.
- `dataset.source: chunks` makes embedding generation read
  `dataset.chunks_path` / `chunks.path`. This avoids re-splitting FineWiki
  during GPU embedding and makes resume operate on flat chunk rows.
- `chunks.strategy: word` is the recommended fast path for large FineWiki
  runs. It uses whitespace word chunks and does not touch the Jina tokenizer
  during preprocessing. `chunks.strategy: paragraph` is available when you want
  paragraph-style chunks, but it is much slower.
- `chunks.sanitize_urls: true` replaces text URLs before saving chunks so Jina
  omni does not treat FineWiki links as image/video/audio/PDF inputs.
- Embedding generation can use multiple GPUs when `dataset.encode_device` is
  `auto_multi_cuda`. SAE training is currently single-device.
- `embeddings.chunk_size` controls how many embeddings are written per shard.
  It does not control text chunk size or encoder batch size.
- `dataset.buffer_size` controls the shuffle buffer over saved text chunks
  during embedding generation. `dataset.embedding_batch_size` controls how many
  chunk texts are handed to each dataset yield, while `dataset.encode_batch_size`
  is forwarded to SentenceTransformers/Jina as the internal encode batch size.
- The example config uses `jinaai/jina-embeddings-v5-omni-nano` with
  `model_kwargs.default_task: clustering`, `model_kwargs.modality: text`, and
  `dataset.encode_method: document` for a FineWiki-only Jina omni SAE.
- If you change the embedding model, also change `embeddings.path` and
  `sae.training.output_dir` to avoid mixing old embeddings/checkpoints. If you
  change chunking settings, also change `chunks.path`, `dataset.chunks_path`,
  and `embeddings.path`.
- `evaluation.llm_backend: none` computes only `D`. Set it to `vllm` to compute
  `CR` and `CI`.
