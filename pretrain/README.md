# Pretrain Workflow

`pretrain/` contains the config-driven research workflow. It is separate from
the package-level `saetopic` CLI:

- `saetopic`: fit, inspect, save, load, and retopic fitted models.
- `saetopic-train`: lower-level embedding and SAE checkpoint training.
- `pretrain/run.py`: reproducible multi-stage text and vision experiments.

See also:

- [docs/pretraining.md](../docs/pretraining.md)
- [docs/vision.md](../docs/vision.md)

## 1. Create a Local Config

`params.yaml.example` is a commented template. Copy it before editing:

```bash
cp pretrain/params.yaml.example pretrain/params.yaml
```

Edit paths and hardware settings in `pretrain/params.yaml`, especially:

- `project.output_dir`
- `embedding_model.device`
- `embedding_model.name`
- `embedding_model.truncate_dim`
- `chunks.path`
- `embeddings.path`
- `sae.training.output_dir`
- `sae.training.batch_size`
- `evaluation.word_embeddings_dir`
- `vision.out_dir`
- `vision.hf_dataset`

The example keeps only common parameters. Advanced keys such as `num_proc`,
`buffer_size`, `encode_batch_size`, `corpus_adapter_epochs`,
`visual_tokenizer.max_patch_samples`, and vLLM settings are still supported by
the code but are not expanded in the minimal template.

## 2. Current Example Defaults

The current template is aligned around:

- embedding model: `jinaai/jina-embeddings-v5-omni-small`
- text modality: `model_kwargs.modality: text`
- task: `clustering`
- embedding dimension: `truncate_dim: 768`
- text chunks: FineWiki word chunks of size `384`
- SAE: `batch_topk`, expansion `64`, top-k `32`
- text topic dataset example: IMDB
- vision dataset example: `clip-benchmark/wds_flickr8k`
- vision visual vocabulary: `facebook/dinov2-base`, codebook size `4096`

If you change the embedding model, also change `embeddings.path` and
`sae.training.output_dir`. If you change chunking settings, also change
`chunks.path` and `embeddings.path`.

## 3. Run Text Stages

Run all stages configured in `pipeline.stages`:

```bash
PYTHONPATH=src python pretrain/run.py --config pretrain/params.yaml
```

Run individual stages:

```bash
PYTHONPATH=src python pretrain/run.py --config pretrain/params.yaml --stages chunks

PYTHONPATH=src python pretrain/run.py \
  --config pretrain/params.yaml \
  --stages embeddings train_sae

PYTHONPATH=src python pretrain/run.py --config pretrain/params.yaml --stages topics

PYTHONPATH=src python pretrain/run.py --config pretrain/params.yaml --stages evaluate
```

## 4. Run Vision Stages

The vision pipeline is experimental and config-driven:

```bash
PYTHONPATH=src python pretrain/run.py \
  --config pretrain/params.yaml \
  --stages vision_vocab vision_bow vision_emission vision_topics vision_visualize
```

The pipeline is:

```text
image -> DINOv2 patches -> visual vocabulary -> BoVW
image -> Jina vision embedding -> SAE theta
theta + BoVW -> B_vis
B_vis -> visual topics
```

Set `vision.visualize.patch_representatives: true` when you want cropped patch
representatives instead of full-image visual-word examples.

## 5. Evaluation Assets

Topic evaluation can use an SAE-TM-compatible WMD cache:

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

`evaluation.llm_backend: none` disables LLM ratings. Set it to `vllm` when you
want LLM-assisted topic ratings and have the hardware available.

## Notes

- `chunks -> embeddings` is the recommended text pretraining path. Chunk once,
  embed once, and reuse the saved embedding shards for SAE sweeps.
- `chunks.sanitize_urls: true` prevents Jina omni from treating FineWiki URLs
  as image/video/audio/PDF inputs.
- `embeddings.chunk_size` controls embeddings per shard. It does not control
  text chunk size or encoder batch size.
- SAE training is single-device. Embedding generation can use multiple GPUs
  when configured through advanced dataset encoding settings.
- Resume is supported for chunking, embedding generation, and SAE training.
