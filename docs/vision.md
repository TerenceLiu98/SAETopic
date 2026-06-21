# Vision Topics

The vision pipeline is currently a research workflow, not yet the public
`SAETopicModel.fit(images=...)` API.

It follows the same SAE-TM structure as text:

```text
image -> Jina vision embedding -> SAE theta
image -> DINOv2 patches -> visual vocabulary -> BoVW
theta + BoVW -> B_vis
B_vis -> merged visual topics
```

## Run the Pipeline

Configure the `vision:` section in `pretrain/params.yaml`, then run:

```bash
PYTHONPATH=src python pretrain/run.py \
  --config pretrain/params.yaml \
  --stages vision_vocab vision_bow vision_emission vision_topics vision_visualize
```

Common dataset settings:

```yaml
vision:
  out_dir: results/vision_topics/flickr8k
  hf_dataset: clip-benchmark/wds_flickr8k
  hf_subset: null
  hf_split: train
  image_column: image
  label_column: null
  n_topics: [50, 100, 200]
```

## Stages

| Stage | Purpose |
|---|---|
| `vision_vocab` | Extract DINOv2 patch tokens and fit a KMeans visual vocabulary |
| `vision_bow` | Convert each image to a bag-of-visual-words histogram |
| `vision_emission` | Learn `B_vis`, the SAE-feature-to-visual-word emission matrix |
| `vision_topics` | Merge SAE features into visual topics |
| `vision_visualize` | Build topic and visual-word contact sheets |

## Important Outputs

| File | Description |
|---|---|
| `visual_vocab_centroids.npy` | DINOv2 patch KMeans centroids |
| `visual_bow.npz` | Image-by-visual-word sparse BoVW matrix |
| `visual_word_df.npy` | Visual-word document frequencies |
| `visual_word_idf.npy` | Visual-word IDF values |
| `visual_emission_probabilities.pt` | SAE-feature-to-visual-word emission matrix |
| `visual_feature_probabilities.pt` | Average SAE feature weights |
| `theta_sae_csr.npz` | Image-by-SAE-feature matrix |
| `topics_{n}/clusters.csv` | Visual topic cluster metadata |
| `topics_{n}/theta_topic_csr.npz` | Image-by-topic matrix |
| `visualizations/topics_{n}/index.html` | Topic visualization index |

## Full-Image vs Patch Representatives

`vision.visualize.patch_representatives: false` shows full images that contain
many instances of each visual word. This is fast and useful for checking topic
quality.

`vision.visualize.patch_representatives: true` reruns DINOv2 patch assignment
and crops representative patches. This is slower but gives a more faithful
visual explanation of the BoVW vocabulary.

## Current Limitations

- Vision support is config-driven and experimental.
- Image-caption joint modeling is not yet exposed as a single public API.
- Visual topic quality should be checked through representative images,
  visual-word sheets, and, when labels or captions exist, external alignment
  metrics.
