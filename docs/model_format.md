# Model Format

`SAETopicModel.save(path)` writes a fitted model directory. The format is
designed to be inspectable and stable enough for local workflows.

## Directory Layout

```text
my_model/
  config.json
  arrays.npz
  vocab.json
  docs.json
  bow.npz                 # sparse BoW, when available
  bow.npy                 # dense fallback, when available
  sae/
    config.json
    model.safetensors     # default
    model.pt              # when serialization="torch"
```

## Top-Level Metadata

`config.json` contains:

- `format`: currently `saetopic.model.v1`
- constructor parameters needed to rebuild `SAETopicModel`
- SAE architecture metadata
- topic cluster metadata
- optional custom topic labels

## Arrays

`arrays.npz` stores fitted numpy arrays:

| Key | Meaning |
|---|---|
| `embeddings_` | Fitted document embeddings |
| `feature_activations_` | SAE theta matrix |
| `theta_avg_` | Average SAE feature weights |
| `feature_word_matrix_` | SAE-feature-to-word matrix |
| `topic_atom_clusters_` | Topic assignment per SAE feature |
| `topic_word_matrix_` | Topic-word matrix |
| `document_topic_matrix_` | Document-topic matrix |
| `topic_embeddings_` | Topic embeddings for search |
| `word_embeddings_` | Cached vocabulary embeddings, if used |
| `idf_` | IDF vector, if used |
| `ctfidf_` | c-TF-IDF display matrix, if used |

## Load Behavior

`SAETopicModel.load(path)` restores:

- topic inspection methods
- `transform` for new documents
- `retopic`
- SAE-TM artifact exports

The loaded model points `sae_model` to the saved `sae/` directory.

## Serialization Options

Default:

```python
model.save("my_model", serialization="safetensors")
```

Torch fallback:

```python
model.save("my_model", serialization="torch")
```

Use the default unless you need a PyTorch `.pt` checkpoint for debugging.
