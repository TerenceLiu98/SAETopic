# API Guide

The main public entry point is:

```python
from saetopic import SAETopicModel
```

## Construction

```python
model = SAETopicModel(
    embedding_model="jinaai/jina-embeddings-v5-text-small",
    sae_model="path/to/sae-checkpoint",
    n_topics=50,
)
```

Use `from_pretrained` to load SAE weights immediately:

```python
model = SAETopicModel.from_pretrained("path/to/sae-checkpoint")
```

`sae_model` may be:

- a local checkpoint directory
- a Hugging Face repo id
- an in-memory SAE module with `.encode`

`embedding_model` may be:

- a model id
- an object with `.encode`
- a callable returning `(n_docs, dim)` embeddings

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
| Change topic granularity | `.retopic(n_topics=100)` |
| BERTopic-compatible alias | `.reduce_topics(nr_topics=30)` |
| Save model | `.save("my_model")` |
| Load model | `SAETopicModel.load("my_model")` |

## SAE-TM Artifact Methods

| Method | Description |
|---|---|
| `.get_cluster_info()` | Topic cluster metadata with `cluster_id`, `cluster_size`, `cluster_prob`, `cluster_words`, and `cluster_ratio` |
| `.get_cluster_to_feature_indices()` | Mapping from topic id to SAE feature ids |
| `.get_theta_topic_matrix(...)` | SAE-TM-style document-topic matrix aggregated from SAE features |

## Public Attributes

Attributes ending in `_` are fitted state:

| Attribute | Description |
|---|---|
| `.docs_` | Fitted documents |
| `.embeddings_` | Document embeddings |
| `.feature_activations_` | SAE feature activations, also called theta |
| `.theta_avg_` | Corpus-average feature activation weights |
| `.vocab_` | Corpus vocabulary |
| `.bow_` | Bag-of-words matrix |
| `.feature_word_matrix_` | SAE feature-to-word emission matrix |
| `.topic_atom_clusters_` | Feature cluster label per SAE atom |
| `.topic_word_matrix_` | Topic-to-word distributions |
| `.document_topic_matrix_` | Document-topic probabilities |
| `.topics_` | Hard topic assignment per document |
| `.topic_embeddings_` | Document-weighted topic embeddings for search |

## Key Parameters

| Parameter | Purpose |
|---|---|
| `n_topics` | Number of merged topics |
| `idf_weighting` | Whether corpus adaptation uses IDF weighting |
| `theta_mode` | `"dense"` for SAE-TM-style dense theta, `"sparse_topk"` for true top-k activations |
| `merge_embedding_model` | Optional gensim word vectors for semantic topic merging |
| `use_ctfidf` | Use c-TF-IDF for display ranking |
| `min_df`, `max_df`, `vocabulary_size` | Vocabulary filtering |
| `corpus_adapter_epochs` | Feature-to-word adaptation epochs |
| `activation_batch_size` | SAE activation extraction batch size |
| `embedding_batch_size` | Embedding backend batch size |

## Not Yet Implemented

The following public methods intentionally raise `NotImplementedError`:

- `visualize_topics`
- `visualize_documents`
- `visualize_hierarchy`
- `visualize_atoms`
- `evaluate`
