"""
Main SAETopicModel class with the planned topic inference API.

The pipeline chains the pretrained SAE topic atoms into end-to-end topic
modeling::

    docs
      -> EmbeddingBackend          (embeddings_)
      -> extract_activations(SAE)  (feature_activations_)
      -> CorpusVectorizer          (vocab_, bow)
      -> CorpusAdapter             (feature_word_matrix_)
      -> TopicMerger               (topic_word_matrix_, document_topic_matrix_)
      -> topics / probs

``retopic`` re-runs only the TopicMerger step, reusing the SAE activations and
the corpus-specific feature-word matrix.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


class SAETopicModel:
    """
    SAETopic: topic modeling with SAE topic atoms.

    Parameters
    ----------
    embedding_model : str or callable, default="jinaai/jina-embeddings-v5-text-small"
        Model to use for embedding documents. A Hugging Face model id, an
        object with ``encode``, or a callable mapping ``list[str]`` to
        ``(n_docs, dim)`` arrays.
    embedding_task : str, default="clustering"
        Task type for Jina embeddings (e.g., "clustering", "retrieval")
    sae_model : str, default="saetopic/jina-v5-sae-small"
        Pretrained SAE checkpoint: a Hugging Face id or a local checkpoint
        directory. May also be an already-instantiated SAE module.
    n_topics : int, default=50
        Initial number of topics to generate
    top_k_features : int, default=32
        Number of top-k features (informational; read from the SAE checkpoint
        at load time)
    min_topic_size : int or None, default=None
        Minimum number of documents per topic (reserved for post-processing)
    vectorizer_model : Any, default=None
        Custom fitted vectorizer exposing ``fit``/``transform``/``vocab_``.
        If None, a :class:`CorpusVectorizer` is built from the parameters below.
    idf_weighting : bool, default=True
        Whether to use IDF weighting in corpus adaptation
    device : str, default="auto"
        Device for computation ("auto", "cpu", "cuda", "mps")
    random_state : int, default=42
        Random seed for reproducibility
    corpus_adapter_epochs : int, default=50
        Number of epochs for the CorpusAdapter (feature->word) optimization
    corpus_adapter_batch_size : int, default=1024
        Batch size for CorpusAdapter training
    activation_batch_size : int, default=512
        Batch size for SAE activation extraction
    embedding_batch_size : int, default=64
        Batch size for the embedding backend
    cluster_method : str, default="kmeans"
        Topic-merging clustering method ("kmeans", "agglomerative")
    sparsity_threshold : float, default=0.9
        Tau threshold for sparsifying the feature-word matrix before clustering
    vocabulary_size : int or None, default=None
        Maximum vocabulary size (None = unlimited)
    min_df : int, default=2
        Minimum document frequency for vocabulary terms
    max_df : float, default=0.95
        Maximum document frequency (ratio) for vocabulary terms
    stop_words : str or None, default="english"
        Stop-word/preprocessing mode for the vectorizer: "english", "saetm",
        None, or a custom list.
    theta_mode : str, default="dense"
        How SAE feature activations (θ) are obtained:
        - "dense": ``sae.encode()`` dense ReLU (SAE-TM faithful, default).
        - "sparse_topk": true top-k sparse activation (sharper atoms).
        Applies to both word-emission training and the document-topic matrix.
    max_seq_length : int or None, default=512
        Max input sequence length for the embedder. Set to match the SAE's
        training chunk size so inference embeddings stay in-distribution.
    use_ctfidf : bool, default=True
        Use c-TF-IDF topic-word scoring for ``get_topic`` / ``get_topic_info``,
        so topic words are ranked by distinctiveness across topics rather than
        raw emission probability. Down-weights corpus-common words (e.g.
        "events", "born", month names).
    drop_empty_topics : bool, default=True
        Drop clusters to which no document is assigned (count == 0) and
        renumber the remaining topics, so the output never shows empty topics.
    """

    def __init__(
        self,
        embedding_model: str | Callable = "jinaai/jina-embeddings-v5-text-small",
        embedding_task: str = "clustering",
        sae_model: str = "saetopic/jina-v5-sae-small",
        n_topics: int = 50,
        top_k_features: int = 32,
        min_topic_size: int | None = None,
        vectorizer_model: Any = None,
        idf_weighting: bool = True,
        device: str = "auto",
        random_state: int = 42,
        corpus_adapter_epochs: int = 50,
        corpus_adapter_batch_size: int = 1024,
        activation_batch_size: int = 512,
        embedding_batch_size: int = 64,
        cluster_method: str = "kmeans",
        sparsity_threshold: float = 0.9,
        vocabulary_size: int | None = None,
        min_df: int = 2,
        max_df: float = 0.95,
        max_seq_length: int | None = 512,
        use_ctfidf: bool = True,
        drop_empty_topics: bool = True,
        stop_words: str | None = "english",
        theta_mode: str = "dense",
    ):
        self.embedding_model = embedding_model
        self.embedding_task = embedding_task
        self.sae_model = sae_model
        self.n_topics = n_topics
        self.top_k_features = top_k_features
        self.min_topic_size = min_topic_size
        self.vectorizer_model = vectorizer_model
        self.idf_weighting = idf_weighting
        self.device = device
        self.random_state = random_state
        self.corpus_adapter_epochs = corpus_adapter_epochs
        self.corpus_adapter_batch_size = corpus_adapter_batch_size
        self.activation_batch_size = activation_batch_size
        self.embedding_batch_size = embedding_batch_size
        self.cluster_method = cluster_method
        self.sparsity_threshold = sparsity_threshold
        self.vocabulary_size = vocabulary_size
        self.min_df = min_df
        self.max_df = max_df
        self.max_seq_length = max_seq_length
        self.use_ctfidf = use_ctfidf
        self.drop_empty_topics = drop_empty_topics
        self.stop_words = stop_words
        self.theta_mode = theta_mode

        # Pipeline components (set during fit)
        self.sae_: Any = None
        self.sae_input_dim_: int | None = None
        self.sae_n_features_: int | None = None
        self.vectorizer_: Any = None
        self.adapter_: Any = None
        self.merger_: Any = None
        self.representation_: Any = None
        self._embedding_backend_: Any = None

        # Internal attributes (set during fit)
        self.docs_: list[str] | None = None
        self.embeddings_: np.ndarray | None = None
        self.feature_activations_: np.ndarray | None = None
        self.feature_word_matrix_: np.ndarray | None = None
        self.topic_atom_clusters_: np.ndarray | None = None
        self.topic_word_matrix_: np.ndarray | None = None
        self.document_topic_matrix_: np.ndarray | None = None
        self.topic_embeddings_: np.ndarray | None = None
        self.word_embeddings_: np.ndarray | None = None
        self.vocab_: list[str] | None = None
        self.idf_: np.ndarray | None = None
        self.topics_: list[int] | None = None
        self.bow_: Any = None
        self.ctfidf_: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Construction / loading helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs) -> "SAETopicModel":
        """
        Load a pretrained SAETopic model from a local path or Hugging Face Hub.

        This resolves and loads the SAE checkpoint. Document embedding and
        topic inference happen lazily when ``fit`` / ``fit_transform`` is
        called.

        Parameters
        ----------
        model_id : str
            Hugging Face model id (e.g., "saetopic/jina-v5-sae-small") or a
            local checkpoint directory.
        **kwargs
            Additional arguments forwarded to the constructor.

        Returns
        -------
        SAETopicModel
            Initialized model with the pretrained SAE checkpoint loaded
        """
        init_keys = {
            "embedding_model",
            "embedding_task",
            "n_topics",
            "top_k_features",
            "min_topic_size",
            "vectorizer_model",
            "idf_weighting",
            "device",
            "random_state",
            "corpus_adapter_epochs",
            "corpus_adapter_batch_size",
            "activation_batch_size",
            "embedding_batch_size",
            "cluster_method",
            "sparsity_threshold",
            "vocabulary_size",
            "min_df",
            "max_df",
            "max_seq_length",
            "use_ctfidf",
            "drop_empty_topics",
            "stop_words",
            "theta_mode",
        }
        init_kwargs = {k: v for k, v in kwargs.items() if k in init_keys}
        instance = cls(sae_model=model_id, **init_kwargs)
        instance._ensure_sae()
        return instance

    def _ensure_sae(self) -> None:
        """Load the SAE (if not already loaded) and record its dimensions."""
        if self.sae_ is not None:
            return

        if isinstance(self.sae_model, (str, Path)):
            from saetopic.sae.loaders import SAECheckpoint

            checkpoint = SAECheckpoint.from_pretrained(self.sae_model)
            self.sae_ = checkpoint.get_model()
            self.sae_input_dim_ = checkpoint.embedding_dim
            self.sae_n_features_ = checkpoint.n_features
            self.top_k_features = checkpoint.top_k
        elif hasattr(self.sae_model, "encode"):
            self.sae_ = self.sae_model
            self.sae_input_dim_ = getattr(self.sae_, "input_dim", None)
            self.sae_n_features_ = getattr(self.sae_, "n_features", None)
            self.top_k_features = getattr(self.sae_, "top_k", self.top_k_features)
        else:
            raise TypeError(
                "sae_model must be a path/repo id or an SAE module with .encode"
            )

    def _get_embedding_backend(self):
        # Cache a single backend: remote-code models (e.g. Jina v5) can fail
        # when SentenceTransformer is instantiated a second time in-process, so
        # we build it once and reuse it for both documents and vocabulary.
        if self._embedding_backend_ is not None:
            return self._embedding_backend_

        from saetopic.embeddings import EmbeddingBackend

        self._embedding_backend_ = EmbeddingBackend(
            model=self.embedding_model,
            task=self.embedding_task,
            device=self.device,
            batch_size=self.embedding_batch_size,
            truncate_dim=self.sae_input_dim_,
            normalize=True,
            max_seq_length=self.max_seq_length,
        )
        return self._embedding_backend_

    def _compute_word_embeddings(self, vocab: list[str]) -> np.ndarray:
        """Embed the vocabulary words with the document embedding model."""
        backend = self._get_embedding_backend()
        return backend.embed(list(vocab))

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit(
        self,
        docs: list[str],
        embeddings: np.ndarray | None = None,
        y: np.ndarray | None = None,
        n_topics: int | None = None,
    ) -> "SAETopicModel":
        """
        Fit the topic model to documents.

        Parameters
        ----------
        docs : list of str
            Documents to fit the model on
        embeddings : np.ndarray or None, default=None
            Pre-computed document embeddings (optional)
        y : np.ndarray or None, default=None
            Reserved for supervised topic modeling (ignored)
        n_topics : int or None, default=None
            Number of topics (overrides self.n_topics if provided)

        Returns
        -------
        SAETopicModel
            Fitted model instance
        """
        del y
        if n_topics is not None:
            self.n_topics = n_topics

        self.docs_ = list(docs)
        if not self.docs_:
            raise ValueError("docs must be a non-empty list of strings")

        # 1. Load SAE
        self._ensure_sae()

        # 2. Embed documents
        if embeddings is None:
            backend = self._get_embedding_backend()
            self.embeddings_ = backend.embed(self.docs_)
        else:
            self.embeddings_ = np.asarray(embeddings, dtype=np.float32)

        if self.embeddings_.shape[1] != self.sae_input_dim_:
            raise ValueError(
                f"Embedding dim {self.embeddings_.shape[1]} does not match SAE "
                f"input_dim {self.sae_input_dim_}. Re-train the SAE or pass "
                "embeddings with the matching dimension."
            )

        # 3. Extract SAE feature activations (topic atoms)
        from saetopic.sae.activations import extract_activations

        self.feature_activations_ = extract_activations(
            self.embeddings_,
            self.sae_,
            batch_size=self.activation_batch_size,
            device=self.device,
            sparse=(self.theta_mode == "sparse_topk"),
        )

        # 4. Build corpus vocabulary and bag-of-words
        self.vectorizer_ = self.vectorizer_model or self._build_vectorizer()
        if getattr(self.vectorizer_, "vocab_", None) is None:
            self.vectorizer_.fit(self.docs_)
        self.vocab_ = list(self.vectorizer_.vocab_)
        bow = self.vectorizer_.transform(self.docs_)
        self.bow_ = bow

        if len(self.vocab_) == 0:
            raise ValueError(
                "CorpusVectorizer produced an empty vocabulary. Lower min_df or "
                "provide longer documents."
            )

        # 5. Adapt SAE features to corpus vocabulary (feature -> word matrix)
        import torch

        from saetopic.interpretation import CorpusAdapter

        self.adapter_ = CorpusAdapter(
            vocab_size=len(self.vocab_),
            n_features=self.sae_n_features_,
            idf_weighting=self.idf_weighting,
            device=self.device,
            use_sparse_activation=(self.theta_mode == "sparse_topk"),
        )
        self.adapter_.fit(
            embeddings=torch.from_numpy(self.embeddings_),
            bow=bow,
            sae=self.sae_,
            n_epochs=self.corpus_adapter_epochs,
            batch_size=self.corpus_adapter_batch_size,
        )
        self.feature_word_matrix_ = self.adapter_.feature_word_matrix_
        self.idf_ = self.adapter_.idf_

        # 6. Word embeddings for meaningful feature clustering (computed once)
        self.word_embeddings_ = self._compute_word_embeddings(self.vocab_)

        # 7. Merge topic atoms into final topics
        self._fit_topics(self.n_topics)
        return self

    def _build_vectorizer(self):
        from saetopic.vectorizers import CorpusVectorizer

        return CorpusVectorizer(
            vocabulary_size=self.vocabulary_size,
            min_df=self.min_df,
            max_df=self.max_df,
            idf_weighting=self.idf_weighting,
            stop_words=self.stop_words,
        )

    def _fit_topics(self, n_topics: int) -> None:
        """Run the topic-merging step (reusable by fit and retopic)."""
        from saetopic.merging import TopicMerger
        from saetopic.representation import TopicRepresentation, compute_ctfidf

        theta = np.asarray(self.feature_activations_, dtype=np.float32)
        row_sums = theta.sum(axis=1, keepdims=True)
        theta_normalized = np.divide(
            theta,
            np.clip(row_sums, 1e-8, None),
            out=np.zeros_like(theta),
            where=row_sums > 0,
        )
        feature_weights = theta_normalized.mean(axis=0).ravel()

        self.merger_ = TopicMerger(
            n_topics=n_topics,
            method=self.cluster_method,
            random_state=self.random_state,
            sparsity_threshold=self.sparsity_threshold,
            word_embeddings=self.word_embeddings_,
        )
        self.merger_.fit(self.feature_word_matrix_, feature_weights, self.vocab_)

        self.topic_word_matrix_ = self.merger_.topic_word_matrix_
        self.topic_atom_clusters_ = self.merger_.feature_clusters_
        self.document_topic_matrix_ = self.merger_.transform(theta_normalized)
        self.n_topics = self.merger_.n_topics

        # c-TF-IDF topic-word scores (distinctiveness-weighted) for display
        if self.use_ctfidf and self.bow_ is not None:
            self.ctfidf_ = compute_ctfidf(self.document_topic_matrix_, self.bow_)
        else:
            self.ctfidf_ = None

        # Drop clusters to which no document is assigned (count == 0)
        if self.drop_empty_topics:
            self._drop_empty_topics()

        self.topics_ = self.document_topic_matrix_.argmax(axis=1).tolist()

        # Topic embeddings as document-weighted centroids (for find_topics)
        if self.embeddings_ is not None and self.document_topic_matrix_ is not None:
            dtm = self.document_topic_matrix_.astype(np.float32)
            col_sums = dtm.sum(axis=0, keepdims=True)
            col_sums[col_sums == 0] = 1.0
            self.topic_embeddings_ = (dtm.T @ self.embeddings_) / col_sums.T

        # Representation: prefer c-TF-IDF for readable topic words
        display_matrix = self.ctfidf_ if self.ctfidf_ is not None else self.topic_word_matrix_
        self.representation_ = TopicRepresentation(
            display_matrix, self.vocab_, self.document_topic_matrix_
        )

    def _drop_empty_topics(self) -> None:
        """Remove topics with zero assigned documents and renumber the rest."""
        dtm = np.asarray(self.document_topic_matrix_)
        if dtm.ndim != 2 or dtm.shape[1] == 0:
            return
        hard = dtm.argmax(axis=1)
        counts = np.bincount(hard, minlength=dtm.shape[1])
        keep = np.where(counts > 0)[0]
        if len(keep) == dtm.shape[1]:
            return  # nothing empty

        # Slice topic matrices
        self.topic_word_matrix_ = np.asarray(self.topic_word_matrix_)[keep]
        if self.ctfidf_ is not None:
            self.ctfidf_ = np.asarray(self.ctfidf_)[keep]
        # Slice + renormalize document-topic columns
        dtm = dtm[:, keep]
        row_sums = dtm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        self.document_topic_matrix_ = dtm / row_sums
        # Renumber feature cluster assignments
        remap = np.full(counts.shape[0], -1, dtype=int)
        remap[keep] = np.arange(len(keep))
        clusters = np.asarray(self.topic_atom_clusters_)
        self.topic_atom_clusters_ = remap[clusters]
        self.n_topics = int(len(keep))

    def fit_transform(
        self,
        docs: list[str],
        embeddings: np.ndarray | None = None,
        y: np.ndarray | None = None,
        n_topics: int | None = None,
    ) -> tuple[list[int], np.ndarray | None]:
        """
        Fit the model and return topic assignments.

        Parameters
        ----------
        docs : list of str
            Documents to fit the model on
        embeddings : np.ndarray or None, default=None
            Pre-computed document embeddings (optional)
        y : np.ndarray or None, default=None
            Reserved for supervised topic modeling (ignored)
        n_topics : int or None, default=None
            Number of topics (overrides self.n_topics if provided)

        Returns
        -------
        topics : list of int
            Topic assignment for each document
        probs : np.ndarray
            Topic probabilities for each document (n_docs x n_topics)
        """
        self.fit(docs, embeddings=embeddings, y=y, n_topics=n_topics)
        return self.topics_, self.document_topic_matrix_

    def transform(
        self,
        docs: list[str],
        embeddings: np.ndarray | None = None,
    ) -> tuple[list[int], np.ndarray | None]:
        """
        Transform new documents to topic assignments.

        Parameters
        ----------
        docs : list of str
            New documents to transform
        embeddings : np.ndarray or None, default=None
            Pre-computed document embeddings (optional)

        Returns
        -------
        topics : list of int
            Topic assignment for each document
        probs : np.ndarray
            Topic probabilities for each document
        """
        if self.merger_ is None or self.sae_ is None:
            raise RuntimeError("Model must be fitted before transform")

        self._ensure_sae()

        if embeddings is None:
            backend = self._get_embedding_backend()
            embs = backend.embed(list(docs))
        else:
            embs = np.asarray(embeddings, dtype=np.float32)

        from saetopic.sae.activations import extract_activations

        activations = extract_activations(
            embs,
            self.sae_,
            batch_size=self.activation_batch_size,
            device=self.device,
            sparse=(self.theta_mode == "sparse_topk"),
        )
        probs = self.merger_.transform(activations)
        topics = probs.argmax(axis=1).tolist()
        return topics, probs

    def retopic(
        self,
        n_topics: int,
        method: str | None = None,
    ) -> "SAETopicModel":
        """
        Change topic granularity without retraining the SAE or corpus adaptation.

        Only re-runs the clustering step, reusing ``feature_activations_`` and
        ``feature_word_matrix_`` (plus the cached word embeddings).

        Parameters
        ----------
        n_topics : int
            New number of topics
        method : str or None, default=None
            Clustering method ("kmeans", "agglomerative"). None keeps current.

        Returns
        -------
        SAETopicModel
            Self with updated topics
        """
        if self.feature_word_matrix_ is None or self.feature_activations_ is None:
            raise RuntimeError("Model must be fitted before retopic")
        if method is not None:
            self.cluster_method = method
        self._fit_topics(n_topics)
        return self

    def reduce_topics(
        self,
        docs: list[str] | None = None,
        nr_topics: int = 30,
    ) -> "SAETopicModel":
        """Alias for :meth:`retopic`."""
        del docs
        return self.retopic(n_topics=nr_topics)

    # ------------------------------------------------------------------
    # Topic / document inspection
    # ------------------------------------------------------------------
    def get_topic_info(self) -> "pd.DataFrame":
        """
        Get information about each topic.

        Returns
        -------
        pd.DataFrame
            DataFrame with Topic, Count, Name, and Top_Words
        """
        self._require_fitted()
        return self.representation_.get_topic_info()

    def get_topic(
        self,
        topic_id: int,
        top_n: int = 10,
    ) -> list[tuple[str, float]]:
        """
        Get top words for a specific topic.

        Parameters
        ----------
        topic_id : int
            Topic identifier
        top_n : int, default=10
            Number of top words to return

        Returns
        -------
        list of (str, float)
            Top words with their scores
        """
        self._require_fitted()
        return self.representation_.get_topic_words(topic_id, top_n=top_n)

    def get_topics(self) -> dict[int, list[tuple[str, float]]]:
        """
        Get top words for all topics.

        Returns
        -------
        dict
            Mapping from topic_id to list of (word, score) tuples
        """
        self._require_fitted()
        return {t: self.get_topic(t) for t in range(self.n_topics)}

    def get_document_info(
        self,
        docs: list[str] | None = None,
    ) -> "pd.DataFrame":
        """
        Get information about each document.

        Parameters
        ----------
        docs : list of str or None, default=None
            Documents to analyze (uses fitted docs if None)

        Returns
        -------
        pd.DataFrame
            DataFrame with Document, Topic, and top topic probability
        """
        import pandas as pd

        self._require_fitted()
        docs = docs if docs is not None else self.docs_
        dtm = np.asarray(self.document_topic_matrix_)
        top_topic = dtm.argmax(axis=1)
        top_prob = dtm.max(axis=1)
        return pd.DataFrame(
            {
                "Document": docs,
                "Topic": top_topic.tolist(),
                "Probability": top_prob.tolist(),
            }
        )

    def get_representative_docs(
        self,
        topic_id: int | None = None,
        n: int = 5,
    ) -> list[str]:
        """
        Get representative documents for a topic (highest topic probability).

        Parameters
        ----------
        topic_id : int or None, default=None
            Topic identifier (None returns representatives for topic 0)
        n : int, default=5
            Number of representative documents

        Returns
        -------
        list of str
            Representative document texts
        """
        self._require_fitted()
        if self.docs_ is None:
            raise RuntimeError("No fitted documents available")
        topic_id = 0 if topic_id is None else topic_id
        probs = np.asarray(self.document_topic_matrix_)[:, topic_id]
        top_idx = np.argsort(probs)[-n:][::-1]
        return [self.docs_[i] for i in top_idx]

    def find_topics(
        self,
        query: str,
        top_n: int = 5,
    ) -> list[tuple[int, float]]:
        """
        Find topics most similar to a search query.

        Parameters
        ----------
        query : str
            Search query text
        top_n : int, default=5
            Number of topics to return

        Returns
        -------
        list of (int, float)
            Topic IDs with cosine similarity scores
        """
        self._require_fitted()
        backend = self._get_embedding_backend()
        query_emb = backend.embed([query])[0]
        topic_emb = np.asarray(self.topic_embeddings_)
        # cosine similarity
        q = query_emb / (np.linalg.norm(query_emb) + 1e-12)
        t = topic_emb / (np.linalg.norm(topic_emb, axis=1, keepdims=True) + 1e-12)
        sims = t @ q
        top_idx = np.argsort(sims)[-top_n:][::-1]
        return [(int(i), float(sims[i])) for i in top_idx]

    def generate_topic_labels(
        self,
        method: str = "words",
        llm: Any = None,
    ) -> dict[int, str]:
        """Generate human-readable topic labels and store them on the model."""
        self._require_fitted()
        labels = self.representation_.generate_topic_labels(method=method, llm=llm)
        self.representation_.topic_labels_ = labels
        return labels

    def set_topic_labels(self, labels: dict[int, str]) -> None:
        """Set custom labels for topics."""
        self._require_fitted()
        self.representation_.topic_labels_ = dict(labels)

    def _require_fitted(self) -> None:
        if self.representation_ is None:
            raise RuntimeError("Model must be fitted before this operation")

    # ------------------------------------------------------------------
    # Not yet implemented (Week 4 / Week 5 scope)
    # ------------------------------------------------------------------
    def visualize_topics(self):
        raise NotImplementedError("visualize_topics is not implemented yet")

    def visualize_documents(self, docs: list[str] | None = None):
        raise NotImplementedError("visualize_documents is not implemented yet")

    def visualize_hierarchy(self):
        raise NotImplementedError("visualize_hierarchy is not implemented yet")

    def visualize_atoms(self, topic_id: int):
        raise NotImplementedError("visualize_atoms is not implemented yet")

    def evaluate(
        self,
        metrics: tuple[str, ...] = ("diversity", "coherence", "stability"),
    ) -> dict[str, float]:
        raise NotImplementedError("evaluate is not implemented yet")

    def save(self, path: str, serialization: str = "safetensors") -> None:
        raise NotImplementedError("save is not implemented yet")

    @classmethod
    def load(cls, path: str) -> "SAETopicModel":
        raise NotImplementedError("load is not implemented yet")
