"""
Main SAETopicModel class with BERTopic-like API.

This module provides the primary interface for topic modeling using
sparse autoencoder topic atoms.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


class SAETopicModel:
    """
    SAETopic: BERTopic-style topic modeling with sparse autoencoder topic atoms.

    The core differentiator is the ability to change topic granularity via
    `retopic()` without retraining the SAE or recomputing corpus adaptation.

    Parameters
    ----------
    embedding_model : str or callable, default="jinaai/jina-embeddings-v5-text-small"
        Model to use for embedding documents. Can be:
        - A Hugging Face model ID (for SentenceTransformers/Jina)
        - A callable that takes a list of strings and returns embeddings
    embedding_task : str, default="clustering"
        Task type for Jina embeddings (e.g., "clustering", "retrieval")
    sae_model : str, default="saetopic/jina-v5-sae-small"
        Pretrained SAE checkpoint ID from Hugging Face Hub
    n_topics : int, default=50
        Initial number of topics to generate
    top_k_features : int, default=32
        Number of top-k features to activate in SAE
    min_topic_size : int or None, default=None
        Minimum number of documents per topic
    vectorizer_model : Any, default=None
        Custom vectorizer for bag-of-words construction
    idf_weighting : bool, default=True
        Whether to use IDF weighting in corpus adaptation
    device : str, default="auto"
        Device for computation ("auto", "cpu", "cuda", "mps")
    random_state : int, default=42
        Random seed for reproducibility
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

        # Internal attributes (set during fit)
        self.embeddings_: np.ndarray | None = None
        self.feature_activations_: np.ndarray | None = None
        self.feature_word_matrix_: np.ndarray | None = None
        self.topic_atom_clusters_: np.ndarray | None = None
        self.topic_word_matrix_: np.ndarray | None = None
        self.document_topic_matrix_: np.ndarray | None = None
        self.topic_embeddings_: np.ndarray | None = None
        self.vocab_: list[str] | None = None
        self.idf_: np.ndarray | None = None

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs) -> "SAETopicModel":
        """
        Load a pretrained SAETopic model from Hugging Face Hub.

        Parameters
        ----------
        model_id : str
            Hugging Face model ID (e.g., "saetopic/jina-v5-sae-small")
        **kwargs
            Additional arguments to override default config

        Returns
        -------
        SAETopicModel
            Initialized model with pretrained SAE checkpoint
        """
        # TODO: Implement checkpoint loading from HF Hub
        raise NotImplementedError("from_pretrained will be implemented in Week 3")

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
            Optional labels for supervised topic modeling
        n_topics : int or None, default=None
            Number of topics (overrides self.n_topics if provided)

        Returns
        -------
        SAETopicModel
            Fitted model instance
        """
        # TODO: Implement full fit pipeline (Week 3)
        raise NotImplementedError("fit will be implemented in Week 3")

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
            Optional labels for supervised topic modeling
        n_topics : int or None, default=None
            Number of topics (overrides self.n_topics if provided)

        Returns
        -------
        topics : list of int
            Topic assignment for each document
        probs : np.ndarray or None
            Topic probabilities for each document
        """
        # TODO: Implement full fit_transform pipeline (Week 3)
        raise NotImplementedError("fit_transform will be implemented in Week 3")

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
        probs : np.ndarray or None
            Topic probabilities for each document
        """
        # TODO: Implement transform (Week 4)
        raise NotImplementedError("transform will be implemented in Week 4")

    def retopic(
        self,
        n_topics: int,
        method: str = "kmeans",
    ) -> "SAETopicModel":
        """
        Change topic granularity without retraining SAE or corpus adaptation.

        This is the key differentiator: only re-runs the clustering step,
        reusing existing feature_activations_ and feature_word_matrix_.

        Parameters
        ----------
        n_topics : int
            New number of topics
        method : str, default="kmeans"
            Clustering method ("kmeans", "agglomerative", "hdbscan")

        Returns
        -------
        SAETopicModel
            Self with updated topics
        """
        # TODO: Implement retopic (Week 3)
        raise NotImplementedError("retopic will be implemented in Week 3")

    def reduce_topics(
        self,
        docs: list[str] | None = None,
        nr_topics: int = 30,
    ) -> "SAETopicModel":
        """
        BERTopic-compatible alias for retopic().

        Parameters
        ----------
        docs : list of str or None, default=None
            Documents (for compatibility, not used in SAETopic)
        nr_topics : int, default=30
            Target number of topics

        Returns
        -------
        SAETopicModel
            Self with updated topics
        """
        return self.retopic(n_topics=nr_topics)

    def get_topic_info(self) -> pd.DataFrame:
        """
        Get information about each topic.

        Returns
        -------
        pd.DataFrame
            DataFrame with topic_id, count, name, and top words
        """
        # TODO: Implement get_topic_info (Week 3)
        raise NotImplementedError("get_topic_info will be implemented in Week 3")

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
        # TODO: Implement get_topic (Week 3)
        raise NotImplementedError("get_topic will be implemented in Week 3")

    def get_topics(self) -> dict[int, list[tuple[str, float]]]:
        """
        Get top words for all topics.

        Returns
        -------
        dict
            Mapping from topic_id to list of (word, score) tuples
        """
        # TODO: Implement get_topics (Week 3)
        raise NotImplementedError("get_topics will be implemented in Week 3")

    def get_document_info(
        self,
        docs: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Get information about each document.

        Parameters
        ----------
        docs : list of str or None, default=None
            Documents to analyze (uses fitted docs if None)

        Returns
        -------
        pd.DataFrame
            DataFrame with document_id, topic, probability, and representative words
        """
        # TODO: Implement get_document_info (Week 4)
        raise NotImplementedError("get_document_info will be implemented in Week 4")

    def get_representative_docs(
        self,
        topic_id: int | None = None,
        n: int = 5,
    ) -> list[str]:
        """
        Get representative documents for a topic.

        Parameters
        ----------
        topic_id : int or None, default=None
            Topic identifier (None returns for all topics)
        n : int, default=5
            Number of representative documents per topic

        Returns
        -------
        list of str
            Representative document texts
        """
        # TODO: Implement get_representative_docs (Week 4)
        raise NotImplementedError("get_representative_docs will be implemented in Week 4")

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
            Topic IDs with similarity scores
        """
        # TODO: Implement find_topics (Week 4)
        raise NotImplementedError("find_topics will be implemented in Week 4")

    def generate_topic_labels(
        self,
        method: str = "words",
        llm: Any = None,
    ) -> dict[int, str]:
        """
        Generate human-readable topic labels.

        Parameters
        ----------
        method : str, default="words"
            Labeling method ("words", "llm")
        llm : Any, default=None
            LLM for label generation (if method="llm")

        Returns
        -------
        dict
            Mapping from topic_id to label string
        """
        # TODO: Implement generate_topic_labels (Week 5)
        raise NotImplementedError("generate_topic_labels will be implemented in Week 5")

    def set_topic_labels(self, labels: dict[int, str]) -> None:
        """
        Set custom labels for topics.

        Parameters
        ----------
        labels : dict
            Mapping from topic_id to label string
        """
        # TODO: Implement set_topic_labels (Week 5)
        raise NotImplementedError("set_topic_labels will be implemented in Week 5")

    def visualize_topics(self):
        """
        Visualize topics in 2D space.

        Returns
        -------
        plotly.graph_objects.Figure
            Interactive 2D visualization
        """
        # TODO: Implement visualize_topics (Week 4)
        raise NotImplementedError("visualize_topics will be implemented in Week 4")

    def visualize_documents(
        self,
        docs: list[str] | None = None,
    ):
        """
        Visualize documents in 2D space colored by topic.

        Parameters
        ----------
        docs : list of str or None, default=None
            Documents to visualize

        Returns
        -------
        plotly.graph_objects.Figure
            Interactive 2D visualization
        """
        # TODO: Implement visualize_documents (Week 4)
        raise NotImplementedError("visualize_documents will be implemented in Week 4")

    def visualize_hierarchy(self):
        """
        Visualize topic merge hierarchy.

        Returns
        -------
        plotly.graph_objects.Figure
            Interactive hierarchical visualization
        """
        # TODO: Implement visualize_hierarchy (Week 4)
        raise NotImplementedError("visualize_hierarchy will be implemented in Week 4")

    def visualize_atoms(
        self,
        topic_id: int,
    ):
        """
        Visualize SAE topic atoms within a merged topic.

        This is an advanced visualization showing the internal structure.

        Parameters
        ----------
        topic_id : int
            Topic identifier

        Returns
        -------
        plotly.graph_objects.Figure
            Interactive atom-level visualization
        """
        # TODO: Implement visualize_atoms (Week 4)
        raise NotImplementedError("visualize_atoms will be implemented in Week 4")

    def evaluate(
        self,
        metrics: tuple[str, ...] = ("diversity", "coherence", "stability"),
    ) -> dict[str, float]:
        """
        Evaluate topic model quality.

        Parameters
        ----------
        metrics : tuple of str, default=("diversity", "coherence", "stability")
            Metrics to compute

        Returns
        -------
        dict
            Mapping from metric name to score
        """
        # TODO: Implement evaluate (Week 5)
        raise NotImplementedError("evaluate will be implemented in Week 5")

    def save(
        self,
        path: str,
        serialization: str = "safetensors",
    ) -> None:
        """
        Save fitted model to disk.

        Parameters
        ----------
        path : str
            Path to save directory
        serialization : str, default="safetensors"
            Serialization format ("safetensors", "pickle")
        """
        # TODO: Implement save (Week 4)
        raise NotImplementedError("save will be implemented in Week 4")

    @classmethod
    def load(cls, path: str) -> "SAETopicModel":
        """
        Load a fitted model from disk.

        Parameters
        ----------
        path : str
            Path to saved model directory

        Returns
        -------
        SAETopicModel
            Loaded model instance
        """
        # TODO: Implement load (Week 4)
        raise NotImplementedError("load will be implemented in Week 4")
