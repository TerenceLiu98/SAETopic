"""
Topic atom merging into final topics.

This module clusters SAE topic atoms into the desired number of topics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import numpy as np


class TopicMerger:
    """
    Clusters SAE topic atoms into final topics.

    Parameters
    ----------
    n_topics : int
        Target number of topics
    method : str, default="kmeans"
        Clustering method ("kmeans", "agglomerative", "hdbscan")
    random_state : int, default=42
        Random seed for reproducibility
    """

    def __init__(
        self,
        n_topics: int,
        method: Literal["kmeans", "agglomerative", "hdbscan"] = "kmeans",
        random_state: int = 42,
    ):
        self.n_topics = n_topics
        self.method = method
        self.random_state = random_state

        # Feature-to-topic cluster assignment
        self.feature_clusters_: np.ndarray | None = None

    def fit(
        self,
        feature_word_matrix: np.ndarray,
    ) -> "TopicMerger":
        """
        Cluster features into topics.

        Parameters
        ----------
        feature_word_matrix : np.ndarray
            Feature-to-word matrix (n_features x vocab_size)

        Returns
        -------
        TopicMerger
            Fitted merger instance
        """
        # TODO: Implement feature clustering (Week 3)
        raise NotImplementedError("TopicMerger.fit will be implemented in Week 3")

    def transform(
        self,
        feature_activations: np.ndarray,
    ) -> np.ndarray:
        """
        Aggregate feature activations to topic probabilities.

        Parameters
        ----------
        feature_activations : np.ndarray
            Document-feature activations (n_docs x n_features)

        Returns
        -------
        np.ndarray
            Document-topic probabilities (n_docs x n_topics)
        """
        # TODO: Implement activation aggregation (Week 3)
        raise NotImplementedError("TopicMerger.transform will be implemented in Week 3")

    def fit_transform(
        self,
        feature_word_matrix: np.ndarray,
        feature_activations: np.ndarray,
    ) -> np.ndarray:
        """
        Fit and transform in one step.

        Parameters
        ----------
        feature_word_matrix : np.ndarray
            Feature-to-word matrix (n_features x vocab_size)
        feature_activations : np.ndarray
            Document-feature activations (n_docs x n_features)

        Returns
        -------
        np.ndarray
            Document-topic probabilities (n_docs x n_topics)
        """
        return self.fit(feature_word_matrix).transform(feature_activations)
