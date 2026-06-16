"""
Topic atom merging into final topics.

This module clusters SAE topic atoms into the desired number of topics,
following the SAE-TM framework.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch
from scipy.sparse import csr_matrix
from sklearn.cluster import AgglomerativeClustering, KMeans

if TYPE_CHECKING:
    import numpy.typing as npt

logger = logging.getLogger(__name__)

_GENSIM_MODEL_CACHE: dict[str, Any] = {}


def _resolve_local_gensim_path(embedding_model: str) -> Path | None:
    """Return a local gensim-data vector file path when it exists."""
    explicit_path = Path(embedding_model).expanduser()
    if explicit_path.exists():
        return explicit_path

    base_dir = Path(os.environ.get("GENSIM_DATA_DIR", "~/gensim-data")).expanduser()
    model_dir = base_dir / embedding_model
    candidates = [
        model_dir / f"{embedding_model}.gz",
        model_dir / embedding_model,
        model_dir / f"{embedding_model}.bin",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_gensim_model(embedding_model: str):
    """Load a gensim keyed-vector model, preferring local cache over downloader metadata."""
    if embedding_model in _GENSIM_MODEL_CACHE:
        return _GENSIM_MODEL_CACHE[embedding_model]

    local_path = _resolve_local_gensim_path(embedding_model)
    if local_path is not None:
        from gensim.models import KeyedVectors

        logger.info("Loading local gensim vectors from %s", local_path)
        model = KeyedVectors.load_word2vec_format(str(local_path), binary=True)
    else:
        import gensim.downloader as api

        logger.info(f"Loading gensim model via downloader: {embedding_model}")
        model = api.load(embedding_model)

    _GENSIM_MODEL_CACHE[embedding_model] = model
    return model


def preload_word_embedding_model(embedding_model: str | None) -> None:
    """Load and cache a gensim word embedding model once for a long-running process."""
    if embedding_model is None:
        return
    _load_gensim_model(embedding_model)


def sparsify_and_renormalize(
    input_tensor: npt.ArrayLike,
    tau: float = 0.9,
    chunk_size: int = 2048,
) -> np.ndarray:
    """
    Transform a matrix by keeping top-n entries per row and renormalizing.

    For each row, keeps the minimum number of largest entries whose
    cumulative sum exceeds tau, then renormalizes to sum to 1.

    Args:
        input_tensor: K x V array with non-negative entries summing to 1 per row
        tau: Cumulative sum threshold (default: 0.9)

    Returns:
        Transformed K x V array with sparse, renormalized rows
    """
    was_numpy = isinstance(input_tensor, np.ndarray)
    arr = torch.from_numpy(input_tensor).float() if was_numpy else input_tensor

    if arr.dim() != 2:
        raise ValueError(f"Input must be 2D, got {arr.dim()}D")

    n_rows, vocab_size = arr.shape
    device = arr.device

    # Process in row-chunks so peak memory is bounded by chunk_size x V,
    # not K x V (a full torch.sort allocates a same-sized int64 index array).
    result = torch.zeros_like(arr)
    for start in range(0, n_rows, chunk_size):
        end = min(start + chunk_size, n_rows)
        k = end - start
        block = arr[start:end]

        sorted_values, sorted_indices = torch.sort(block, dim=-1, descending=True)
        cumulative_sums = torch.cumsum(sorted_values, dim=-1)
        n_elements = torch.argmax((cumulative_sums > tau).int(), dim=-1) + 1
        never_exceeds = (cumulative_sums > tau).sum(dim=-1) == 0
        n_elements[never_exceeds] = vocab_size

        arange = torch.arange(vocab_size, device=device).expand(k, -1)
        mask_sorted = arange < n_elements.unsqueeze(-1)
        final_mask = torch.zeros_like(block, dtype=torch.bool)
        final_mask.scatter_(dim=1, index=sorted_indices, src=mask_sorted)

        block_result = block * final_mask
        row_sums = block_result.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        result[start:end] = block_result / row_sums

    return result.numpy() if was_numpy else result


def get_word_embeddings(
    vocab: list[str],
    embedding_model: str | None = None,
    embedding_dim: int = 300,
    device: str = "cpu",
    allow_random_fallback: bool = True,
) -> np.ndarray:
    """
    Get word embeddings for vocabulary.

    Parameters
    ----------
    vocab : list of str
        List of words in vocabulary
    embedding_model : str, optional
        Name of embedding model to load from gensim
        (e.g., "word2vec-google-news-300"). If None, uses random embeddings.
    embedding_dim : int, default=300
        Dimension of random embeddings if model not specified
    device : str, default="cpu"
        Device for computation
    allow_random_fallback : bool, default=True
        If False, raise when ``embedding_model`` cannot be loaded instead of
        silently using random embeddings.

    Returns
    -------
    np.ndarray
        Word embeddings (vocab_size x embedding_dim)
    """
    logger.info(f"Getting word embeddings for {len(vocab)} words")

    if embedding_model is not None:
        try:
            w2v = _load_gensim_model(embedding_model)

            embeddings = []
            missing = 0
            mean_vector = None

            for word in vocab:
                try:
                    embeddings.append(w2v[word])
                except KeyError:
                    # Use mean embedding for missing words
                    if mean_vector is None:
                        mean_vector = w2v.get_mean_vector(w2v.key_to_index.keys())
                    embeddings.append(mean_vector)
                    missing += 1

            logger.info(f"Loaded embeddings: {missing} words missing from model")

            return np.stack(embeddings).astype(np.float32, copy=False)

        except Exception as e:
            if not allow_random_fallback:
                raise RuntimeError(
                    f"Failed to load gensim embedding model {embedding_model!r}"
                ) from e
            logger.warning(f"Failed to load gensim model: {e}. Using random embeddings.")

    # Fall back to random embeddings
    logger.info(f"Using random embeddings with dim={embedding_dim}")
    np.random.seed(42)
    return np.random.randn(len(vocab), embedding_dim).astype(np.float32) * 0.01


class TopicMerger:
    """
    Clusters SAE topic atoms into final topics.

    This component implements the SAE-TM topic merging algorithm:
    1. Sparsify the feature-to-word matrix
    2. Compute feature embeddings via weighted word embeddings
    3. Cluster feature embeddings using KMeans or Agglomerative clustering
    4. Aggregate features within each cluster

    Parameters
    ----------
    n_topics : int
        Target number of topics
    method : str, default="kmeans"
        Clustering method ("kmeans" or "agglomerative")
    random_state : int, default=42
        Random seed for reproducibility
    sparsity_threshold : float, default=0.9
        Tau threshold for sparsifying feature-to-word matrix
    embedding_model : str, optional
        Pretrained word embedding model for semantic clustering
    embedding_dim : int, default=300
        Dimension for random word embeddings if model not specified
    word_embeddings : np.ndarray or None, default=None
        Precomputed word embeddings for the vocabulary (vocab_size x dim). When
        provided, these are used directly instead of loading ``embedding_model``
        (or falling back to random embeddings), giving meaningful semantic
        clustering. Recommended: reuse the document embedding model.
    allow_random_word_embeddings : bool, default=True
        Whether to fall back to random word embeddings when ``embedding_model``
        cannot be loaded. Set False for paper-aligned runs.
    """

    def __init__(
        self,
        n_topics: int,
        method: Literal["kmeans", "agglomerative"] = "kmeans",
        random_state: int = 42,
        sparsity_threshold: float = 0.9,
        embedding_model: str | None = None,
        embedding_dim: int = 300,
        word_embeddings: np.ndarray | None = None,
        allow_random_word_embeddings: bool = True,
    ):
        self.n_topics = n_topics
        self.method = method
        self.random_state = random_state
        self.sparsity_threshold = sparsity_threshold
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self.word_embeddings = word_embeddings
        self.allow_random_word_embeddings = allow_random_word_embeddings

        # Cluster results
        self.feature_clusters_: np.ndarray | None = None  # K, cluster assignment per feature
        self.topic_word_matrix_: np.ndarray | None = None  # n_topics x V
        self.cluster_info_: list[dict] | None = None  # Per-cluster metadata

        # Fitted state
        self._is_fitted = False

    def _compute_feature_embeddings(
        self,
        feature_word_matrix: np.ndarray,
        vocab: list[str],
    ) -> np.ndarray:
        """
        Compute feature embeddings via sparsified word weighting.

        Args:
            feature_word_matrix: K x V matrix
            vocab: Vocabulary list

        Returns:
            Feature embeddings: K x embedding_dim
        """
        logger.info(
            f"Computing feature embeddings with sparsity threshold={self.sparsity_threshold}"
        )

        # Get word embeddings first (precomputed, model, or random fallback)
        if self.word_embeddings is not None:
            word_embeddings = np.asarray(self.word_embeddings, dtype=np.float32)
            if word_embeddings.shape[0] != len(vocab):
                raise ValueError(
                    "word_embeddings has "
                    f"{word_embeddings.shape[0]} rows but vocab has {len(vocab)} "
                    "entries"
                )
            logger.info(
                "Using precomputed word embeddings: shape=%s", word_embeddings.shape
            )
        else:
            word_embeddings = get_word_embeddings(
                vocab=vocab,
                embedding_model=self.embedding_model,
                embedding_dim=self.embedding_dim,
                allow_random_fallback=self.allow_random_word_embeddings,
            )

        # Fused sparsify + matmul in feature-row chunks: avoids materializing
        # the full sparsified B (K x V) alongside the input feature_word_matrix.
        n_features = feature_word_matrix.shape[0]
        dim = word_embeddings.shape[1]
        feature_embeddings = np.zeros((n_features, dim), dtype=np.float32)
        chunk = 2048
        for start in range(0, n_features, chunk):
            end = min(start + chunk, n_features)
            block_sparse = sparsify_and_renormalize(
                feature_word_matrix[start:end], tau=self.sparsity_threshold
            )
            feature_embeddings[start:end] = block_sparse @ word_embeddings

        logger.info(f"Feature embeddings shape: {feature_embeddings.shape}")

        return feature_embeddings

    def _cluster_features(
        self,
        feature_embeddings: np.ndarray,
        feature_weights: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Cluster features based on their embeddings.

        Args:
            feature_embeddings: K x embedding_dim
            feature_weights: Optional weights for KMeans (K,)

        Returns:
            Cluster labels for each feature (K,)
        """
        logger.info(
            f"Clustering {len(feature_embeddings)} features into {self.n_topics} topics "
            f"using {self.method}"
        )

        if self.method == "kmeans":
            clusterer = KMeans(
                n_clusters=self.n_topics,
                random_state=self.random_state,
            )
            if feature_weights is not None:
                labels = clusterer.fit_predict(
                    feature_embeddings,
                    sample_weight=feature_weights,
                )
            else:
                labels = clusterer.fit_predict(feature_embeddings)
        elif self.method == "agglomerative":
            clusterer = AgglomerativeClustering(
                n_clusters=self.n_topics,
            )
            # AgglomerativeClustering doesn't support sample_weight
            labels = clusterer.fit_predict(feature_embeddings)
        else:
            raise ValueError(f"Unknown clustering method: {self.method}")

        logger.info("Clustering complete. Feature distribution:")
        unique, counts = np.unique(labels, return_counts=True)
        for u, c in zip(unique, counts):
            logger.info(f"  Cluster {u}: {c} features")

        return labels

    def _aggregate_clusters(
        self,
        feature_word_matrix: np.ndarray,
        feature_weights: np.ndarray,
        feature_clusters: np.ndarray,
        vocab: list[str],
    ) -> tuple[np.ndarray, list[dict]]:
        """
        Aggregate features within clusters to produce final topics.

        Args:
            feature_word_matrix: K x V
            feature_weights: K,
            feature_clusters: K, cluster assignment
            vocab: Vocabulary list

        Returns:
            (topic_word_matrix, cluster_info)
        """
        logger.info("Aggregating features into topics")

        n_topics = self.n_topics
        vocab_size = len(vocab)

        topic_word_matrix = np.zeros((n_topics, vocab_size), dtype=np.float32)
        cluster_info = []

        for cluster_id in range(n_topics):
            mask = feature_clusters == cluster_id
            cluster_features = np.where(mask)[0]

            if len(cluster_features) == 0:
                logger.warning(f"Cluster {cluster_id} has no features")
                # Use uniform distribution as fallback
                topic_word_matrix[cluster_id] = np.ones(vocab_size) / vocab_size
                cluster_info.append({
                    "cluster_id": cluster_id,
                    "n_features": 0,
                    "total_weight": 0.0,
                    "top_words": " ".join(vocab[:20]),
                })
                continue

            # Get feature-to-word distributions for this cluster
            cluster_b = feature_word_matrix[cluster_features]
            cluster_theta = feature_weights[cluster_features]

            # Weighted average: theta-weighted sum of word distributions
            # This gives the cluster's topic-word distribution
            weighted_b = cluster_b * cluster_theta[:, np.newaxis]
            cluster_weight = cluster_theta.sum()
            if cluster_weight <= 0:
                logger.warning(f"Cluster {cluster_id} has non-positive total weight")
                topic_dist = np.ones(vocab_size, dtype=np.float32) / vocab_size
            else:
                topic_dist = weighted_b.sum(axis=0) / cluster_weight

            topic_word_matrix[cluster_id] = topic_dist

            # Get top words
            top_indices = np.argsort(topic_dist)[-50:][::-1]
            top_words = " ".join([vocab[i] for i in top_indices])

            cluster_info.append({
                "cluster_id": cluster_id,
                "n_features": len(cluster_features),
                "total_weight": float(cluster_theta.sum()),
                "top_words": top_words,
            })

        return topic_word_matrix, cluster_info

    def fit(
        self,
        feature_word_matrix: np.ndarray,
        feature_weights: np.ndarray,
        vocab: list[str],
    ) -> "TopicMerger":
        """
        Cluster features into topics.

        Parameters
        ----------
        feature_word_matrix : np.ndarray
            Feature-to-word matrix (n_features x vocab_size)
        feature_weights : np.ndarray
            Average feature activation across corpus (n_features,)
        vocab : list of str
            Vocabulary list

        Returns
        -------
        TopicMerger
            Fitted merger instance
        """
        logger.info(
            f"Fitting TopicMerger: n_features={feature_word_matrix.shape[0]}, "
            f"vocab_size={len(vocab)}, n_topics={self.n_topics}"
        )

        # Validate inputs
        assert feature_word_matrix.shape[1] == len(vocab), \
            "Vocab size mismatch with feature_word_matrix"
        assert feature_word_matrix.shape[0] == len(feature_weights), \
            "Feature weights length mismatch"

        # Filter out unused features (zero activation)
        valid_mask = feature_weights > 0
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            raise ValueError("No features with positive activation found")

        if len(valid_indices) < self.n_topics:
            logger.warning(
                f"Only {len(valid_indices)} valid features but n_topics={self.n_topics}. "
                f"Using {len(valid_indices)} topics."
            )
            self.n_topics = len(valid_indices)

        # Avoid a full-matrix copy when all features are valid (the dense-θ case),
        # which would double resident memory (e.g. ~1.2 GB for 24k x 12k).
        if valid_mask.all():
            feature_word_valid = feature_word_matrix
            feature_weights_valid = feature_weights
        else:
            feature_word_valid = feature_word_matrix[valid_indices]
            feature_weights_valid = feature_weights[valid_indices]

        # Step 1: Compute feature embeddings
        feature_embeddings = self._compute_feature_embeddings(
            feature_word_valid,
            vocab,
        )

        # Step 2: Cluster features
        cluster_labels = self._cluster_features(
            feature_embeddings,
            feature_weights_valid,
        )

        # Map back to full feature space
        full_cluster_labels = np.full(len(feature_weights), -1, dtype=int)
        full_cluster_labels[valid_indices] = cluster_labels

        # Step 3: Aggregate into topics
        topic_word_matrix, cluster_info = self._aggregate_clusters(
            feature_word_matrix,
            feature_weights,
            full_cluster_labels,
            vocab,
        )

        self.feature_clusters_ = full_cluster_labels
        self.topic_word_matrix_ = topic_word_matrix
        self.cluster_info_ = cluster_info
        self._is_fitted = True

        logger.info("TopicMerger fitting complete")

        return self

    def transform(
        self,
        feature_activations: csr_matrix | np.ndarray,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Aggregate feature activations to topic probabilities.

        Parameters
        ----------
        feature_activations : scipy.sparse.csr_matrix or np.ndarray
            Document-feature activations (n_docs x n_features)
        normalize : bool, default=True
            Whether to row-normalize the aggregated document-topic matrix.
            SAE-TM's exported ``theta_topic_csr.npz`` leaves this unnormalized
            after multiplying normalized theta by the feature-cluster map.

        Returns
        -------
        np.ndarray
            Document-topic probabilities (n_docs x n_topics)
        """
        if not self._is_fitted:
            raise RuntimeError("TopicMerger must be fitted before transform")

        if self.feature_clusters_ is None:
            raise RuntimeError("Cluster assignments not available")

        # Convert sparse to dense if needed
        if isinstance(feature_activations, csr_matrix):
            feature_activations = feature_activations.toarray()

        n_docs, n_features = feature_activations.shape

        # Handle case where n_features differs from fitted size
        if n_features != len(self.feature_clusters_):
            raise ValueError(
                f"Feature dimension mismatch: expected {len(self.feature_clusters_)}, "
                f"got {n_features}"
            )

        # Aggregate: for each document, sum activations within each cluster
        n_topics = self.n_topics
        topic_activations = np.zeros((n_docs, n_topics), dtype=np.float32)

        for topic_id in range(n_topics):
            mask = self.feature_clusters_ == topic_id
            if mask.any():
                topic_activations[:, topic_id] = feature_activations[:, mask].sum(axis=1)

        if not normalize:
            return topic_activations

        # Normalize to get probabilities
        row_sums = topic_activations.sum(axis=1, keepdims=True)

        topic_probs = np.divide(
            topic_activations,
            row_sums,
            out=np.zeros_like(topic_activations),
            where=row_sums > 0,
        )

        return topic_probs

    def fit_transform(
        self,
        feature_word_matrix: np.ndarray,
        feature_weights: np.ndarray,
        feature_activations: csr_matrix | np.ndarray,
        vocab: list[str],
    ) -> np.ndarray:
        """
        Fit and transform in one step.

        Parameters
        ----------
        feature_word_matrix : np.ndarray
            Feature-to-word matrix (n_features x vocab_size)
        feature_weights : np.ndarray
            Average feature activation across corpus (n_features,)
        feature_activations : scipy.sparse.csr_matrix or np.ndarray
            Document-feature activations (n_docs x n_features)
        vocab : list of str
            Vocabulary list

        Returns
        -------
        np.ndarray
            Document-topic probabilities (n_docs x n_topics)
        """
        self.fit(
            feature_word_matrix=feature_word_matrix,
            feature_weights=feature_weights,
            vocab=vocab,
        )
        return self.transform(feature_activations)

    def get_topic_info(self) -> list[dict]:
        """
        Get information about each learned topic.

        Returns
        -------
        list of dict
            Per-topic information including top words
        """
        if not self._is_fitted:
            raise RuntimeError("TopicMerger must be fitted first")

        return self.cluster_info_

    def get_topic_words(
        self,
        topic_id: int,
        vocab: list[str],
        top_n: int = 20,
    ) -> list[tuple[str, float]]:
        """
        Get top words for a specific topic.

        Parameters
        ----------
        topic_id : int
            Index of the topic
        vocab : list of str
            Vocabulary list
        top_n : int, default=20
            Number of top words to return

        Returns
        -------
        list of (str, float)
            Top words and their probabilities
        """
        if not self._is_fitted:
            raise RuntimeError("TopicMerger must be fitted first")

        if topic_id < 0 or topic_id >= self.n_topics:
            raise ValueError(f"Invalid topic_id: {topic_id}")

        topic_probs = self.topic_word_matrix_[topic_id]
        top_indices = np.argsort(topic_probs)[-top_n:][::-1]

        return [(vocab[i], float(topic_probs[i])) for i in top_indices]

    def get_feature_clusters(self) -> np.ndarray:
        """Return cluster assignment for each feature."""
        if not self._is_fitted:
            raise RuntimeError("TopicMerger must be fitted first")
        return self.feature_clusters_

    def get_topic_word_matrix(self) -> np.ndarray:
        """Return the learned topic-to-word matrix."""
        if not self._is_fitted:
            raise RuntimeError("TopicMerger must be fitted first")
        return self.topic_word_matrix_
