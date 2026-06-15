"""
Configuration management for SAETopic.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SAETopicConfig:
    """
    Configuration for SAETopic model.

    Attributes
    ----------
    embedding_model : str
        Hugging Face model ID for embeddings
    embedding_task : str
        Task type for embeddings
    sae_model : str
        Hugging Face model ID for SAE checkpoint
    n_topics : int
        Initial number of topics
    top_k_features : int
        Number of top-k features in SAE
    min_topic_size : int or None
        Minimum documents per topic
    idf_weighting : bool
        Whether to use IDF weighting
    device : str
        Device for computation
    random_state : int
        Random seed
    """

    embedding_model: str = "jinaai/jina-embeddings-v5-text-small"
    embedding_task: str = "clustering"
    sae_model: str = "saetopic/jina-v5-sae-small"
    n_topics: int = 50
    top_k_features: int = 32
    min_topic_size: int | None = None
    idf_weighting: bool = False
    device: str = "auto"
    random_state: int = 42


@dataclass
class SAETrainingConfig:
    """
    Configuration for training a new SAE.

    Attributes
    ----------
    input_dim : int
        Input embedding dimension
    expansion_factor : int
        Ratio of features to input dimension
    top_k : int
        Number of features activated per input
    learning_rate : float
        Learning rate for training
    batch_size : int
        Training batch size
    n_epochs : int
        Number of training epochs
    """

    input_dim: int = 1024
    expansion_factor: int = 32
    top_k: int = 32
    learning_rate: float = 1e-4
    batch_size: int = 256
    n_epochs: int = 100


@dataclass
class HFHubConfig:
    """
    Configuration for Hugging Face Hub integration.

    Attributes
    ----------
    cache_dir : str or None
        Local cache directory for downloads
    offline : bool
        Whether to work offline
    timeout : int
        Request timeout in seconds
    """

    cache_dir: str | None = None
    offline: bool = False
    timeout: int = 30
