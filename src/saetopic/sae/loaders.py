"""
SAE checkpoint loading utilities.

This module handles downloading and loading pretrained SAE checkpoints
from Hugging Face Hub or local paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class SAECheckpoint:
    """
    Container for SAE checkpoint metadata and weights.

    Attributes
    ----------
    repo_id : str
        Hugging Face repository ID
    embedding_model : str
        Embedding model used for training
    embedding_task : str
        Task type for embeddings (e.g., "clustering")
    embedding_dim : int
        Dimension of embedding space
    sae_architecture : str
        SAE architecture type ("topk", "batch_topk")
    expansion_factor : int
        Ratio of features to input dimension
    n_features : int
        Number of SAE features (topic atoms)
    top_k : int
        Number of features activated per input
    config : dict
        Full checkpoint configuration
    """

    repo_id: str
    embedding_model: str
    embedding_task: str
    embedding_dim: int
    sae_architecture: str
    expansion_factor: int
    n_features: int
    top_k: int
    config: dict[str, Any]

    @classmethod
    def from_pretrained(cls, repo_id: str) -> "SAECheckpoint":
        """
        Load checkpoint metadata and weights from Hugging Face Hub.

        Parameters
        ----------
        repo_id : str
            Hugging Face model ID (e.g., "saetopic/jina-v5-sae-small")

        Returns
        -------
        SAECheckpoint
            Loaded checkpoint with metadata and weights
        """
        # TODO: Implement HF Hub download
        raise NotImplementedError("SAECheckpoint.from_pretrained is not implemented yet")


def load_sae_weights(repo_id: str, local_cache: str | None = None):
    """
    Load SAE weights from checkpoint.

    Parameters
    ----------
    repo_id : str
        Hugging Face model ID
    local_cache : str or None, default=None
        Local cache path for weights

    Returns
    -------
    Loaded model weights
    """
    # TODO: Implement weight loading
    raise NotImplementedError("load_sae_weights is not implemented yet")
