"""
SAE activation extraction utilities.

This module handles computing sparse feature activations from
document embeddings using a trained SAE.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def extract_activations(
    embeddings: np.ndarray,
    sae_model,
    batch_size: int = 32,
    device: str = "auto",
) -> np.ndarray:
    """
    Extract sparse SAE activations from document embeddings.

    Parameters
    ----------
    embeddings : np.ndarray
        Document embeddings (n_docs x embedding_dim)
    sae_model : SAE model
        Trained sparse autoencoder
    batch_size : int, default=32
        Batch size for processing
    device : str, default="auto"
        Device for computation

    Returns
    -------
    np.ndarray
        Sparse activations (n_docs x n_features)
    """
    # TODO: Implement activation extraction (Week 3)
    raise NotImplementedError("extract_activations will be implemented in Week 3")
