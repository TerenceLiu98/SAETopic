"""
SAE architecture modules.

This module defines the sparse autoencoder architectures used for
learning reusable topic atoms.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    import torch.nn as nn


class TopKSAE:
    """
    Top-K Sparse Autoencoder for topic atom learning.

    This architecture learns a set of sparse features that can be
    interpreted as topic atoms. Features are activated competitively
    using top-k selection.

    Parameters
    ----------
    input_dim : int
        Dimension of input embeddings
    n_features : int
        Number of SAE features (topic atoms)
    expansion_factor : int, default=32
        Ratio of n_features to input_dim
    top_k : int, default=32
        Number of features to activate per input
    """

    def __init__(
        self,
        input_dim: int,
        n_features: int | None = None,
        expansion_factor: int = 32,
        top_k: int = 32,
    ):
        self.input_dim = input_dim
        self.n_features = n_features or input_dim * expansion_factor
        self.expansion_factor = expansion_factor
        self.top_k = top_k

    # TODO: Implement full architecture (Week 2)
    raise NotImplementedError("TopKSAE will be implemented in Week 2")


class BatchTopKSAE:
    """
    Batch Top-K Sparse Autoencoder for efficient training.

    Similar to TopKSAE but processes inputs in batches with
    optimized top-k computation.
    """

    # TODO: Implement full architecture (Week 2)
    raise NotImplementedError("BatchTopKSAE will be implemented in Week 2")
