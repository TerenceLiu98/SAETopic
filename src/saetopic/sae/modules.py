"""
SAE architecture modules.

This module defines the sparse autoencoder architectures used for
learning reusable topic atoms.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch import Tensor


class TopKSAE(nn.Module):
    """
    Top-K Sparse Autoencoder for topic atom learning.

    This architecture learns a set of sparse features that can be
    interpreted as topic atoms. Features are activated competitively
    using top-k selection.

    The architecture consists of:
    - Encoder: Linear projection from input to feature space
    - Decoder: Linear reconstruction from features to input space
    - Top-k activation: Only keep top-k features per input

    Parameters
    ----------
    input_dim : int
        Dimension of input embeddings
    n_features : int or None, default=None
        Number of SAE features (topic atoms). If None, uses input_dim * expansion_factor
    expansion_factor : int, default=32
        Ratio of n_features to input_dim
    top_k : int, default=32
        Number of features to activate per input
    decoder_bias : bool, default=True
        Whether to use bias in decoder
    encoder_bias : bool, default=False
        Whether to use bias in encoder
    normalization : str or None, default=None
        Normalization method ("batch_norm", "layer_norm", or None)
    """

    def __init__(
        self,
        input_dim: int,
        n_features: int | None = None,
        expansion_factor: int = 32,
        top_k: int = 32,
        decoder_bias: bool = True,
        encoder_bias: bool = False,
        normalization: str | None = None,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.n_features = n_features or input_dim * expansion_factor
        self.expansion_factor = expansion_factor
        self.top_k = top_k
        self.decoder_bias = decoder_bias
        self.encoder_bias = encoder_bias
        self.normalization = normalization
        self.encoder_norm: nn.Module | None
        self.decoder_norm: nn.Module | None

        # Encoder: projects input to feature space
        self.encoder = nn.Linear(input_dim, self.n_features, bias=encoder_bias)

        # Decoder: reconstructs input from features
        self.decoder = nn.Linear(self.n_features, input_dim, bias=decoder_bias)

        # Optional normalization
        if normalization == "batch_norm":
            self.encoder_norm = nn.BatchNorm1d(self.n_features)
            self.decoder_norm = nn.BatchNorm1d(input_dim)
        elif normalization == "layer_norm":
            self.encoder_norm = nn.LayerNorm(self.n_features)
            self.decoder_norm = nn.LayerNorm(input_dim)
        else:
            self.encoder_norm = None
            self.decoder_norm = None

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights using Xavier/Glorot initialization."""
        for module in [self.encoder, self.decoder]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def encode(self, x: Tensor) -> Tensor:
        """
        Encode input to pre-activation features.

        Parameters
        ----------
        x : Tensor
            Input tensor (batch_size x input_dim)

        Returns
        -------
        Tensor
            Pre-activation features (batch_size x n_features)
        """
        h = self.encoder(x)
        if self.encoder_norm is not None:
            h = self.encoder_norm(h)
        return cast(Tensor, h)

    def activate(self, h: Tensor) -> tuple[Tensor, Tensor]:
        """
        Apply top-k activation to pre-activation features.

        Parameters
        ----------
        h : Tensor
            Pre-activation features (batch_size x n_features)

        Returns
        -------
        f : Tensor
            Sparse activated features (batch_size x n_features)
        indices : Tensor
            Top-k indices for each sample (batch_size x top_k)
        """
        topk_values, topk_indices = torch.topk(h, k=self.top_k, dim=-1)

        # Create sparse feature tensor
        f = torch.zeros_like(h)
        f.scatter_(dim=-1, index=topk_indices, src=torch.ones_like(topk_values))

        # Multiply values with binary mask
        f = f * h

        return f, topk_indices

    def decode(self, f: Tensor) -> Tensor:
        """
        Decode features to reconstruct input.

        Parameters
        ----------
        f : Tensor
            Activated features (batch_size x n_features)

        Returns
        -------
        Tensor
            Reconstructed input (batch_size x input_dim)
        """
        x_recon = self.decoder(f)
        if self.decoder_norm is not None:
            x_recon = self.decoder_norm(x_recon)
        return cast(Tensor, x_recon)

    def decode_sparse(self, topk_values: Tensor, topk_indices: Tensor) -> Tensor:
        """
        Decode top-k features without materializing a dense feature tensor.

        Parameters
        ----------
        topk_values : Tensor
            Activated feature values (batch_size x top_k)
        topk_indices : Tensor
            Activated feature indices (batch_size x top_k)

        Returns
        -------
        Tensor
            Reconstructed input (batch_size x input_dim)
        """
        decoder_weight = self.decoder.weight.t()  # (n_features x input_dim)
        selected_weight = decoder_weight[topk_indices]  # (batch_size x top_k x input_dim)
        x_recon = (topk_values.unsqueeze(-1) * selected_weight).sum(dim=1)

        if self.decoder.bias is not None:
            x_recon = x_recon + self.decoder.bias
        if self.decoder_norm is not None:
            x_recon = self.decoder_norm(x_recon)

        return x_recon

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Forward pass through the SAE.

        Parameters
        ----------
        x : Tensor
            Input tensor (batch_size x input_dim)

        Returns
        -------
        x_recon : Tensor
            Reconstructed input (batch_size x input_dim)
        h : Tensor
            Pre-activation features (batch_size x n_features)
        f : Tensor
            Activated sparse features (batch_size x n_features)
        topk_indices : Tensor
            Top-k feature indices (batch_size x top_k)
        """
        h = self.encode(x)
        f, topk_indices = self.activate(h)
        x_recon = self.decode(f)
        return x_recon, h, f, topk_indices

    def forward_sparse(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Forward pass using sparse top-k values for memory-efficient training.

        This avoids allocating the dense activated feature tensor
        (batch_size x n_features), while preserving the same reconstruction as
        the dense forward path.
        """
        h = self.encode(x)
        topk_values, topk_indices = torch.topk(h, k=self.top_k, dim=-1)
        x_recon = self.decode_sparse(topk_values, topk_indices)
        return x_recon, h, topk_values, topk_indices

    def compute_loss(
        self,
        x: Tensor,
        x_recon: Tensor,
        h: Tensor,
        f: Tensor,
        recon_loss_weight: float = 1.0,
        sparsity_loss_weight: float = 1.0,
        aux_loss_weight: float = 0.001,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """
        Compute SAE loss with reconstruction, sparsity, and usage-balance terms.

        Parameters
        ----------
        x : Tensor
            Input tensor (batch_size x input_dim)
        x_recon : Tensor
            Reconstructed input (batch_size x input_dim)
        h : Tensor
            Pre-activation features
        f : Tensor
            Activated features (batch_size x n_features)
        recon_loss_weight : float, default=1.0
            Weight for reconstruction loss
        sparsity_loss_weight : float, default=1.0
            Weight for L1 sparsity loss
        aux_loss_weight : float, default=0.001
            Weight for auxiliary usage-balance loss

        Returns
        -------
        total_loss : Tensor
            Combined loss tensor
        losses : dict
            Detached individual loss components
        """
        # h is accepted for API symmetry with SAE variants that need it.
        del h

        recon_loss = functional.mse_loss(x_recon, x)
        sparsity_loss = f.abs().sum(dim=-1).mean()

        feature_usage = (f > 0).float().sum(dim=0)
        target_usage = float(self.top_k)
        aux_loss = functional.mse_loss(feature_usage, torch.full_like(feature_usage, target_usage))

        total_loss = (
            recon_loss_weight * recon_loss
            + sparsity_loss_weight * sparsity_loss
            + aux_loss_weight * aux_loss
        )

        losses = {
            "total": total_loss.detach(),
            "reconstruction": recon_loss.detach(),
            "sparsity": sparsity_loss.detach(),
            "auxiliary": aux_loss.detach(),
        }

        return total_loss, losses

    def compute_loss_sparse(
        self,
        x: Tensor,
        x_recon: Tensor,
        h: Tensor,
        topk_values: Tensor,
        topk_indices: Tensor,
        recon_loss_weight: float = 1.0,
        sparsity_loss_weight: float = 1.0,
        aux_loss_weight: float = 0.001,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """
        Compute SAE loss from sparse top-k activations.

        This matches compute_loss() without requiring a dense activated feature
        tensor.
        """
        del h

        recon_loss = functional.mse_loss(x_recon, x)
        sparsity_loss = topk_values.abs().sum(dim=-1).mean()

        active_values = (topk_values > 0).to(topk_values.dtype)
        feature_usage = torch.zeros(
            self.n_features,
            dtype=topk_values.dtype,
            device=topk_values.device,
        )
        feature_usage.scatter_add_(
            dim=0,
            index=topk_indices.reshape(-1),
            src=active_values.reshape(-1),
        )
        target_usage = float(self.top_k)
        aux_loss = functional.mse_loss(feature_usage, torch.full_like(feature_usage, target_usage))

        total_loss = (
            recon_loss_weight * recon_loss
            + sparsity_loss_weight * sparsity_loss
            + aux_loss_weight * aux_loss
        )

        losses = {
            "total": total_loss.detach(),
            "reconstruction": recon_loss.detach(),
            "sparsity": sparsity_loss.detach(),
            "auxiliary": aux_loss.detach(),
        }

        return total_loss, losses


class BatchTopKSAE(TopKSAE):
    """
    Batch Top-K Sparse Autoencoder with efficient training utilities.

    Extends TopKSAE with additional methods for training:
    - Helper for loss computation
    - Feature activation statistics
    - Gradient checkpointing support

    Parameters
    ----------
    input_dim : int
        Dimension of input embeddings
    n_features : int or None, default=None
        Number of SAE features (topic atoms)
    expansion_factor : int, default=32
        Ratio of n_features to input_dim
    top_k : int, default=32
        Number of features to activate per input
    decoder_bias : bool, default=True
        Whether to use bias in decoder
    encoder_bias : bool, default=False
        Whether to use bias in encoder
    normalization : str or None, default=None
        Normalization method ("batch_norm", "layer_norm", or None)
    """

    def __init__(
        self,
        input_dim: int,
        n_features: int | None = None,
        expansion_factor: int = 32,
        top_k: int = 32,
        decoder_bias: bool = True,
        encoder_bias: bool = False,
        normalization: str | None = None,
    ):
        super().__init__(
            input_dim=input_dim,
            n_features=n_features,
            expansion_factor=expansion_factor,
            top_k=top_k,
            decoder_bias=decoder_bias,
            encoder_bias=encoder_bias,
            normalization=normalization,
        )

        # Statistics tracking
        self.feature_counts: Tensor
        self.update_count: Tensor
        self.register_buffer("feature_counts", torch.zeros(self.n_features))
        self.register_buffer("update_count", torch.tensor(0.0))

    def compute_loss(
        self,
        x: Tensor,
        x_recon: Tensor,
        h: Tensor,
        f: Tensor,
        recon_loss_weight: float = 1.0,
        sparsity_loss_weight: float = 1.0,
        aux_loss_weight: float = 0.001,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """
        Compute SAE loss with multiple components.

        Parameters
        ----------
        x : Tensor
            Input tensor (batch_size x input_dim)
        x_recon : Tensor
            Reconstructed input (batch_size x input_dim)
        h : Tensor
            Pre-activation features (batch_size x n_features)
        f : Tensor
            Activated features (batch_size x n_features)
        recon_loss_weight : float, default=1.0
            Weight for reconstruction loss
        sparsity_loss_weight : float, default=1.0
            Weight for L1 sparsity loss
        aux_loss_weight : float, default=0.001
            Weight for auxiliary loss (balance feature usage)

        Returns
        -------
        total_loss : Tensor
            Combined loss tensor
        losses : dict
            Dictionary of individual loss components
        """
        return super().compute_loss(
            x,
            x_recon,
            h,
            f,
            recon_loss_weight=recon_loss_weight,
            sparsity_loss_weight=sparsity_loss_weight,
            aux_loss_weight=aux_loss_weight,
        )

    def update_feature_stats(self, f: Tensor) -> None:
        """
        Update feature activation statistics.

        Parameters
        ----------
        f : Tensor
            Activated features (batch_size x n_features)
        """
        with torch.no_grad():
            active_features = (f > 0).float().sum(dim=0)
            self.feature_counts += active_features
            self.update_count += f.shape[0]

    def update_feature_stats_sparse(
        self,
        topk_values: Tensor,
        topk_indices: Tensor,
    ) -> None:
        """
        Update feature activation statistics from sparse top-k activations.
        """
        with torch.no_grad():
            active_values = (topk_values > 0).to(self.feature_counts.dtype)
            active_features = torch.zeros_like(self.feature_counts)
            active_features.scatter_add_(
                dim=0,
                index=topk_indices.reshape(-1),
                src=active_values.reshape(-1),
            )
            self.feature_counts += active_features
            self.update_count += topk_values.shape[0]

    def get_feature_usage(self) -> Tensor:
        """
        Get normalized feature usage statistics.

        Returns
        -------
        Tensor
            Feature usage frequencies (n_features,)
        """
        if self.update_count > 0:
            return self.feature_counts / self.update_count
        return torch.zeros_like(self.feature_counts)

    def get_dead_features(self, threshold: float = 0.01) -> Tensor:
        """
        Identify dead (rarely activated) features.

        Parameters
        ----------
        threshold : float, default=0.01
            Usage frequency below which a feature is considered dead

        Returns
        -------
        Tensor
            Boolean mask of dead features (n_features,)
        """
        usage = self.get_feature_usage()
        return usage < threshold

    def reset_feature_stats(self) -> None:
        """Reset feature activation statistics."""
        self.feature_counts.zero_()
        self.update_count.zero_()


def create_sae(
    input_dim: int,
    architecture: str = "topk",
    n_features: int | None = None,
    expansion_factor: int = 32,
    top_k: int = 32,
    **kwargs,
) -> TopKSAE | BatchTopKSAE:
    """
    Factory function to create SAE models.

    Parameters
    ----------
    input_dim : int
        Dimension of input embeddings
    architecture : str, default="topk"
        SAE architecture type ("topk", "batch_topk")
    n_features : int or None, default=None
        Number of SAE features
    expansion_factor : int, default=32
        Ratio of n_features to input_dim
    top_k : int, default=32
        Number of features to activate per input
    **kwargs
        Additional arguments passed to SAE constructor

    Returns
    -------
    TopKSAE or BatchTopKSAE
        Initialized SAE model
    """
    if architecture == "topk":
        return TopKSAE(
            input_dim=input_dim,
            n_features=n_features,
            expansion_factor=expansion_factor,
            top_k=top_k,
            **kwargs,
        )
    elif architecture == "batch_topk":
        return BatchTopKSAE(
            input_dim=input_dim,
            n_features=n_features,
            expansion_factor=expansion_factor,
            top_k=top_k,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
