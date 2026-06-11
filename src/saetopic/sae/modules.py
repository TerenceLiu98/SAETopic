"""
SAE architecture modules.

This module defines the sparse autoencoder architectures used for
learning reusable topic atoms.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as functional
from torch import Tensor


@torch.no_grad()
def set_decoder_norm_to_unit_norm(decoder_weight: Tensor) -> None:
    """Normalize decoder columns to unit norm, matching SAE-TM."""
    eps = torch.finfo(decoder_weight.dtype).eps
    decoder_weight.div_(decoder_weight.norm(dim=0, keepdim=True) + eps)


@torch.no_grad()
def remove_gradient_parallel_to_decoder_directions(decoder_weight: Tensor) -> None:
    """Remove decoder gradients parallel to decoder directions, matching SAE-TM."""
    if decoder_weight.grad is None:
        return

    normed_weight = decoder_weight / (decoder_weight.norm(dim=0, keepdim=True) + 1e-6)
    parallel_component = (decoder_weight.grad * normed_weight).sum(dim=0, keepdim=True)
    decoder_weight.grad.sub_(parallel_component * normed_weight)


@torch.no_grad()
def set_decoder_rows_to_unit_norm(decoder_weight: Tensor) -> None:
    """Normalize decoder rows to unit norm for JumpReLU-style W_dec."""
    eps = torch.finfo(decoder_weight.dtype).eps
    decoder_weight.div_(decoder_weight.norm(dim=1, keepdim=True) + eps)


@torch.no_grad()
def remove_gradient_parallel_to_decoder_rows(decoder_weight: Tensor) -> None:
    """Remove decoder row-parallel gradients for JumpReLU-style W_dec."""
    if decoder_weight.grad is None:
        return

    normed_weight = decoder_weight / (decoder_weight.norm(dim=1, keepdim=True) + 1e-6)
    parallel_component = (decoder_weight.grad * normed_weight).sum(dim=1, keepdim=True)
    decoder_weight.grad.sub_(parallel_component * normed_weight)


@torch.no_grad()
def geometric_median(points: Tensor, max_iter: int = 100, tol: float = 1e-5) -> Tensor:
    """Compute the geometric median used by SAE-TM to initialize b_dec."""
    guess = points.mean(dim=0)
    previous = torch.zeros_like(guess)
    weights = torch.ones(len(points), device=points.device)

    for _ in range(max_iter):
        previous = guess
        distances = torch.norm(points - guess, dim=1).clamp_min(1e-9)
        weights = 1 / distances
        weights = weights / weights.sum()
        guess = (weights.unsqueeze(1) * points).sum(dim=0)
        if torch.norm(guess - previous) < tol:
            break

    return guess


class RectangleFunction(autograd.Function):
    """Straight-through rectangle used by SAE-TM JumpReLU threshold gradients."""

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:  # type: ignore[override]
        ctx.save_for_backward(x)
        return ((x > -0.5) & (x < 0.5)).float()

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor]:  # type: ignore[override]
        (x,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[(x <= -0.5) | (x >= 0.5)] = 0
        return (grad_input,)


class JumpReLUFunction(autograd.Function):
    """SAE-TM JumpReLU with a surrogate threshold gradient."""

    @staticmethod
    def forward(ctx, x: Tensor, threshold: Tensor, bandwidth: float) -> Tensor:  # type: ignore[override]
        ctx.save_for_backward(x, threshold, torch.tensor(bandwidth, device=x.device))
        return x * (x > threshold).float()

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, Tensor, None]:  # type: ignore[override]
        x, threshold, bandwidth_tensor = ctx.saved_tensors
        bandwidth = float(bandwidth_tensor.item())
        x_grad = (x > threshold).float() * grad_output
        threshold_grad = (
            -(threshold / bandwidth)
            * RectangleFunction.apply((x - threshold) / bandwidth)
            * grad_output
        )
        return x_grad, threshold_grad, None


class StepFunction(autograd.Function):
    """SAE-TM hard step with a surrogate threshold gradient."""

    @staticmethod
    def forward(ctx, x: Tensor, threshold: Tensor, bandwidth: float) -> Tensor:  # type: ignore[override]
        ctx.save_for_backward(x, threshold, torch.tensor(bandwidth, device=x.device))
        return (x > threshold).float()

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, Tensor, None]:  # type: ignore[override]
        x, threshold, bandwidth_tensor = ctx.saved_tensors
        bandwidth = float(bandwidth_tensor.item())
        x_grad = torch.zeros_like(x)
        threshold_grad = (
            -(1.0 / bandwidth)
            * RectangleFunction.apply((x - threshold) / bandwidth)
            * grad_output
        )
        return x_grad, threshold_grad, None


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
        self.encoder = nn.Linear(input_dim, self.n_features, bias=True)

        # SAE-TM uses a bias-free decoder plus a separate decoder bias b_dec.
        self.decoder = nn.Linear(self.n_features, input_dim, bias=False)
        self.b_dec = nn.Parameter(torch.zeros(input_dim))
        self.threshold: Tensor
        self.register_buffer("threshold", torch.tensor(-1.0, dtype=torch.float32))
        self.num_tokens_since_fired: Tensor
        self.register_buffer(
            "num_tokens_since_fired",
            torch.zeros(self.n_features, dtype=torch.long),
        )
        self.dead_feature_threshold = 10_000_000
        self.top_k_aux = input_dim // 2

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
        """Initialize weights to match SAE-TM TopK/BatchTopK."""
        nn.init.kaiming_uniform_(self.decoder.weight)
        set_decoder_norm_to_unit_norm(self.decoder.weight.data)
        self.encoder.weight.data = self.decoder.weight.data.t().clone()
        self.encoder.bias.data.zero_()

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
        h = functional.relu(self.encoder(x - self.b_dec))
        if self.encoder_norm is not None:
            h = self.encoder_norm(h)
        return cast(Tensor, h)

    def activate(
        self,
        h: Tensor,
        use_threshold: bool = False,
    ) -> tuple[Tensor, Tensor]:
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
        if use_threshold:
            f = h * (h > self.threshold)
            topk_values, topk_indices = torch.topk(h, k=self.top_k, dim=-1, sorted=False)
            return f, topk_indices

        topk_values, topk_indices = torch.topk(h, k=self.top_k, dim=-1, sorted=False)

        # Create sparse feature tensor
        f = torch.zeros_like(h)
        f.scatter_(dim=-1, index=topk_indices, src=topk_values)

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
        x_recon = self.decoder(f) + self.b_dec
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
        if topk_indices.dim() == 1:
            batch_size = topk_values.shape[0]
            rows = topk_indices // self.n_features
            cols = topk_indices % self.n_features
            x_recon = self.b_dec.unsqueeze(0).expand(batch_size, -1).clone()
            contributions = topk_values.unsqueeze(-1) * decoder_weight[cols]
            x_recon.index_add_(dim=0, index=rows, source=contributions)
            return x_recon

        selected_weight = decoder_weight[topk_indices]  # (batch_size x top_k x input_dim)
        x_recon = (topk_values.unsqueeze(-1) * selected_weight).sum(dim=1) + self.b_dec

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
        topk_values, topk_indices = torch.topk(h, k=self.top_k, dim=-1, sorted=False)
        x_recon = self.decode_sparse(topk_values, topk_indices)
        return x_recon, h, topk_values, topk_indices

    def initialize_decoder_bias(self, x: Tensor) -> None:
        """Initialize b_dec with the geometric median of the first batch."""
        with torch.no_grad():
            self.b_dec.data = geometric_median(x).to(self.b_dec.dtype)

    def update_threshold(self, topk_values: Tensor, step: int, threshold_start_step: int = 1000) -> None:
        """Update the SAE-TM inference threshold after a warmup period."""
        if step <= threshold_start_step:
            return

        with torch.no_grad():
            if topk_values.dim() == 1:
                active = topk_values[topk_values > 0]
                min_activation = (
                    active.min().detach().to(torch.float32)
                    if active.numel()
                    else torch.tensor(0.0, device=topk_values.device)
                )
            else:
                active = topk_values.detach().clone()
                active[active <= 0] = float("inf")
                min_activation = active.min(dim=1).values.to(torch.float32).mean()

            if self.threshold < 0:
                self.threshold.data = min_activation
            else:
                self.threshold.data = 0.999 * self.threshold + 0.001 * min_activation

    def _update_firing_stats(self, topk_indices: Tensor, batch_size: int) -> None:
        """Update SAE-TM dead-feature counters."""
        with torch.no_grad():
            feature_indices = topk_indices % self.n_features
            did_fire = torch.zeros_like(self.num_tokens_since_fired, dtype=torch.bool)
            did_fire[feature_indices.reshape(-1)] = True
            self.num_tokens_since_fired += batch_size
            self.num_tokens_since_fired[did_fire] = 0

    def _auxk_loss(self, residual: Tensor, post_relu_acts: Tensor, aux_loss_weight: float) -> Tensor:
        """SAE-TM dead-feature auxiliary reconstruction loss."""
        if aux_loss_weight <= 0:
            return torch.tensor(0.0, dtype=residual.dtype, device=residual.device)

        dead_features = self.num_tokens_since_fired >= self.dead_feature_threshold
        n_dead = int(dead_features.sum().item())
        if n_dead == 0:
            return torch.tensor(0.0, dtype=residual.dtype, device=residual.device)

        k_aux = min(self.top_k_aux, n_dead)
        aux_latents = torch.where(dead_features.unsqueeze(0), post_relu_acts, -torch.inf)
        aux_values, aux_indices = aux_latents.topk(k_aux, sorted=False)
        aux_f = torch.zeros_like(post_relu_acts)
        aux_f.scatter_(dim=-1, index=aux_indices, src=aux_values)
        aux_recon = self.decoder(aux_f)
        aux_l2 = (residual.float() - aux_recon.float()).pow(2).sum(dim=-1).mean()
        residual_mu = residual.mean(dim=0, keepdim=True).expand_as(residual)
        denom = (residual.float() - residual_mu.float()).pow(2).sum(dim=-1).mean()
        return cast(Tensor, (aux_l2 / denom).nan_to_num(0.0))

    def compute_loss(
        self,
        x: Tensor,
        x_recon: Tensor,
        h: Tensor,
        f: Tensor,
        recon_loss_weight: float = 1.0,
        sparsity_loss_weight: float = 1.0,
        aux_loss_weight: float = 0.001,
        update_stats: bool = True,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """
        Compute SAE-TM TopK/BatchTopK loss.

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
        del sparsity_loss_weight

        residual = x - x_recon
        recon_loss = residual.pow(2).sum(dim=-1).mean()
        aux_loss = self._auxk_loss(residual.detach(), h, aux_loss_weight)
        total_loss = recon_loss_weight * recon_loss + aux_loss_weight * aux_loss
        if update_stats:
            self._update_firing_stats((f > 0).nonzero(as_tuple=False)[:, 1], x.shape[0])

        losses = {
            "total": total_loss.detach(),
            "reconstruction": recon_loss.detach(),
            "sparsity": torch.tensor(0.0, device=x.device),
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
        update_stats: bool = True,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """
        Compute SAE loss from sparse top-k activations.

        This matches compute_loss() without requiring a dense activated feature
        tensor.
        """
        del sparsity_loss_weight

        residual = x - x_recon
        recon_loss = residual.pow(2).sum(dim=-1).mean()
        aux_loss = self._auxk_loss(residual.detach(), h, aux_loss_weight)
        total_loss = recon_loss_weight * recon_loss + aux_loss_weight * aux_loss
        if update_stats:
            self._update_firing_stats(topk_indices, x.shape[0])

        losses = {
            "total": total_loss.detach(),
            "reconstruction": recon_loss.detach(),
            "sparsity": torch.tensor(0.0, device=x.device),
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

    def activate(
        self,
        h: Tensor,
        use_threshold: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Apply SAE-TM global batch top-k activation."""
        if use_threshold:
            f = h * (h > self.threshold)
            active_indices = (f > 0).sum(dim=0) > 0
            return f, active_indices.nonzero(as_tuple=False).flatten()

        flat = h.flatten()
        k_total = min(self.top_k * h.shape[0], flat.numel())
        topk_values, flat_indices = flat.topk(k_total, sorted=False)
        f = torch.zeros_like(flat)
        f.scatter_(dim=0, index=flat_indices, src=topk_values)
        return f.reshape_as(h), flat_indices

    def forward_sparse(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Forward pass using SAE-TM global batch top-k selection."""
        h = self.encode(x)
        flat = h.flatten()
        k_total = min(self.top_k * x.shape[0], flat.numel())
        topk_values, flat_indices = flat.topk(k_total, sorted=False)
        rows = flat_indices // self.n_features
        cols = flat_indices % self.n_features
        decoder_weight = self.decoder.weight.t()
        x_recon = self.b_dec.unsqueeze(0).expand(x.shape[0], -1).clone()
        contributions = topk_values.unsqueeze(-1) * decoder_weight[cols]
        x_recon.index_add_(dim=0, index=rows, source=contributions)
        return x_recon, h, topk_values, flat_indices

    def compute_loss(
        self,
        x: Tensor,
        x_recon: Tensor,
        h: Tensor,
        f: Tensor,
        recon_loss_weight: float = 1.0,
        sparsity_loss_weight: float = 1.0,
        aux_loss_weight: float = 0.001,
        update_stats: bool = True,
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
            update_stats=update_stats,
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
            feature_indices = topk_indices.reshape(-1) % self.n_features
            active_features.scatter_add_(
                dim=0,
                index=feature_indices,
                src=active_values.reshape(-1),
            )
            self.feature_counts += active_features
            if topk_indices.dim() == 1:
                self.update_count += max(1, topk_values.numel() // self.top_k)
            else:
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


class StandardSAE(nn.Module):
    """
    Standard SAE matching SAE-TM's StandardTrainer AutoEncoder.

    This architecture uses all ReLU encoder activations and is trained with
    reconstruction loss plus L1 sparsity, usually with sparsity warmup.
    """

    def __init__(
        self,
        input_dim: int,
        n_features: int | None = None,
        expansion_factor: int = 32,
        top_k: int = 32,
        decoder_bias: bool = True,
        encoder_bias: bool = True,
        normalization: str | None = None,
    ):
        super().__init__()
        del top_k, decoder_bias, encoder_bias, normalization

        self.input_dim = input_dim
        self.n_features = n_features or input_dim * expansion_factor
        self.expansion_factor = expansion_factor
        self.bias = nn.Parameter(torch.zeros(input_dim))
        self.encoder = nn.Linear(input_dim, self.n_features, bias=True)
        self.decoder = nn.Linear(self.n_features, input_dim, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        weight = torch.randn(
            self.input_dim,
            self.n_features,
            dtype=self.decoder.weight.dtype,
        )
        weight = weight / weight.norm(dim=0, keepdim=True).clamp_min(1e-9) * 0.1
        self.encoder.weight = nn.Parameter(weight.t().clone())
        self.decoder.weight = nn.Parameter(weight.clone())
        self.encoder.bias.data.zero_()

    @property
    def b_dec(self) -> Tensor:
        """Compatibility alias for TopKSAE's decoder bias name."""
        return self.bias

    def encode(self, x: Tensor) -> Tensor:
        """Encode input to ReLU features."""
        return cast(Tensor, functional.relu(self.encoder(x - self.bias)))

    def decode(self, f: Tensor) -> Tensor:
        """Decode feature activations."""
        return cast(Tensor, self.decoder(f) + self.bias)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Forward pass returning the same tuple shape as TopKSAE."""
        f = self.encode(x)
        x_recon = self.decode(f)
        active_indices = (f > 0).nonzero(as_tuple=False)
        return x_recon, f, f, active_indices

    def forward_sparse(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Standard SAE uses dense ReLU features; keep API compatible."""
        return self.forward(x)

    def compute_loss(
        self,
        x: Tensor,
        x_recon: Tensor,
        h: Tensor,
        f: Tensor,
        recon_loss_weight: float = 1.0,
        sparsity_loss_weight: float = 1.0,
        aux_loss_weight: float = 0.0,
        update_stats: bool = True,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute SAE-TM StandardTrainer loss."""
        del h, aux_loss_weight, update_stats

        residual = x - x_recon
        recon_loss = residual.pow(2).sum(dim=-1).mean()
        sparsity_loss = f.norm(p=1, dim=-1).mean()
        total_loss = recon_loss_weight * recon_loss + sparsity_loss_weight * sparsity_loss

        losses = {
            "total": total_loss.detach(),
            "reconstruction": recon_loss.detach(),
            "sparsity": sparsity_loss.detach(),
            "auxiliary": torch.tensor(0.0, device=x.device),
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
        aux_loss_weight: float = 0.0,
        update_stats: bool = True,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Standard SAE sparse API compatibility."""
        del topk_indices
        return self.compute_loss(
            x,
            x_recon,
            h,
            topk_values,
            recon_loss_weight=recon_loss_weight,
            sparsity_loss_weight=sparsity_loss_weight,
            aux_loss_weight=aux_loss_weight,
            update_stats=update_stats,
        )


class JumpReLUSAE(nn.Module):
    """
    JumpReLU SAE matching SAE-TM's JumpReluTrainer AutoEncoder.

    This architecture learns per-feature thresholds and optimizes a target-L0
    penalty rather than top-k selection.
    """

    def __init__(
        self,
        input_dim: int,
        n_features: int | None = None,
        expansion_factor: int = 32,
        top_k: int = 32,
        decoder_bias: bool = True,
        encoder_bias: bool = True,
        normalization: str | None = None,
        bandwidth: float = 0.001,
        target_l0: float = 20.0,
    ):
        super().__init__()
        del top_k, decoder_bias, encoder_bias, normalization

        self.input_dim = input_dim
        self.n_features = n_features or input_dim * expansion_factor
        self.expansion_factor = expansion_factor
        self.bandwidth = bandwidth
        self.target_l0 = target_l0
        self.apply_b_dec_to_input = False

        self.W_enc = nn.Parameter(torch.empty(input_dim, self.n_features))
        self.b_enc = nn.Parameter(torch.zeros(self.n_features))
        self.W_dec = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(self.n_features, input_dim)))
        self.b_dec = nn.Parameter(torch.zeros(input_dim))
        self.threshold = nn.Parameter(torch.ones(self.n_features) * 0.001)

        self.num_tokens_since_fired: Tensor
        self.register_buffer(
            "num_tokens_since_fired",
            torch.zeros(self.n_features, dtype=torch.long),
        )
        self.dead_feature_threshold = 10_000_000
        self.dead_features = -1

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights to match SAE-TM JumpReluAutoEncoder."""
        set_decoder_rows_to_unit_norm(self.W_dec.data)
        self.W_enc.data = self.W_dec.data.clone().t()

    def encode(self, x: Tensor, output_pre_jump: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        """Encode input using learned JumpReLU thresholds."""
        if self.apply_b_dec_to_input:
            x = x - self.b_dec
        pre_jump = x @ self.W_enc + self.b_enc
        f = cast(Tensor, JumpReLUFunction.apply(pre_jump, self.threshold, self.bandwidth))
        if output_pre_jump:
            return f, pre_jump
        return f

    def decode(self, f: Tensor) -> Tensor:
        """Decode feature activations."""
        return f @ self.W_dec + self.b_dec

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Forward pass returning API-compatible tensors."""
        f, pre_jump = cast(tuple[Tensor, Tensor], self.encode(x, output_pre_jump=True))
        x_recon = self.decode(f)
        active_indices = (f > 0).nonzero(as_tuple=False)
        return x_recon, pre_jump, f, active_indices

    def forward_sparse(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """JumpReLU uses dense thresholded features; keep API compatible."""
        return self.forward(x)

    def _update_firing_stats(self, f: Tensor, batch_size: int) -> None:
        """Track dead features like SAE-TM JumpReluTrainer."""
        with torch.no_grad():
            active_indices = f.sum(0) > 0
            did_fire = torch.zeros_like(self.num_tokens_since_fired, dtype=torch.bool)
            did_fire[active_indices] = True
            self.num_tokens_since_fired += batch_size
            self.num_tokens_since_fired[did_fire] = 0
            self.dead_features = int(
                (self.num_tokens_since_fired > self.dead_feature_threshold).sum().item()
            )

    def compute_loss(
        self,
        x: Tensor,
        x_recon: Tensor,
        h: Tensor,
        f: Tensor,
        recon_loss_weight: float = 1.0,
        sparsity_loss_weight: float = 1.0,
        aux_loss_weight: float = 0.0,
        update_stats: bool = True,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute SAE-TM JumpReluTrainer reconstruction plus target-L0 loss."""
        del h, aux_loss_weight

        residual = x - x_recon
        recon_loss = residual.pow(2).sum(dim=-1).mean()
        l0 = StepFunction.apply(f, self.threshold, self.bandwidth).sum(dim=-1).mean()
        sparsity_loss = ((l0 / self.target_l0) - 1).pow(2)
        total_loss = recon_loss_weight * recon_loss + sparsity_loss_weight * sparsity_loss
        if update_stats:
            self._update_firing_stats(f, x.shape[0])

        losses = {
            "total": total_loss.detach(),
            "reconstruction": recon_loss.detach(),
            "sparsity": sparsity_loss.detach(),
            "auxiliary": torch.tensor(0.0, device=x.device),
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
        aux_loss_weight: float = 0.0,
        update_stats: bool = True,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """JumpReLU sparse API compatibility."""
        del topk_indices
        return self.compute_loss(
            x,
            x_recon,
            h,
            topk_values,
            recon_loss_weight=recon_loss_weight,
            sparsity_loss_weight=sparsity_loss_weight,
            aux_loss_weight=aux_loss_weight,
            update_stats=update_stats,
        )


def create_sae(
    input_dim: int,
    architecture: str = "topk",
    n_features: int | None = None,
    expansion_factor: int = 32,
    top_k: int = 32,
    **kwargs,
) -> TopKSAE | BatchTopKSAE | StandardSAE | JumpReLUSAE:
    """
    Factory function to create SAE models.

    Parameters
    ----------
    input_dim : int
        Dimension of input embeddings
    architecture : str, default="topk"
        SAE architecture type ("standard", "jumprelu", "topk", "batch_topk")
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
    StandardSAE, JumpReLUSAE, TopKSAE, or BatchTopKSAE
        Initialized SAE model
    """
    if architecture == "standard":
        return StandardSAE(
            input_dim=input_dim,
            n_features=n_features,
            expansion_factor=expansion_factor,
            top_k=top_k,
            **kwargs,
        )
    elif architecture == "jumprelu":
        return JumpReLUSAE(
            input_dim=input_dim,
            n_features=n_features,
            expansion_factor=expansion_factor,
            top_k=top_k,
            **kwargs,
        )
    elif architecture == "topk":
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
