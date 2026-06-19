"""
Training utilities for Sparse Autoencoders.

This module provides the main training loop and optimizer classes
for training SAE models on embedding datasets.
"""

from __future__ import annotations

import json
import math
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
import torch.nn as nn
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from torch import Tensor
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from saetopic.sae.modules import (
        BatchTopKSAE,
        JumpReLUSAE,
        MatryoshkaBatchTopKSAE,
        OrtBatchTopKSAE,
        StandardSAE,
        TopKSAE,
    )
    from saetopic.training.data import (
        EmbeddingDataset,
        ShardedEmbeddingDataset,
        StreamingEmbeddingDataset,
    )


@dataclass
class TrainingConfig:
    """
    Configuration for SAE training.

    Attributes
    ----------
    input_dim : int
        Input embedding dimension
    n_features : int or None
        Number of SAE features (default: input_dim * expansion_factor)
    expansion_factor : int
        Ratio of features to input dimension
    top_k : int
        Number of features to activate per input
    learning_rate : float
        Learning rate for optimizer
    batch_size : int
        Training batch size
    n_epochs : int
        Number of training epochs
    device : str
        Device for training ("auto", "cpu", "cuda", "mps")
    architecture : str
        SAE architecture ("standard", "jumprelu", "topk", "batch_topk",
        "matryoshka_batch_topk", "ort_batch_topk")
    seed : int
        Random seed for reproducibility
    save_frequency : int
        Save checkpoint every N epochs
    log_frequency : int
        Log metrics every N batches
    recon_loss_weight : float
        Weight for reconstruction loss
    sparsity_loss_weight : float
        Weight for sparsity loss
    aux_loss_weight : float
        Weight for auxiliary loss
    """

    input_dim: int = 1024
    n_features: int | None = None
    expansion_factor: int = 32
    top_k: int = 32
    learning_rate: float = 1e-3
    batch_size: int = 256
    n_epochs: int = 100
    steps: int | None = None
    warmup_ratio: float = 0.1
    warmup_steps: int | None = None
    device: str = "auto"
    architecture: str = "batch_topk"
    seed: int = 42
    save_frequency: int = 10
    log_frequency: int = 10
    recon_loss_weight: float = 1.0
    sparsity_loss_weight: float = 1.0
    sparsity_warmup_steps: int | None = 2000
    aux_loss_weight: float = 1 / 32
    bandwidth: float = 0.001
    target_l0: float = 20.0
    matryoshka_group_sizes: list[int] | None = None
    matryoshka_group_fractions: list[float] | None = None
    matryoshka_group_weights: list[float] | None = None
    matryoshka_active_groups: int | None = None
    # OrtSAE (ort_batch_topk) orthogonality penalty hyperparameters.
    orthogonality_weight: float = 0.25
    orthogonality_chunk_size: int = 8192
    orthogonality_freq: int = 1
    early_stopping: bool = False
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 1e-4
    early_stopping_metric: str = "reconstruction"

    # Additional model parameters
    decoder_bias: bool = True
    encoder_bias: bool = False
    normalization: str | None = None

    # Output paths
    output_dir: str = "checkpoints/sae"
    checkpoint_name: str = "sae_checkpoint"
    resume: bool = False
    resume_from_checkpoint: str | None = None

    # Dataset info (for model card)
    dataset_name: str = ""
    dataset_license: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "input_dim": self.input_dim,
            "n_features": self.n_features or self.input_dim * self.expansion_factor,
            "expansion_factor": self.expansion_factor,
            "top_k": self.top_k,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
            "steps": self.steps,
            "warmup_ratio": self.warmup_ratio,
            "warmup_steps": self.warmup_steps,
            "architecture": self.architecture,
            "seed": self.seed,
            "decoder_bias": self.decoder_bias,
            "encoder_bias": self.encoder_bias,
            "normalization": self.normalization,
            "recon_loss_weight": self.recon_loss_weight,
            "sparsity_loss_weight": self.sparsity_loss_weight,
            "sparsity_warmup_steps": self.sparsity_warmup_steps,
            "aux_loss_weight": self.aux_loss_weight,
            "bandwidth": self.bandwidth,
            "target_l0": self.target_l0,
            "matryoshka_group_sizes": self.matryoshka_group_sizes,
            "matryoshka_group_fractions": self.matryoshka_group_fractions,
            "matryoshka_group_weights": self.matryoshka_group_weights,
            "matryoshka_active_groups": self.matryoshka_active_groups,
            "orthogonality_weight": self.orthogonality_weight,
            "orthogonality_chunk_size": self.orthogonality_chunk_size,
            "orthogonality_freq": self.orthogonality_freq,
            "early_stopping": self.early_stopping,
            "early_stopping_patience": self.early_stopping_patience,
            "early_stopping_min_delta": self.early_stopping_min_delta,
            "early_stopping_metric": self.early_stopping_metric,
            "resume": self.resume,
            "resume_from_checkpoint": self.resume_from_checkpoint,
        }


@dataclass
class TrainingState:
    """
    Training state for checkpointing.

    Attributes
    ----------
    epoch : int
        Current epoch number
    global_step : int
        Global training step
    best_loss : float
        Best loss seen so far
    losses : dict
        Loss history
    """

    epoch: int = 0
    global_step: int = 0
    best_loss: float = float("inf")
    losses: dict[str, list[float]] = field(default_factory=dict)

    def update(self, losses: dict[str, float], increment_step: bool = True) -> None:
        """Update training state with new losses."""
        if increment_step:
            self.global_step += 1
        for key, value in losses.items():
            if key not in self.losses:
                self.losses[key] = []
            self.losses[key].append(value)

        if "total" in losses and losses["total"] < self.best_loss:
            self.best_loss = losses["total"]


class SAEOptimizer:
    """
    Optimizer wrapper for SAE training.

    Handles the SAE-TM optimizer path: Adam, linear warmup/optional decay,
    decoder-gradient projection, gradient clipping, and decoder unit-norm
    maintenance.

    Parameters
    ----------
    model : nn.Module
        SAE model to optimize
    learning_rate : float
        Learning rate
    weight_decay : float, default=0.0
        Weight decay for regularization
    use_scheduler : bool, default=True
        Whether to use learning rate scheduler
    warmup_steps : int, default=1000
        Number of warmup steps for scheduler
    total_steps : int or None, default=None
        Total training steps for scheduler (None for default estimate)
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        use_scheduler: bool = True,
        warmup_steps: int = 1000,
        total_steps: int | None = None,
        betas: tuple[float, float] = (0.9, 0.999),
    ):
        del weight_decay
        self.model = model
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            betas=betas,
        )

        self.use_scheduler = use_scheduler
        self.scheduler = None
        self.total_steps = total_steps if total_steps is not None else 100_000

        if use_scheduler:
            def lr_lambda(step: int) -> float:
                if warmup_steps and step < warmup_steps:
                    return step / warmup_steps
                return 1.0

            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lr_lambda,
            )

    def step(self, loss: Tensor) -> None:
        """Perform optimizer step."""
        from saetopic.sae.modules import (
            remove_gradient_parallel_to_decoder_directions,
            remove_gradient_parallel_to_decoder_rows,
            set_decoder_norm_to_unit_norm,
            set_decoder_rows_to_unit_norm,
        )

        self.optimizer.zero_grad()
        loss.backward()
        decoder = getattr(self.model, "decoder", None)
        jump_decoder = getattr(self.model, "W_dec", None)
        if jump_decoder is not None:
            remove_gradient_parallel_to_decoder_rows(jump_decoder)
        elif decoder is not None and getattr(decoder, "weight", None) is not None:
            remove_gradient_parallel_to_decoder_directions(decoder.weight)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        if jump_decoder is not None:
            set_decoder_rows_to_unit_norm(jump_decoder.data)
        elif decoder is not None and getattr(decoder, "weight", None) is not None:
            set_decoder_norm_to_unit_norm(decoder.weight.data)

        if self.scheduler is not None:
            self.scheduler.step()

    def state_dict(self) -> dict:
        """Get optimizer state dict."""
        state = {"optimizer": self.optimizer.state_dict()}
        if self.scheduler is not None:
            state["scheduler"] = self.scheduler.state_dict()
        return state

    def load_state_dict(self, state_dict: dict) -> None:
        """Load optimizer state dict."""
        self.optimizer.load_state_dict(state_dict["optimizer"])
        if "scheduler" in state_dict and self.scheduler is not None:
            self.scheduler.load_state_dict(state_dict["scheduler"])


class SAETrainer:
    """
    Trainer for Sparse Autoencoder models.

    Parameters
    ----------
    model : TopKSAE
        SAE model to train
    config : TrainingConfig
        Training configuration
    output_dir : str
        Output directory for checkpoints
    """

    def __init__(
        self,
        model: "TopKSAE | BatchTopKSAE | MatryoshkaBatchTopKSAE | StandardSAE | JumpReLUSAE",
        config: TrainingConfig,
        output_dir: str | None = None,
    ):
        self.config = config
        self.output_dir = Path(output_dir or config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)

        self.model = model.to(self.device)

        # Optimizer will be created in fit() once we know the dataset size
        # This allows accurate total_steps calculation for the scheduler
        self.optimizer: SAEOptimizer | None = None

        # Training state
        self.state = TrainingState()

        # Set random seed
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

    def _load_resume_checkpoint(self) -> None:
        """Restore model, optimizer, and training state from a checkpoint."""
        if self.config.resume_from_checkpoint is None:
            return
        if self.optimizer is None:
            raise RuntimeError("Optimizer must be created before loading a resume checkpoint")

        checkpoint_dir = Path(self.config.resume_from_checkpoint)
        if not checkpoint_dir.is_dir():
            raise FileNotFoundError(f"Resume checkpoint directory not found: {checkpoint_dir}")

        state_dict = _load_checkpoint_weights(checkpoint_dir, map_location="cpu")
        self.model.load_state_dict(state_dict, strict=True)

        optimizer_path = checkpoint_dir / "optimizer.pt"
        if not optimizer_path.exists():
            raise FileNotFoundError(f"optimizer.pt not found in {checkpoint_dir}")
        optimizer_state = torch.load(optimizer_path, map_location=self.device)
        self.optimizer.load_state_dict(optimizer_state)

        training_state_path = checkpoint_dir / "training_state.pt"
        if not training_state_path.exists():
            raise FileNotFoundError(f"training_state.pt not found in {checkpoint_dir}")
        raw_state = torch.load(training_state_path, map_location="cpu")
        self.state = TrainingState(
            epoch=int(raw_state.get("epoch", 0)),
            global_step=int(raw_state.get("global_step", 0)),
            best_loss=float(raw_state.get("best_loss", float("inf"))),
            losses=dict(raw_state.get("losses", {})),
        )
        print(
            "Resumed SAE training from "
            f"{checkpoint_dir} (epoch={self.state.epoch}, step={self.state.global_step})"
        )

    def _train_batch(self, batch: Tensor) -> dict[str, Tensor]:
        """Run one SAE-TM training step."""
        batch = batch.to(self.device)
        initialize_decoder_bias = cast(Any, getattr(self.model, "initialize_decoder_bias", None))
        if self.state.global_step == 0 and callable(initialize_decoder_bias):
            initialize_decoder_bias(batch)

        x_recon, h, topk_values, topk_indices = self.model.forward_sparse(batch)
        update_threshold = cast(Any, getattr(self.model, "update_threshold", None))
        if callable(update_threshold):
            update_threshold(topk_values, self.state.global_step)

        sparsity_loss_weight = self.config.sparsity_loss_weight
        if self.config.architecture in {"standard", "jumprelu"} and self.config.sparsity_warmup_steps:
            sparsity_scale = min(
                self.state.global_step / self.config.sparsity_warmup_steps,
                1.0,
            )
            sparsity_loss_weight *= sparsity_scale

        loss, losses = self.model.compute_loss_sparse(
            batch,
            x_recon,
            h,
            topk_values,
            topk_indices,
            recon_loss_weight=self.config.recon_loss_weight,
            sparsity_loss_weight=sparsity_loss_weight,
            aux_loss_weight=self.config.aux_loss_weight,
        )

        assert self.optimizer is not None
        self.optimizer.step(loss)
        self.state.global_step += 1

        return losses

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> dict[str, float]:
        """
        Train for one epoch.

        Parameters
        ----------
        train_loader : DataLoader
            Training data loader
        epoch : int
        Current epoch number

        Returns
        -------
        dict
            Average losses for the epoch
        """
        self.model.train()

        epoch_losses = {
            key: 0.0 for key in ["total", "reconstruction", "sparsity", "auxiliary", "r2"]
        }
        n_batches = 0

        # Use rich.progress for better terminal output
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(show_speed=True),
            TimeRemainingColumn(),
            TextColumn("• loss: {task.fields[loss]}"),
        ) as progress:
            task = progress.add_task(
                f"[cyan]Epoch {epoch}",
                total=len(train_loader),
                loss="0.0000",
            )

            for batch in train_loader:
                losses = self._train_batch(batch)

                # Update statistics
                n_batches += 1
                for key in epoch_losses:
                    epoch_losses[key] += losses[key].item()

                # Update progress bar
                avg_loss = epoch_losses["total"] / n_batches
                progress.update(
                    task,
                    advance=1,
                    description=f"[cyan]Epoch {epoch}",
                    loss=f"{avg_loss:.6f}",
                )

        # Average losses
        return {k: v / n_batches for k, v in epoch_losses.items()}

    def validate_epoch(self, val_loader: DataLoader) -> dict[str, float]:
        """Evaluate one validation epoch without updating SAE training statistics."""
        self.model.eval()

        val_losses = {
            key: 0.0 for key in ["total", "reconstruction", "sparsity", "auxiliary", "r2"]
        }
        n_batches = 0

        sparsity_loss_weight = self.config.sparsity_loss_weight
        if self.config.architecture in {"standard", "jumprelu"} and self.config.sparsity_warmup_steps:
            sparsity_scale = min(
                self.state.global_step / self.config.sparsity_warmup_steps,
                1.0,
            )
            sparsity_loss_weight *= sparsity_scale

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                x_recon, h, topk_values, topk_indices = self.model.forward_sparse(batch)
                _, losses = self.model.compute_loss_sparse(
                    batch,
                    x_recon,
                    h,
                    topk_values,
                    topk_indices,
                    recon_loss_weight=self.config.recon_loss_weight,
                    sparsity_loss_weight=sparsity_loss_weight,
                    aux_loss_weight=self.config.aux_loss_weight,
                    update_stats=False,
                )

                n_batches += 1
                for key in val_losses:
                    val_losses[key] += losses[key].item()

        self.model.train()
        return {k: v / max(1, n_batches) for k, v in val_losses.items()}

    def _create_optimizer(
        self,
        dataset: "EmbeddingDataset | ShardedEmbeddingDataset | StreamingEmbeddingDataset",
    ) -> None:
        """
        Create optimizer with scheduler based on dataset size.

        Parameters
        ----------
        dataset : EmbeddingDataset or StreamingEmbeddingDataset
            Training dataset
        """
        # Calculate total steps based on dataset type
        if self.config.steps is not None:
            total_steps = self.config.steps
        elif hasattr(dataset, "__len__"):
            # Standard dataset - calculate exact steps
            n_samples = len(dataset)
            steps_per_epoch = max(1, math.ceil(n_samples / self.config.batch_size))
            total_steps = steps_per_epoch * self.config.n_epochs
        elif hasattr(dataset, "max_samples") and dataset.max_samples:
            # Streaming dataset with known max_samples
            embedding_batch_size = getattr(dataset, "embedding_batch_size", None)
            if embedding_batch_size:
                full_embedding_batches = dataset.max_samples // embedding_batch_size
                remaining_samples = dataset.max_samples % embedding_batch_size
                steps_per_embedding_batch = math.ceil(
                    embedding_batch_size / self.config.batch_size
                )
                steps_per_epoch = full_embedding_batches * steps_per_embedding_batch
                if remaining_samples:
                    steps_per_epoch += math.ceil(
                        remaining_samples / self.config.batch_size
                    )
                steps_per_epoch = max(1, steps_per_epoch)
            else:
                steps_per_epoch = max(
                    1, math.ceil(dataset.max_samples / self.config.batch_size)
                )
            total_steps = steps_per_epoch * self.config.n_epochs
        else:
            # Streaming dataset without known size - use default
            total_steps = 100_000

        warmup_steps = (
            self.config.warmup_steps
            if self.config.warmup_steps is not None
            else int(self.config.warmup_ratio * total_steps)
        )
        self.optimizer = SAEOptimizer(
            self.model,
            learning_rate=self.config.learning_rate,
            use_scheduler=True,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            betas=(0.0, 0.999) if self.config.architecture == "jumprelu" else (0.9, 0.999),
        )

        print(f"Total training steps: {total_steps}")

    def fit(
        self,
        dataset: "EmbeddingDataset | ShardedEmbeddingDataset | StreamingEmbeddingDataset",
        val_dataset: "EmbeddingDataset | ShardedEmbeddingDataset | None" = None,
    ) -> TrainingState:
        """
        Train the SAE model.

        Supports both standard PyTorch Dataset and streaming iterators.

        Parameters
        ----------
        dataset : EmbeddingDataset or StreamingEmbeddingDataset
            Training dataset
        val_dataset : EmbeddingDataset or None
            Optional validation dataset

        Returns
        -------
        TrainingState
            Final training state
        """
        # Create optimizer with correct total_steps
        self._create_optimizer(dataset)
        self._load_resume_checkpoint()

        # Check if dataset is a streaming iterator
        is_streaming = hasattr(dataset, "__iter__") and not hasattr(dataset, "__len__")

        if is_streaming:
            return self._fit_streaming(cast("StreamingEmbeddingDataset", dataset), val_dataset)
        else:
            return self._fit_standard(
                cast("EmbeddingDataset | ShardedEmbeddingDataset", dataset),
                val_dataset,
            )

    def _fit_standard(
        self,
        dataset: "EmbeddingDataset | ShardedEmbeddingDataset",
        val_dataset: "EmbeddingDataset | ShardedEmbeddingDataset | None" = None,
    ) -> TrainingState:
        """Train with standard PyTorch Dataset (uses DataLoader)."""
        train_loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=self.device.type == "cuda",
        )
        val_loader = (
            DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=self.device.type == "cuda",
            )
            if val_dataset is not None
            else None
        )
        if self.config.early_stopping and val_loader is None:
            print("Warning: early_stopping=True requires val_dataset; disabling early stopping")
        early_stopping_enabled = self.config.early_stopping and val_loader is not None
        best_metric, epochs_without_improvement = self._resume_early_stopping_state()

        if self.config.steps is not None:
            return self._fit_standard_steps(train_loader)

        start_epoch = min(self.state.epoch + 1, self.config.n_epochs + 1)
        for epoch in range(start_epoch, self.config.n_epochs + 1):
            # Train epoch
            epoch_losses = self.train_epoch(train_loader, epoch)

            # Update state
            self.state.epoch = epoch
            self.state.update(epoch_losses, increment_step=False)

            # Print summary
            print(f"Epoch {epoch}/{self.config.n_epochs}")
            for key, value in epoch_losses.items():
                print(f"  {key}: {value:.6f}")

            if val_loader is not None:
                val_losses = self.validate_epoch(val_loader)
                self.state.update(
                    {f"val_{key}": value for key, value in val_losses.items()},
                    increment_step=False,
                )
                print("Validation")
                for key, value in val_losses.items():
                    print(f"  val_{key}: {value:.6f}")

                metric_name = self.config.early_stopping_metric
                if metric_name.startswith("val_"):
                    metric_name = metric_name.removeprefix("val_")
                if metric_name not in val_losses:
                    raise ValueError(
                        f"Unknown early_stopping_metric={self.config.early_stopping_metric!r}; "
                        f"available metrics: {', '.join(sorted(val_losses))}"
                    )

                current_metric = val_losses[metric_name]
                if self._is_improved_metric(current_metric, best_metric):
                    best_metric = current_metric
                    epochs_without_improvement = 0
                    self.save_checkpoint("best")
                else:
                    epochs_without_improvement += 1

                if (
                    early_stopping_enabled
                    and epochs_without_improvement >= self.config.early_stopping_patience
                ):
                    print(
                        "Early stopping triggered "
                        f"after {epochs_without_improvement} epochs without improvement "
                        f"on val_{metric_name}"
                    )
                    break

            # Save checkpoint
            if epoch % self.config.save_frequency == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}")

        # Save final checkpoint
        self.save_checkpoint("final")

        # Save training config and state
        self.save_metadata()

        return self.state

    def _resume_early_stopping_state(self) -> tuple[float, int]:
        """Recover early-stopping bookkeeping from saved validation history."""
        default_best = float("-inf") if self._metric_higher_is_better() else float("inf")
        metric_name = self.config.early_stopping_metric.removeprefix("val_")
        history_key = f"val_{metric_name}"
        history = self.state.losses.get(history_key, [])
        if not history:
            return default_best, 0

        if self._metric_higher_is_better():
            best_metric = max(history)
            best_index = history.index(best_metric)
        else:
            best_metric = min(history)
            best_index = history.index(best_metric)
        epochs_without_improvement = max(0, len(history) - best_index - 1)
        return float(best_metric), epochs_without_improvement

    def _metric_higher_is_better(self) -> bool:
        """Return whether the configured early stopping metric should increase."""
        metric_name = self.config.early_stopping_metric.removeprefix("val_")
        return metric_name == "r2"

    def _is_improved_metric(self, current_metric: float, best_metric: float) -> bool:
        """Check early stopping improvement with metric direction."""
        min_delta = self.config.early_stopping_min_delta
        if self._metric_higher_is_better():
            return current_metric > best_metric + min_delta
        return current_metric < best_metric - min_delta

    def _fit_standard_steps(self, train_loader: DataLoader) -> TrainingState:
        """Train with a finite DataLoader cycled to a fixed number of SAE-TM steps."""
        assert self.config.steps is not None
        self.model.train()

        running_losses = {
            key: 0.0 for key in ["total", "reconstruction", "sparsity", "auxiliary", "r2"]
        }
        n_batches = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(show_speed=True),
            TimeRemainingColumn(),
            TextColumn("• loss: {task.fields[loss]}"),
        ) as progress:
            task = progress.add_task(
                "[cyan]Training SAE",
                total=self.config.steps,
                loss="0.0000",
            )

            while self.state.global_step < self.config.steps:
                for batch in train_loader:
                    if self.state.global_step >= self.config.steps:
                        break

                    losses = self._train_batch(batch)
                    n_batches += 1
                    for key in running_losses:
                        running_losses[key] += losses[key].item()

                    avg_loss = running_losses["total"] / n_batches
                    progress.update(task, advance=1, loss=f"{avg_loss:.6f}")

                    if (
                        self.config.save_frequency > 0
                        and self.state.global_step % self.config.save_frequency == 0
                    ):
                        self.save_checkpoint(f"checkpoint_step_{self.state.global_step}")

        avg_losses = {key: value / max(1, n_batches) for key, value in running_losses.items()}
        self.state.epoch = math.ceil(self.state.global_step / max(1, len(train_loader)))
        self.state.update(avg_losses, increment_step=False)

        self.save_checkpoint("final")
        self.save_metadata()
        return self.state

    def _fit_streaming(
        self,
        streaming_dataset: "StreamingEmbeddingDataset",
        val_dataset: "EmbeddingDataset | ShardedEmbeddingDataset | None" = None,
    ) -> TrainingState:
        """
        Train with streaming dataset (iterator mode).

        In streaming mode, each "epoch" iterates through the streaming dataset
        once. The dataset handles buffering and shuffling internally.

        Parameters
        ----------
        streaming_dataset : StreamingEmbeddingDataset
            Streaming dataset that yields batches of embeddings
        val_dataset : EmbeddingDataset or None
            Optional validation dataset (not supported in streaming mode)

        Returns
        -------
        TrainingState
            Final training state
        """
        if val_dataset is not None:
            print("Warning: Validation dataset not supported in streaming mode")

        # Get embedding dim from streaming dataset
        if hasattr(streaming_dataset, "embedding_dim"):
            print(f"Detected embedding dim: {streaming_dataset.embedding_dim}")

        if self.config.steps is not None:
            return self._fit_streaming_steps(streaming_dataset)

        start_epoch = min(self.state.epoch + 1, self.config.n_epochs + 1)
        for epoch in range(start_epoch, self.config.n_epochs + 1):
            self.model.train()

            epoch_losses = {
                "total": 0.0,
                "reconstruction": 0.0,
                "sparsity": 0.0,
                "auxiliary": 0.0,
                "r2": 0.0,
            }
            n_batches = 0

            # Create progress bar for streaming (no total length known)
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(show_speed=True),
                TextColumn("• loss: {task.fields[loss]}"),
            ) as progress:
                task = progress.add_task(
                    f"[cyan]Epoch {epoch} (streaming)",
                    total=None,  # Unknown total for streaming
                    loss="0.0000",
                )

                for batch_embeddings in streaming_dataset:
                    # batch_embeddings might be larger than training batch size
                    # Split into smaller batches for training
                    for i in range(0, batch_embeddings.shape[0], self.config.batch_size):
                        batch = batch_embeddings[i : i + self.config.batch_size]
                        losses = self._train_batch(batch)

                        # Update statistics
                        n_batches += 1
                        for key in epoch_losses:
                            epoch_losses[key] += losses[key].item()

                        # Update progress bar
                        avg_loss = epoch_losses["total"] / n_batches
                        progress.update(task, advance=1, loss=f"{avg_loss:.6f}")

            # Average losses
            avg_losses = {k: v / n_batches for k, v in epoch_losses.items()}

            # Update state
            self.state.epoch = epoch
            self.state.update(avg_losses, increment_step=False)

            # Print summary
            print(f"Epoch {epoch}/{self.config.n_epochs} (streaming)")
            print(f"  Batches: {n_batches}")
            for key, value in avg_losses.items():
                print(f"  {key}: {value:.6f}")

            # Save checkpoint
            if epoch % self.config.save_frequency == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}")

        # Save final checkpoint
        self.save_checkpoint("final")

        # Save training config and state
        self.save_metadata()

        return self.state

    def _fit_streaming_steps(
        self,
        streaming_dataset: "StreamingEmbeddingDataset",
    ) -> TrainingState:
        """Train a streaming dataset for an exact fixed number of SAE-TM steps."""
        assert self.config.steps is not None
        self.model.train()

        running_losses = {
            "total": 0.0,
            "reconstruction": 0.0,
            "sparsity": 0.0,
            "auxiliary": 0.0,
            "r2": 0.0,
        }
        n_batches = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(show_speed=True),
            TimeRemainingColumn(),
            TextColumn("• loss: {task.fields[loss]}"),
        ) as progress:
            task = progress.add_task(
                "[cyan]Training SAE (streaming)",
                total=self.config.steps,
                loss="0.0000",
            )

            while self.state.global_step < self.config.steps:
                made_progress = False
                for batch_embeddings in streaming_dataset:
                    for i in range(0, batch_embeddings.shape[0], self.config.batch_size):
                        if self.state.global_step >= self.config.steps:
                            break

                        batch = batch_embeddings[i : i + self.config.batch_size]
                        losses = self._train_batch(batch)
                        made_progress = True

                        n_batches += 1
                        for key in running_losses:
                            running_losses[key] += losses[key].item()

                        avg_loss = running_losses["total"] / n_batches
                        progress.update(task, advance=1, loss=f"{avg_loss:.6f}")

                        if (
                            self.config.save_frequency > 0
                            and self.state.global_step % self.config.save_frequency == 0
                        ):
                            self.save_checkpoint(f"checkpoint_step_{self.state.global_step}")

                    if self.state.global_step >= self.config.steps:
                        break

                if not made_progress:
                    raise RuntimeError("Streaming dataset yielded no batches")

        avg_losses = {key: value / max(1, n_batches) for key, value in running_losses.items()}
        self.state.epoch = 1
        self.state.update(avg_losses, increment_step=False)

        self.save_checkpoint("final")
        self.save_metadata()
        return self.state

    def save_checkpoint(
        self,
        name: str,
    ) -> None:
        """
        Save model checkpoint.

        Parameters
        ----------
        name : str
            Checkpoint name
        """
        checkpoint_dir = self.output_dir / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model weights using safetensors if available
        try:
            from safetensors.torch import save_file

            state_dict = self.model.state_dict()
            save_file(
                {k: v.detach().cpu().contiguous() for k, v in state_dict.items()},
                checkpoint_dir / "model.safetensors",
            )
        except ImportError:
            torch.save(self.model.state_dict(), checkpoint_dir / "model.pt")

        # Save optimizer state
        assert self.optimizer is not None
        torch.save(self.optimizer.state_dict(), checkpoint_dir / "optimizer.pt")

        # Save training state
        torch.save(self.state.__dict__, checkpoint_dir / "training_state.pt")

        # Keep each checkpoint self-contained for direct upload/reuse.
        self._save_metadata_files(checkpoint_dir)

        print(f"Saved checkpoint to {checkpoint_dir}")

    def save_metadata(self) -> None:
        """Save training metadata (config, model card, etc.)."""
        self._save_metadata_files(self.output_dir)

    def _save_metadata_files(self, output_dir: Path) -> None:
        """Save metadata files into a checkpoint or output directory."""
        # Save config
        config_path = output_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        # Save model card template. README.md is recognized by Hugging Face Hub.
        model_card_path = output_dir / "model_card.md"
        self._create_model_card(model_card_path)
        (output_dir / "README.md").write_text(model_card_path.read_text())

        # Save checksums
        self._save_checksums(output_dir)

    def _create_model_card(self, path: Path) -> None:
        """Create model card markdown file."""
        n_features = self.config.n_features or self.config.input_dim * self.config.expansion_factor
        latest_r2 = self.state.losses.get("r2", [None])[-1]
        latest_val_r2 = self.state.losses.get("val_r2", [None])[-1]

        card = f"""# {self.config.checkpoint_name}

## Model Description

Sparse Autoencoder trained for topic modeling. This model learns {n_features} sparse features (topic atoms) from {self.config.input_dim}-dimensional embeddings.

## Training Data

- **Dataset**: {self.config.dataset_name or "See training script"}
- **License**: {self.config.dataset_license or "See dataset source"}

## Model Architecture

- **Architecture**: {self.config.architecture}
- **Input Dimension**: {self.config.input_dim}
- **Number of Features**: {n_features}
- **Expansion Factor**: {self.config.expansion_factor}
- **Top-K**: {self.config.top_k}
- **Matryoshka Group Sizes**: {self.config.matryoshka_group_sizes or "n/a"}
- **Matryoshka Group Fractions**: {self.config.matryoshka_group_fractions or "n/a"}
- **Matryoshka Group Weights**: {self.config.matryoshka_group_weights or "n/a"}
- **Matryoshka Active Groups**: {self.config.matryoshka_active_groups or "all"}
- **OrtSAE Orthogonality Weight**: {self.config.orthogonality_weight if self.config.architecture == "ort_batch_topk" else "n/a"}
- **OrtSAE Orthogonality Chunk Size**: {self.config.orthogonality_chunk_size if self.config.architecture == "ort_batch_topk" else "n/a"}
- **OrtSAE Orthogonality Frequency**: {self.config.orthogonality_freq if self.config.architecture == "ort_batch_topk" else "n/a"}

## Checkpoint Contents

This checkpoint contains SAE training artifacts:

- `model.safetensors` or `model.pt`: SAE model weights
- `optimizer.pt`: optimizer state for resuming experiments
- `training_state.pt`: epoch/loss history
- `config.json`: training and architecture configuration
- `checksums.txt`: SHA256 checksums for checkpoint tensors

## Usage Status

This is a trained SAE checkpoint. The end-to-end `SAETopicModel.from_pretrained`
and topic inference APIs are still in development, so this artifact is intended
for training experiments and future loader integration.

## Training

- **Epochs**: {self.config.n_epochs}
- **Steps**: {self.config.steps or "epoch-based"}
- **Warmup Ratio**: {self.config.warmup_ratio}
- **Batch Size**: {self.config.batch_size}
- **Learning Rate**: {self.config.learning_rate}
- **Sparsity Warmup Steps**: {self.config.sparsity_warmup_steps}
- **AuxK Alpha**: {self.config.aux_loss_weight}
- **Early Stopping**: {self.config.early_stopping}
- **Early Stopping Patience**: {self.config.early_stopping_patience}
- **Early Stopping Metric**: {self.config.early_stopping_metric}
- **Best Loss**: {self.state.best_loss:.6f}
- **Latest R2**: {latest_r2 if latest_r2 is not None else "n/a"}
- **Latest Validation R2**: {latest_val_r2 if latest_val_r2 is not None else "n/a"}

## License

Apache-2.0

## Attribution

This is a clean-room implementation trained on permissively licensed data.
"""
        path.write_text(card)

    def _save_checksums(self, output_dir: Path) -> None:
        """Save SHA256 checksums for model files."""
        import hashlib

        checksums = []
        for file in sorted(output_dir.rglob("*.safetensors")):
            sha256 = hashlib.sha256()
            sha256.update(file.read_bytes())
            checksums.append(f"{sha256.hexdigest()}  {file.relative_to(output_dir)}")

        for file in sorted(output_dir.rglob("*.pt")):
            sha256 = hashlib.sha256()
            sha256.update(file.read_bytes())
            checksums.append(f"{sha256.hexdigest()}  {file.relative_to(output_dir)}")

        checksum_path = output_dir / "checksums.txt"
        checksum_path.write_text("\n".join(checksums))


def _load_checkpoint_weights(
    checkpoint_dir: Path,
    map_location: str | torch.device = "cpu",
) -> dict[str, Tensor]:
    """Load model weights from a training checkpoint directory."""
    safetensors_path = checkpoint_dir / "model.safetensors"
    if safetensors_path.exists():
        from safetensors.torch import load_file

        return cast(dict[str, Tensor], load_file(str(safetensors_path), device=str(map_location)))

    pt_path = checkpoint_dir / "model.pt"
    if pt_path.exists():
        return cast(dict[str, Tensor], torch.load(pt_path, map_location=map_location))

    raise FileNotFoundError(
        f"No model weights found in {checkpoint_dir} "
        "(expected model.safetensors or model.pt)"
    )


def _checkpoint_global_step(checkpoint_dir: Path) -> int | None:
    """Return the saved global step for a checkpoint directory."""
    training_state_path = checkpoint_dir / "training_state.pt"
    if not training_state_path.exists():
        return None
    try:
        raw_state = torch.load(training_state_path, map_location="cpu")
    except Exception:
        return None
    if not isinstance(raw_state, dict) or "global_step" not in raw_state:
        return None
    return int(raw_state["global_step"])


def _latest_training_checkpoint(output_dir: Path) -> Path | None:
    """Find the child checkpoint with the largest saved global_step."""
    if not output_dir.exists():
        return None

    candidates: list[tuple[int, float, Path]] = []
    for checkpoint_dir in output_dir.iterdir():
        if not checkpoint_dir.is_dir():
            continue
        step = _checkpoint_global_step(checkpoint_dir)
        if step is None:
            continue
        has_weights = (checkpoint_dir / "model.safetensors").exists() or (
            checkpoint_dir / "model.pt"
        ).exists()
        if not has_weights or not (checkpoint_dir / "optimizer.pt").exists():
            continue
        candidates.append((step, checkpoint_dir.stat().st_mtime, checkpoint_dir))

    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[-1][2]


def _resolve_resume_checkpoint(config: TrainingConfig) -> Path | None:
    """Resolve explicit or automatic SAE training resume checkpoint."""
    if config.resume_from_checkpoint is not None:
        checkpoint_dir = Path(config.resume_from_checkpoint)
        if not checkpoint_dir.is_dir():
            raise FileNotFoundError(f"Resume checkpoint directory not found: {checkpoint_dir}")
        return checkpoint_dir

    if not config.resume:
        return None

    checkpoint_dir = _latest_training_checkpoint(Path(config.output_dir))
    if checkpoint_dir is None:
        print(f"No SAE checkpoint found in {config.output_dir}; starting from scratch")
        return None
    return checkpoint_dir


def _apply_model_config_from_checkpoint(config: TrainingConfig, checkpoint_dir: Path) -> None:
    """Use checkpoint model metadata for architecture-defining config fields."""
    config_path = checkpoint_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {checkpoint_dir}")
    with open(config_path) as f:
        checkpoint_config = json.load(f)

    model_fields = [
        "input_dim",
        "n_features",
        "expansion_factor",
        "top_k",
        "architecture",
        "decoder_bias",
        "encoder_bias",
        "normalization",
        "bandwidth",
        "target_l0",
        "matryoshka_group_sizes",
        "matryoshka_group_fractions",
        "matryoshka_group_weights",
        "matryoshka_active_groups",
        "orthogonality_weight",
        "orthogonality_chunk_size",
        "orthogonality_freq",
    ]
    for field_name in model_fields:
        if field_name in checkpoint_config:
            setattr(config, field_name, checkpoint_config[field_name])


def train_sae(
    embeddings_path: str | Path | None = None,
    dataset: "EmbeddingDataset | ShardedEmbeddingDataset | None" = None,
    val_dataset: "EmbeddingDataset | ShardedEmbeddingDataset | None" = None,
    output_dir: str | None = None,
    config: TrainingConfig | None = None,
    normalize_embeddings: bool = True,
    **kwargs,
) -> SAETrainer:
    """
    Train a Sparse Autoencoder on embeddings.

    Parameters
    ----------
    embeddings_path : str or Path or None
        Path to embeddings file (.npy or .pt) or sharded embedding directory.
        Ignored if dataset is provided.
    dataset : EmbeddingDataset or None
        Pre-configured dataset. If None, loads from embeddings_path.
    val_dataset : EmbeddingDataset or None
        Optional validation dataset for epoch-based validation and early stopping.
    output_dir : str or None
        Output directory for checkpoints. If None, uses config.output_dir.
    config : TrainingConfig or None
        Training configuration. If None, uses defaults.
    normalize_embeddings : bool, default=True
        Whether to L2-normalize embeddings loaded from embeddings_path. Set to
        False when training on embeddings produced by compute_and_save_embeddings
        with normalization enabled.
    **kwargs
        Additional arguments to override config

    Returns
    -------
    SAETrainer
        Trainer instance with trained model

    Examples
    --------
    >>> from saetopic.training import train_sae
    >>> trainer = train_sae(
    ...     embeddings_path="embeddings.npy",
    ...     input_dim=1024,
    ...     n_epochs=50,
    ...     output_dir="checkpoints/my_sae",
    ... )
    """
    # Create or update config
    if config is None:
        config = TrainingConfig(
            output_dir=output_dir or TrainingConfig.output_dir,
        )
    elif output_dir is not None:
        config.output_dir = output_dir

    # Update config with any provided kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    # Load dataset
    if dataset is None:
        if embeddings_path is None:
            raise ValueError("Either embeddings_path or dataset must be provided")

        from saetopic.training.data import EmbeddingDataset

        dataset = EmbeddingDataset.from_file(
            embeddings_path,
            normalize=normalize_embeddings,
            mmap_mode="r",
        )

    # Auto-detect and validate input_dim from dataset
    if hasattr(dataset, "embedding_dim"):
        actual_dim = dataset.embedding_dim
        if config.input_dim != actual_dim:
            print(f"[yellow]Warning: config.input_dim={config.input_dim} but detected {actual_dim} from dataset. Using {actual_dim}.[/yellow]")
            config.input_dim = actual_dim
        print(f"Detected embedding dim: {config.input_dim}")

    resume_checkpoint = _resolve_resume_checkpoint(config)
    if resume_checkpoint is not None:
        _apply_model_config_from_checkpoint(config, resume_checkpoint)
        if hasattr(dataset, "embedding_dim") and int(config.input_dim) != int(dataset.embedding_dim):
            raise ValueError(
                f"Resume checkpoint input_dim={config.input_dim} does not match "
                f"dataset embedding_dim={dataset.embedding_dim}"
            )
        config.resume_from_checkpoint = str(resume_checkpoint)

    # Create model
    from saetopic.sae.modules import create_sae

    model_kwargs: dict[str, Any] = {
        "decoder_bias": config.decoder_bias,
        "encoder_bias": config.encoder_bias,
        "normalization": config.normalization,
    }
    if config.architecture == "jumprelu":
        model_kwargs.update(
            {
                "bandwidth": config.bandwidth,
                "target_l0": config.target_l0,
            }
        )
    elif config.architecture == "matryoshka_batch_topk":
        model_kwargs.update(
            {
                "group_sizes": config.matryoshka_group_sizes,
                "group_fractions": config.matryoshka_group_fractions,
                "group_weights": config.matryoshka_group_weights,
                "active_groups": config.matryoshka_active_groups,
            }
        )
    elif config.architecture == "ort_batch_topk":
        model_kwargs.update(
            {
                "orthogonality_weight": config.orthogonality_weight,
                "orthogonality_chunk_size": config.orthogonality_chunk_size,
                "orthogonality_freq": config.orthogonality_freq,
            }
        )

    model = create_sae(
        input_dim=config.input_dim,
        architecture=config.architecture,
        n_features=config.n_features,
        expansion_factor=config.expansion_factor,
        top_k=config.top_k,
        **model_kwargs,
    )

    # Create trainer
    trainer = SAETrainer(model, config)

    # Train
    trainer.fit(dataset, val_dataset=val_dataset)

    return trainer


def _format_embedding_progress_remaining(
    dataset: Any,
    total_samples: int | None,
    n_saved: int,
    n_pending: int = 0,
) -> str:
    """Format remaining progress without mixing source rows and embeddings."""
    if total_samples is not None:
        return f"{max(total_samples - n_saved - n_pending, 0):,}"

    source_total = getattr(dataset, "source_total", None)
    source_rows_seen = getattr(dataset, "source_rows_seen", None)
    if isinstance(source_total, int) and isinstance(source_rows_seen, int):
        return f"{max(source_total - source_rows_seen, 0):,} source rows"

    return "unknown"


def _embedding_progress_total(dataset: Any, total_samples: int | None) -> int | None:
    """Choose the progress-bar total without mixing source and embedding units."""
    if total_samples is not None:
        return total_samples

    source_total = getattr(dataset, "source_total", None)
    if isinstance(source_total, int):
        return source_total

    return None


def _embedding_progress_completed(
    dataset: Any,
    total_samples: int | None,
    n_saved: int,
    n_pending: int = 0,
) -> int:
    """Choose the completed value in the same unit as the progress-bar total."""
    if total_samples is not None:
        return n_saved + n_pending

    source_rows_seen = getattr(dataset, "source_rows_seen", None)
    if isinstance(source_rows_seen, int):
        return source_rows_seen

    return n_saved + n_pending


def _valid_resume_cursor(dataset: Any, n_saved: int) -> dict[str, int] | None:
    """Return the latest dataset cursor that is safe for a manifest."""
    cursor = getattr(dataset, "safe_resume_cursor", None)
    if not isinstance(cursor, dict):
        return None

    try:
        source_rows_seen = int(cursor["source_rows_seen"])
        chunks_seen = int(cursor["chunks_seen"])
    except (KeyError, TypeError, ValueError):
        return None

    if source_rows_seen < 0 or chunks_seen < 0 or chunks_seen > n_saved:
        return None

    return {
        "source_rows_seen": source_rows_seen,
        "chunks_seen": chunks_seen,
    }


def compute_and_save_embeddings(
    dataset: StreamingEmbeddingDataset,
    output_path: str | Path,
    chunk_size: int = 10000,
    resume: bool = True,
) -> tuple[int, int]:
    """
    Compute embeddings from streaming dataset and save them to disk.

    This is useful for pre-computing embeddings once and reusing them
    for multiple training runs with different hyperparameters.

    Parameters
    ----------
    dataset : StreamingEmbeddingDataset
        Streaming dataset that computes embeddings on-the-fly
    output_path : str or Path
        Path to save embeddings. Paths ending in `.npy` use a single file.
        Paths without a file suffix use a sharded embedding directory.
    chunk_size : int, default=10000
        Number of embeddings to accumulate before saving to disk
    resume : bool, default=True
        Whether sharded embedding output should resume from
        `manifest.partial.json` when available. Only applies to sharded output.

    Returns
    -------
    n_embeddings : int
        Total number of embeddings computed
    embedding_dim : int
        Dimension of each embedding

    Examples
    --------
    >>> from saetopic.training import create_streaming_dataset, compute_and_save_embeddings
    >>> from sentence_transformers import SentenceTransformer
    >>>
    >>> embedder = SentenceTransformer("jinaai/jina-embeddings-v5-text-nano")
    >>> dataset = create_streaming_dataset(
    ...     dataset_name="HuggingFaceFW/finewiki",
    ...     embedder=embedder,
    ...     max_samples=100000,
    ... )
    >>> n, dim = compute_and_save_embeddings(dataset, "finewiki_embeddings")
    >>> print(f"Saved {n} embeddings of dimension {dim}")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if output_path.suffix not in {".npy", ".pt"}:
        return _compute_and_save_sharded_embeddings(
            dataset=dataset,
            output_dir=output_path,
            chunk_size=chunk_size,
            resume=resume,
        )

    n_total = 0
    dim = None
    batch_count = 0
    chunk_arrays: list[np.ndarray] = []
    chunk_paths: list[tuple[Path, int]] = []
    chunk_pending = 0
    chunk_index = 0

    print(f"Computing embeddings and saving to {output_path}")
    print("Note: This may take a while for large datasets...")

    total_samples = getattr(dataset, "max_samples", None)
    known_total = total_samples if isinstance(total_samples, int) and total_samples > 0 else None

    if known_total is not None:
        partial_path = output_path.with_name(
            f"{output_path.stem}.partial{output_path.suffix}"
        )
        if partial_path.exists():
            partial_path.unlink()

        print(f"Writing embeddings incrementally to {partial_path}")
        final_embeddings = None

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TextColumn("• {task.fields[remaining]} remaining"),
                TimeRemainingColumn(),
            ) as progress:
                task = progress.add_task(
                    "[cyan]Computing embeddings...",
                    total=known_total,
                    remaining=f"{known_total:,}",
                )

                for batch in dataset:
                    if dim is None:
                        dim = batch.shape[1]
                        final_embeddings = np.lib.format.open_memmap(
                            partial_path,
                            mode="w+",
                            dtype=np.float32,
                            shape=(known_total, dim),
                        )

                    if batch.device.type != "cpu":
                        batch = batch.cpu()

                    assert final_embeddings is not None
                    batch_np = batch.numpy().astype(np.float32, copy=False)
                    batch_end = n_total + batch_np.shape[0]
                    if batch_end > known_total:
                        raise ValueError(
                            "Streaming dataset produced more embeddings than "
                            f"max_samples ({known_total})"
                        )

                    final_embeddings[n_total:batch_end] = batch_np
                    n_total = batch_end
                    batch_count += 1

                    remaining = f"{max(known_total - n_total, 0):,}"
                    progress.update(
                        task,
                        completed=n_total,
                        remaining=remaining,
                    )

                    if batch_count % 10 == 0:
                        progress.refresh()
                        final_embeddings.flush()

                    if batch_count % 100 == 0:
                        import torch

                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

            if dim is None or final_embeddings is None:
                raise ValueError("No embeddings were produced from the dataset")

            final_embeddings.flush()
            del final_embeddings

            if n_total == known_total:
                partial_path.replace(output_path)
            else:
                print("Compacting final embeddings file...")
                partial_embeddings = np.load(partial_path, mmap_mode="r")
                compact_embeddings = np.lib.format.open_memmap(
                    output_path,
                    mode="w+",
                    dtype=np.float32,
                    shape=(n_total, dim),
                )
                for start in range(0, n_total, chunk_size):
                    end = min(start + chunk_size, n_total)
                    compact_embeddings[start:end] = partial_embeddings[start:end]
                compact_embeddings.flush()
                del compact_embeddings
                del partial_embeddings
                partial_path.unlink()

            print(
                f"[green]✓[/green] Saved {n_total} embeddings ({(n_total, dim)}) "
                f"to {output_path}"
            )
            return n_total, dim
        except Exception:
            if final_embeddings is not None:
                del final_embeddings
            raise

    with tempfile.TemporaryDirectory(
        prefix=f".{output_path.stem}_chunks_",
        dir=output_path.parent,
    ) as temp_dir_name:
        temp_dir = Path(temp_dir_name)

        def flush_chunk() -> None:
            nonlocal chunk_arrays, chunk_index, chunk_pending

            if not chunk_arrays:
                return

            chunk = np.concatenate(chunk_arrays, axis=0)
            chunk_path = temp_dir / f"chunk_{chunk_index:06d}.npy"
            np.save(chunk_path, chunk)
            chunk_paths.append((chunk_path, chunk.shape[0]))

            chunk_arrays = []
            chunk_pending = 0
            chunk_index += 1

        progress_total = _embedding_progress_total(dataset, total_samples)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("• {task.fields[remaining]} remaining"),
            TextColumn("• {task.fields[embedded]} embedded"),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task(
                "[cyan]Computing embeddings...",
                total=progress_total,
                remaining=_format_embedding_progress_remaining(
                    dataset,
                    total_samples,
                    n_total,
                ),
                embedded="0",
            )

            for batch in dataset:
                if dim is None:
                    dim = batch.shape[1]

                # CRITICAL: Move to CPU immediately to avoid GPU memory leak
                if batch.device.type != "cpu":
                    batch = batch.cpu()

                batch_np = batch.numpy()
                batch_start = 0
                while batch_start < batch_np.shape[0]:
                    remaining_space = chunk_size - chunk_pending
                    batch_end = min(batch_start + remaining_space, batch_np.shape[0])

                    chunk_arrays.append(batch_np[batch_start:batch_end].copy())
                    chunk_pending += batch_end - batch_start
                    batch_start = batch_end

                    if chunk_pending >= chunk_size:
                        flush_chunk()

                n_total += batch.shape[0]
                batch_count += 1

                # Progress update
                remaining = _format_embedding_progress_remaining(
                    dataset,
                    total_samples,
                    n_total,
                )

                progress.update(
                    task,
                    completed=_embedding_progress_completed(
                        dataset,
                        total_samples,
                        n_total,
                    ),
                    remaining=remaining,
                    embedded=f"{n_total:,}",
                )

                # Print progress periodically
                if batch_count % 10 == 0:
                    progress.refresh()

                # Clear GPU cache periodically to prevent memory buildup
                if batch_count % 100 == 0:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        flush_chunk()

        if dim is None:
            raise ValueError("No embeddings were produced from the dataset")

        print("Writing final embeddings file...")
        final_embeddings = np.lib.format.open_memmap(
            output_path,
            mode="w+",
            dtype=np.float32,
            shape=(n_total, dim),
        )

        offset = 0
        for chunk_path, n_chunk in chunk_paths:
            chunk = np.load(chunk_path, mmap_mode="r")
            final_embeddings[offset : offset + n_chunk] = chunk
            offset += n_chunk

        final_embeddings.flush()
        del final_embeddings

    print(f"[green]✓[/green] Saved {n_total} embeddings ({(n_total, dim)}) to {output_path}")

    return n_total, dim


def _compute_and_save_sharded_embeddings(
    dataset: StreamingEmbeddingDataset,
    output_dir: Path,
    chunk_size: int,
    resume: bool,
) -> tuple[int, int]:
    """Compute embeddings from a stream and save them as `.npy` shards."""
    manifest_path = output_dir / "manifest.json"
    partial_manifest_path = output_dir / "manifest.partial.json"

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        shape = manifest["shape"]
        print(f"Found completed sharded embeddings at {output_dir}")
        return int(shape[0]), int(shape[1])

    if output_dir.exists() and not resume:
        raise ValueError(
            f"Sharded embedding output directory already exists: {output_dir}. "
            "Choose a new path, remove the directory, or enable resume."
        )

    output_dir.mkdir(parents=True, exist_ok=resume)

    n_total = 0
    dim = None
    batch_count = 0
    shard_index = 0
    shard_pending = 0
    shard_arrays: list[np.ndarray] = []
    shards: list[dict[str, Any]] = []
    total_samples = getattr(dataset, "max_samples", None)
    skip_remaining = 0
    resume_cursor: dict[str, int] | None = _valid_resume_cursor(dataset, 0)

    def write_manifest(path: Path, *, completed: bool) -> None:
        nonlocal resume_cursor
        if dim is None:
            return

        current_cursor = _valid_resume_cursor(dataset, n_total)
        if current_cursor is not None:
            resume_cursor = current_cursor

        manifest = {
            "format": "saetopic.sharded_embeddings.v1",
            "dtype": "float32",
            "shape": [int(n_total), int(dim)],
            "shard_size": int(chunk_size),
            "completed": completed,
            "shards": shards,
        }
        if resume_cursor is not None:
            manifest["resume_cursor"] = resume_cursor
        temp_path = path.with_suffix(f"{path.suffix}.tmp")
        temp_path.write_text(json.dumps(manifest, indent=2))
        temp_path.replace(path)

    if partial_manifest_path.exists():
        if not resume:
            raise ValueError(
                f"Found incomplete sharded embeddings at {partial_manifest_path} "
                "but resume is disabled"
            )

        partial_manifest = json.loads(partial_manifest_path.read_text())
        if partial_manifest.get("format") != "saetopic.sharded_embeddings.v1":
            raise ValueError(
                f"Unknown sharded embedding format in {partial_manifest_path}"
            )
        if int(partial_manifest.get("shard_size", chunk_size)) != chunk_size:
            raise ValueError(
                "Cannot resume with a different chunk_size: "
                f"existing={partial_manifest.get('shard_size')}, requested={chunk_size}"
            )

        shards = list(partial_manifest["shards"])
        shape = partial_manifest["shape"]
        n_total = int(shape[0])
        dim = int(shape[1])
        shard_index = len(shards)
        manifest_cursor = partial_manifest.get("resume_cursor")
        if isinstance(manifest_cursor, dict):
            cursor_source_rows = int(manifest_cursor.get("source_rows_seen", 0))
            cursor_chunks = int(manifest_cursor.get("chunks_seen", 0))
            if cursor_source_rows < 0 or cursor_chunks < 0 or cursor_chunks > n_total:
                raise ValueError(
                    f"Invalid resume_cursor in {partial_manifest_path}: "
                    f"{manifest_cursor}"
                )
            resume_cursor = {
                "source_rows_seen": cursor_source_rows,
                "chunks_seen": cursor_chunks,
            }
            if hasattr(dataset, "skip_source_rows"):
                setattr(dataset, "skip_source_rows", cursor_source_rows)
            if hasattr(dataset, "skip_chunk_offset"):
                setattr(dataset, "skip_chunk_offset", cursor_chunks)
            skip_remaining = n_total - cursor_chunks
        else:
            skip_remaining = n_total

        for shard in shards:
            shard_path = output_dir / shard["file"]
            if not shard_path.exists():
                raise ValueError(f"Manifest references missing shard: {shard_path}")

        print(
            f"Resuming sharded embeddings at {output_dir}: "
            f"{n_total:,} rows across {len(shards):,} shards"
        )
        if resume_cursor is not None:
            print(
                "Resume cursor: "
                f"source_rows_seen={resume_cursor['source_rows_seen']:,}, "
                f"chunks_seen={resume_cursor['chunks_seen']:,}"
            )
    elif output_dir.exists():
        existing_shards = sorted(output_dir.glob("shard_*.npy"))
        if existing_shards:
            if not resume:
                raise ValueError(
                    f"Found existing embedding shards in {output_dir} but resume is disabled"
                )

            for expected_index, shard_path in enumerate(existing_shards):
                expected_name = f"shard_{expected_index:06d}.npy"
                if shard_path.name != expected_name:
                    raise ValueError(
                        "Cannot resume from non-contiguous shard files: "
                        f"expected {expected_name}, found {shard_path.name}"
                    )

                shard = np.load(shard_path, mmap_mode="r")
                if shard.ndim != 2:
                    raise ValueError(f"Shard must be 2D: {shard_path}")
                if dim is None:
                    dim = int(shard.shape[1])
                elif dim != int(shard.shape[1]):
                    raise ValueError(
                        f"Shard dimension mismatch in {shard_path}: "
                        f"expected {dim}, got {shard.shape[1]}"
                    )

                n_rows = int(shard.shape[0])
                n_total += n_rows
                shards.append(
                    {
                        "file": shard_path.name,
                        "shape": [n_rows, int(shard.shape[1])],
                    }
                )

            shard_index = len(shards)
            skip_remaining = n_total
            write_manifest(partial_manifest_path, completed=False)
            print(
                f"Recovered resumable sharded embeddings at {output_dir}: "
                f"{n_total:,} rows across {len(shards):,} shards"
            )
        elif any(output_dir.iterdir()):
            raise ValueError(
                f"Sharded embedding output directory is not empty and has no manifest: "
                f"{output_dir}"
            )

    print(f"Computing embeddings and saving shards to {output_dir}")
    print("Note: This may take a while for large datasets...")
    if skip_remaining and hasattr(dataset, "skip_samples"):
        setattr(dataset, "skip_samples", skip_remaining)
        if resume_cursor is not None:
            print(
                "Skipping "
                f"{skip_remaining:,} saved text chunks after resume cursor before encoding..."
            )
        else:
            print(
                f"Skipping {skip_remaining:,} previously saved text chunks before encoding..."
            )
        skip_remaining = 0

    def flush_shard() -> None:
        nonlocal n_total, shard_arrays, shard_index, shard_pending

        if not shard_arrays:
            return

        shard = np.concatenate(shard_arrays, axis=0).astype(np.float32, copy=False)
        shard_path = output_dir / f"shard_{shard_index:06d}.npy"
        np.save(shard_path, shard)
        shards.append(
            {
                "file": shard_path.name,
                "shape": [int(shard.shape[0]), int(shard.shape[1])],
            }
        )
        n_total += int(shard.shape[0])

        shard_arrays = []
        shard_pending = 0
        shard_index += 1
        write_manifest(partial_manifest_path, completed=False)

    progress_total = _embedding_progress_total(dataset, total_samples)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("• {task.fields[remaining]} remaining"),
        TextColumn("• {task.fields[embedded]} embedded"),
        TextColumn("• {task.fields[shards]} shards"),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task(
            "[cyan]Computing embeddings...",
            total=progress_total,
            remaining=_format_embedding_progress_remaining(
                dataset,
                total_samples,
                n_total,
            ),
            embedded=f"{n_total:,}",
            shards="0",
        )
        if n_total:
            remaining = _format_embedding_progress_remaining(
                dataset,
                total_samples,
                n_total,
            )
            progress.update(
                task,
                completed=_embedding_progress_completed(
                    dataset,
                    total_samples,
                    n_total,
                ),
                remaining=remaining,
                embedded=f"{n_total:,}",
                shards=f"{len(shards):,}",
            )

        for batch in dataset:
            if dim is None:
                dim = batch.shape[1]
            elif dim != batch.shape[1]:
                raise ValueError(
                    f"Embedding dimension changed from {dim} to {batch.shape[1]}"
                )

            if batch.device.type != "cpu":
                batch = batch.cpu()

            batch_np = batch.numpy().astype(np.float32, copy=False)
            if skip_remaining:
                if skip_remaining >= batch_np.shape[0]:
                    skip_remaining -= batch_np.shape[0]
                    continue

                batch_np = batch_np[skip_remaining:]
                skip_remaining = 0

            batch_start = 0
            while batch_start < batch_np.shape[0]:
                remaining_space = chunk_size - shard_pending
                batch_end = min(batch_start + remaining_space, batch_np.shape[0])

                shard_arrays.append(batch_np[batch_start:batch_end].copy())
                shard_pending += batch_end - batch_start
                batch_start = batch_end

                if shard_pending >= chunk_size:
                    flush_shard()

            batch_count += 1

            pending_rows = sum(array.shape[0] for array in shard_arrays)
            remaining = _format_embedding_progress_remaining(
                dataset,
                total_samples,
                n_total,
                pending_rows,
            )

            progress.update(
                task,
                completed=_embedding_progress_completed(
                    dataset,
                    total_samples,
                    n_total,
                    pending_rows,
                ),
                remaining=remaining,
                embedded=f"{n_total + pending_rows:,}",
                shards=f"{len(shards):,}",
            )

            if batch_count % 10 == 0:
                progress.refresh()

            if batch_count % 100 == 0:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    flush_shard()

    if dim is None:
        raise ValueError("No embeddings were produced from the dataset")

    write_manifest(manifest_path, completed=True)
    if partial_manifest_path.exists():
        partial_manifest_path.unlink()

    print(
        f"[green]✓[/green] Saved {n_total} embeddings ({(n_total, dim)}) "
        f"as {len(shards)} shards to {output_dir}"
    )

    return n_total, dim


def save_embeddings(
    streaming_dataset: StreamingEmbeddingDataset,
    output_path: str | Path,
    batch_size: int = 10000,
    show_progress: bool = True,
) -> tuple[int, int]:
    """
    Alias for compute_and_save_embeddings for backward compatibility.
    """
    return compute_and_save_embeddings(
        streaming_dataset, output_path, chunk_size=batch_size
    )
