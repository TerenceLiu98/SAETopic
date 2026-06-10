"""
Training utilities for Sparse Autoencoders.

This module provides the main training loop and optimizer classes
for training SAE models on embedding datasets.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
import numpy as np

if TYPE_CHECKING:
    from torch import Tensor

    from saetopic.sae.modules import TopKSAE
    from saetopic.training.data import EmbeddingDataset, StreamingEmbeddingDataset


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
        SAE architecture ("topk", "batch_topk")
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
    learning_rate: float = 1e-4
    batch_size: int = 256
    n_epochs: int = 100
    device: str = "auto"
    architecture: str = "batch_topk"
    seed: int = 42
    save_frequency: int = 10
    log_frequency: int = 10
    recon_loss_weight: float = 1.0
    sparsity_loss_weight: float = 1.0
    aux_loss_weight: float = 0.001

    # Additional model parameters
    decoder_bias: bool = True
    encoder_bias: bool = False
    normalization: str | None = None

    # Output paths
    output_dir: str = "checkpoints/sae"
    checkpoint_name: str = "sae_checkpoint"

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
            "architecture": self.architecture,
            "seed": self.seed,
            "decoder_bias": self.decoder_bias,
            "encoder_bias": self.encoder_bias,
            "normalization": self.normalization,
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

    def update(self, losses: dict[str, float]) -> None:
        """Update training state with new losses."""
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

    Handles AdamW optimizer with optional learning rate scheduling.

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
    ):
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )

        self.use_scheduler = use_scheduler
        self.scheduler = None
        self.total_steps = total_steps if total_steps is not None else 100_000

        if use_scheduler:
            # Calculate pct_start (warmup percentage), capped at 0.3 (30%)
            pct_start = min(0.3, warmup_steps / self.total_steps) if warmup_steps else 0.1
            if pct_start >= 1.0:
                pct_start = 0.3  # Cap at 30% if calculation gives > 100%

            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=learning_rate,
                total_steps=self.total_steps,
                pct_start=pct_start,
                anneal_strategy="cos",
            )

    def step(self, loss: Tensor) -> None:
        """Perform optimizer step."""
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

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
        model: "TopKSAE",
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
        self.optimizer = None

        # Training state
        self.state = TrainingState()

        # Set random seed
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

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

        epoch_losses = {key: 0.0 for key in ["total", "reconstruction", "sparsity", "auxiliary"]}
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

            for batch in progress:
                batch = batch.to(self.device)

                # Forward pass
                x_recon, h, f, _ = self.model(batch)

                # Compute loss
                loss, losses = self.model.compute_loss(
                    batch,
                    x_recon,
                    h,
                    f,
                    recon_loss_weight=self.config.recon_loss_weight,
                    sparsity_loss_weight=self.config.sparsity_loss_weight,
                    aux_loss_weight=self.config.aux_loss_weight,
                )

                # Optimizer step
                self.optimizer.step(loss)

                # Update statistics
                n_batches += 1
                for key in epoch_losses:
                    epoch_losses[key] += losses[key].item()

                # Update progress bar
                avg_loss = epoch_losses["total"] / n_batches
                progress.update(task, advance=1, description=f"[cyan]Epoch {epoch}")
                progress.columns[-2].formatter = lambda _: f"• loss: {avg_loss:.6f}"

        # Average losses
        return {k: v / n_batches for k, v in epoch_losses.items()}

    def _create_optimizer(self, dataset: "EmbeddingDataset | StreamingEmbeddingDataset") -> None:
        """
        Create optimizer with scheduler based on dataset size.

        Parameters
        ----------
        dataset : EmbeddingDataset or StreamingEmbeddingDataset
            Training dataset
        """
        # Calculate total steps based on dataset type
        if hasattr(dataset, "__len__"):
            # Standard dataset - calculate exact steps
            n_samples = len(dataset)
            steps_per_epoch = max(1, n_samples // self.config.batch_size)
            total_steps = steps_per_epoch * self.config.n_epochs
        elif hasattr(dataset, "max_samples") and dataset.max_samples:
            # Streaming dataset with known max_samples
            steps_per_epoch = max(1, dataset.max_samples // self.config.batch_size)
            total_steps = steps_per_epoch * self.config.n_epochs
        else:
            # Streaming dataset without known size - use default
            total_steps = 100_000

        self.optimizer = SAEOptimizer(
            self.model,
            learning_rate=self.config.learning_rate,
            use_scheduler=True,
            total_steps=total_steps,
        )

        print(f"Total training steps: {total_steps}")

    def fit(
        self,
        dataset: "EmbeddingDataset | StreamingEmbeddingDataset",
        val_dataset: "EmbeddingDataset | None" = None,
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

        # Check if dataset is a streaming iterator
        is_streaming = hasattr(dataset, "__iter__") and not hasattr(dataset, "__len__")

        if is_streaming:
            return self._fit_streaming(dataset, val_dataset)
        else:
            return self._fit_standard(dataset, val_dataset)

    def _fit_standard(
        self,
        dataset: "EmbeddingDataset",
        val_dataset: "EmbeddingDataset | None" = None,
    ) -> TrainingState:
        """Train with standard PyTorch Dataset (uses DataLoader)."""
        train_loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

        for epoch in range(1, self.config.n_epochs + 1):
            # Train epoch
            epoch_losses = self.train_epoch(train_loader, epoch)

            # Update state
            self.state.epoch = epoch
            self.state.update(epoch_losses)

            # Print summary
            print(f"Epoch {epoch}/{self.config.n_epochs}")
            for key, value in epoch_losses.items():
                print(f"  {key}: {value:.6f}")

            # Save checkpoint
            if epoch % self.config.save_frequency == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}")

            # Update feature stats for BatchTopKSAE
            if hasattr(self.model, "update_feature_stats"):
                with torch.no_grad():
                    for batch in train_loader:
                        batch = batch.to(self.device)
                        _, _, f, _ = self.model(batch)
                        self.model.update_feature_stats(f)

        # Save final checkpoint
        self.save_checkpoint("final")

        # Save training config and state
        self.save_metadata()

        return self.state

    def _fit_streaming(
        self,
        streaming_dataset: "StreamingEmbeddingDataset",
        val_dataset: "EmbeddingDataset | None" = None,
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

        for epoch in range(1, self.config.n_epochs + 1):
            self.model.train()

            epoch_losses = {
                "total": 0.0,
                "reconstruction": 0.0,
                "sparsity": 0.0,
                "auxiliary": 0.0,
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
                        batch = batch.to(self.device)

                        # Forward pass
                        x_recon, h, f, _ = self.model(batch)

                        # Compute loss
                        loss, losses = self.model.compute_loss(
                            batch,
                            x_recon,
                            h,
                            f,
                            recon_loss_weight=self.config.recon_loss_weight,
                            sparsity_loss_weight=self.config.sparsity_loss_weight,
                            aux_loss_weight=self.config.aux_loss_weight,
                        )

                        # Optimizer step
                        self.optimizer.step(loss)

                        # Update statistics
                        n_batches += 1
                        for key in epoch_losses:
                            epoch_losses[key] += losses[key].item()

                        # Update progress bar
                        avg_loss = epoch_losses["total"] / n_batches
                        progress.update(task, advance=1, loss=f"{avg_loss:.6f}")

                        # Update feature stats for BatchTopKSAE
                        if hasattr(self.model, "update_feature_stats"):
                            self.model.update_feature_stats(f)

            # Average losses
            avg_losses = {k: v / n_batches for k, v in epoch_losses.items()}

            # Update state
            self.state.epoch = epoch
            self.state.update(avg_losses)

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
                {k: v.cpu().numpy() for k, v in state_dict.items()},
                checkpoint_dir / "model.safetensors",
            )
        except ImportError:
            torch.save(self.model.state_dict(), checkpoint_dir / "model.pt")

        # Save optimizer state
        torch.save(self.optimizer.state_dict(), checkpoint_dir / "optimizer.pt")

        # Save training state
        torch.save(self.state.__dict__, checkpoint_dir / "training_state.pt")

        print(f"Saved checkpoint to {checkpoint_dir}")

    def save_metadata(self) -> None:
        """Save training metadata (config, model card, etc.)."""
        # Save config
        config_path = self.output_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        # Save model card template
        model_card_path = self.output_dir / "model_card.md"
        self._create_model_card(model_card_path)

        # Save checksums
        self._save_checksums()

    def _create_model_card(self, path: Path) -> None:
        """Create model card markdown file."""
        n_features = self.config.n_features or self.config.input_dim * self.config.expansion_factor

        card = f"""# {self.config.checkpoint_name}

## Model Description

Sparse Autoencoder trained for topic modeling. This model learns {n_features} sparse features (topic atoms) from {self.input_dim}-dimensional embeddings.

## Training Data

- **Dataset**: {self.config.dataset_name or "See training script"}
- **License**: {self.config.dataset_license or "See dataset source"}

## Model Architecture

- **Architecture**: {self.config.architecture}
- **Input Dimension**: {self.config.input_dim}
- **Number of Features**: {n_features}
- **Expansion Factor**: {self.config.expansion_factor}
- **Top-K**: {self.config.top_k}

## Usage

```python
from saetopic import SAETopicModel

model = SAETopicModel.from_pretrained("{self.config.checkpoint_name}")
topics, probs = model.fit_transform(docs, n_topics=50)
```

## Training

- **Epochs**: {self.config.n_epochs}
- **Batch Size**: {self.config.batch_size}
- **Learning Rate**: {self.config.learning_rate}
- **Best Loss**: {self.state.best_loss:.6f}

## License

Apache-2.0

## Attribution

This is a clean-room implementation trained on permissively licensed data.
Inspired by ["Sparse Autoencoders are Topic Models"](https://github.com/ExplainableML/SAE-TM).
"""
        path.write_text(card)

    def _save_checksums(self) -> None:
        """Save SHA256 checksums for model files."""
        import hashlib

        checksums = []
        for file in self.output_dir.glob("*.safetensors"):
            sha256 = hashlib.sha256()
            sha256.update(file.read_bytes())
            checksums.append(f"{sha256.hexdigest()}  {file.name}")

        for file in self.output_dir.glob("*.pt"):
            sha256 = hashlib.sha256()
            sha256.update(file.read_bytes())
            checksums.append(f"{sha256.hexdigest()}  {file.name}")

        checksum_path = self.output_dir / "checksums.txt"
        checksum_path.write_text("\n".join(checksums))


def train_sae(
    embeddings_path: str | Path | None = None,
    dataset: "EmbeddingDataset | None" = None,
    output_dir: str = "checkpoints/sae",
    config: TrainingConfig | None = None,
    **kwargs,
) -> SAETrainer:
    """
    Train a Sparse Autoencoder on embeddings.

    Parameters
    ----------
    embeddings_path : str or Path or None
        Path to embeddings file (.npy or .pt). Ignored if dataset is provided.
    dataset : EmbeddingDataset or None
        Pre-configured dataset. If None, loads from embeddings_path.
    output_dir : str
        Output directory for checkpoints
    config : TrainingConfig or None
        Training configuration. If None, uses defaults.
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
        config = TrainingConfig(output_dir=output_dir)

    # Update config with any provided kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    # Load dataset
    if dataset is None:
        if embeddings_path is None:
            raise ValueError("Either embeddings_path or dataset must be provided")

        from saetopic.training.data import EmbeddingDataset

        dataset = EmbeddingDataset.from_file(embeddings_path)

    # Auto-detect and validate input_dim from dataset
    if hasattr(dataset, "embedding_dim"):
        actual_dim = dataset.embedding_dim
        if config.input_dim != actual_dim:
            print(f"[yellow]Warning: config.input_dim={config.input_dim} but detected {actual_dim} from dataset. Using {actual_dim}.[/yellow]")
            config.input_dim = actual_dim
        print(f"Detected embedding dim: {config.input_dim}")

    # Create model
    from saetopic.sae.modules import create_sae

    model = create_sae(
        input_dim=config.input_dim,
        architecture=config.architecture,
        n_features=config.n_features,
        expansion_factor=config.expansion_factor,
        top_k=config.top_k,
        decoder_bias=config.decoder_bias,
        encoder_bias=config.encoder_bias,
        normalization=config.normalization,
    )

    # Create trainer
    trainer = SAETrainer(model, config, output_dir)

    # Train
    trainer.fit(dataset)

    return trainer


def compute_and_save_embeddings(
    dataset: StreamingEmbeddingDataset,
    output_path: str | Path,
    chunk_size: int = 10000,
) -> tuple[int, int]:
    """
    Compute embeddings from streaming dataset and save to .npy file.

    This is useful for pre-computing embeddings once and reusing them
    for multiple training runs with different hyperparameters.

    Parameters
    ----------
    dataset : StreamingEmbeddingDataset
        Streaming dataset that computes embeddings on-the-fly
    output_path : str or Path
        Path to save embeddings (.npy or .pt)
    chunk_size : int, default=10000
        Number of embeddings to accumulate before saving to disk

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
    >>> n, dim = compute_and_save_embeddings(dataset, "finewiki_embeddings.npy")
    >>> print(f"Saved {n} embeddings of dimension {dim}")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_embeddings = []
    n_total = 0
    dim = None
    batch_count = 0

    print(f"Computing embeddings and saving to {output_path}")
    print("Note: This may take a while for large datasets...")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task(
            "[cyan]Computing embeddings...", total=None
        )

        for batch in dataset:
            if dim is None:
                dim = batch.shape[1]

            # CRITICAL: Move to CPU immediately to avoid GPU memory leak
            if batch.device.type != "cpu":
                batch = batch.cpu()

            all_embeddings.append(batch.numpy())
            n_total += batch.shape[0]
            batch_count += 1

            # Progress update
            progress.update(
                task,
                completed=n_total,
                total=getattr(dataset, "max_samples", None),
            )

            # Print progress periodically
            if batch_count % 10 == 0:
                progress.refresh()

            # Clear GPU cache periodically to prevent memory buildup
            if batch_count % 100 == 0:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # Concatenate all embeddings and save
    print("Concatenating embeddings...")
    final_embeddings = np.concatenate(all_embeddings, axis=0)
    np.save(output_path, final_embeddings)

    print(f"[green]✓[/green] Saved {n_total} embeddings ({final_embeddings.shape}) to {output_path}")

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
