"""
Command-line interface for training SAE models.

Usage:
    python -m saetopic.training.cli train --embeddings path/to/embeddings.npy --output checkpoints/sae
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    """Main CLI entry point for training."""
    parser = argparse.ArgumentParser(
        description="Train a Sparse Autoencoder for topic modeling"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # train command
    train_parser = subparsers.add_parser("train", help="Train an SAE model")

    # Input arguments
    train_parser.add_argument(
        "--embeddings",
        type=str,
        required=True,
        help="Path to embeddings file (.npy or .pt)",
    )
    train_parser.add_argument(
        "--dataset-name",
        type=str,
        default="",
        help="Dataset name for model card",
    )
    train_parser.add_argument(
        "--dataset-license",
        type=str,
        default="",
        help="Dataset license for model card",
    )

    # Model arguments
    train_parser.add_argument(
        "--input-dim",
        type=int,
        default=None,
        help="Input embedding dimension (auto-detected from file if not provided)",
    )
    train_parser.add_argument(
        "--n-features",
        type=int,
        default=None,
        help="Number of SAE features (default: input_dim * expansion_factor)",
    )
    train_parser.add_argument(
        "--expansion-factor",
        type=int,
        default=32,
        help="Expansion factor for n_features calculation",
    )
    train_parser.add_argument(
        "--top-k",
        type=int,
        default=32,
        help="Number of features to activate per input",
    )
    train_parser.add_argument(
        "--architecture",
        type=str,
        default="batch_topk",
        choices=["topk", "batch_topk"],
        help="SAE architecture type",
    )
    train_parser.add_argument(
        "--decoder-bias",
        action="store_true",
        default=True,
        help="Use bias in decoder",
    )
    train_parser.add_argument(
        "--no-decoder-bias",
        dest="decoder_bias",
        action="store_false",
        help="Don't use bias in decoder",
    )
    train_parser.add_argument(
        "--encoder-bias",
        action="store_true",
        help="Use bias in encoder",
    )
    train_parser.add_argument(
        "--normalization",
        type=str,
        default=None,
        choices=["batch_norm", "layer_norm", None],
        help="Normalization method",
    )

    # Training arguments
    train_parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Training batch size",
    )
    train_parser.add_argument(
        "--n-epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    train_parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device for training",
    )
    train_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    train_parser.add_argument(
        "--save-frequency",
        type=int,
        default=10,
        help="Save checkpoint every N epochs",
    )

    # Loss weights
    train_parser.add_argument(
        "--recon-loss-weight",
        type=float,
        default=1.0,
        help="Weight for reconstruction loss",
    )
    train_parser.add_argument(
        "--sparsity-loss-weight",
        type=float,
        default=1.0,
        help="Weight for sparsity loss",
    )
    train_parser.add_argument(
        "--aux-loss-weight",
        type=float,
        default=0.001,
        help="Weight for auxiliary loss",
    )

    # Output arguments
    train_parser.add_argument(
        "--output",
        type=str,
        default="checkpoints/sae",
        help="Output directory for checkpoints",
    )
    train_parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="sae_checkpoint",
        help="Checkpoint name (for model card)",
    )

    # Upload arguments
    train_parser.add_argument(
        "--upload-to-hf",
        type=str,
        default=None,
        help="Upload to HuggingFace Hub (provide repo_id)",
    )
    train_parser.add_argument(
        "--create-repo",
        action="store_true",
        help="Create HF repository if it doesn't exist",
    )

    args = parser.parse_args()

    if args.command == "train":
        train_sae_from_args(args)
    else:
        parser.print_help()


def train_sae_from_args(args: argparse.Namespace) -> None:
    """Train SAE from CLI arguments."""
    from saetopic.training import train_sae
    from saetopic.training.train_sae import TrainingConfig

    # Load dataset first to detect input_dim
    from saetopic.training.data import EmbeddingDataset

    dataset = EmbeddingDataset.from_file(args.embeddings)

    # Auto-detect input_dim if not provided
    input_dim = args.input_dim or dataset.embedding_dim

    # Create config
    config = TrainingConfig(
        input_dim=input_dim,
        n_features=args.n_features,
        expansion_factor=args.expansion_factor,
        top_k=args.top_k,
        architecture=args.architecture,
        decoder_bias=args.decoder_bias,
        encoder_bias=args.encoder_bias,
        normalization=args.normalization,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        device=args.device,
        seed=args.seed,
        save_frequency=args.save_frequency,
        recon_loss_weight=args.recon_loss_weight,
        sparsity_loss_weight=args.sparsity_loss_weight,
        aux_loss_weight=args.aux_loss_weight,
        output_dir=args.output,
        checkpoint_name=args.checkpoint_name,
        dataset_name=args.dataset_name,
        dataset_license=args.dataset_license,
    )

    print(f"Training SAE with config:")
    print(f"  Input dim: {input_dim}")
    print(f"  Features: {config.n_features or input_dim * config.expansion_factor}")
    print(f"  Top-K: {config.top_k}")
    print(f"  Architecture: {config.architecture}")
    print(f"  Epochs: {config.n_epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Output: {config.output_dir}")

    # Train
    trainer = train_sae(
        dataset=dataset,
        config=config,
    )

    # Upload to HF if requested
    if args.upload_to_hf:
        from saetopic.hf_utils import upload_checkpoint

        upload_checkpoint(
            f"{args.output}/final",
            args.upload_to_hf,
            create_repo=args.create_repo,
        )


if __name__ == "__main__":
    main()
