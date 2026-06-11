"""
Command-line interface for training SAE models.

Usage:
    python -m saetopic.training.cli embed --dataset-name HuggingFaceFW/finewiki --output data/finewiki_embeddings
    python -m saetopic.training.cli train --embeddings path/to/embeddings --output checkpoints/sae
    python -m saetopic.training.cli upload --checkpoint-dir checkpoints/sae/final --repo-id your-org/sae
"""

from __future__ import annotations

import argparse
from typing import Literal


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
        help="Path to embeddings file (.npy or .pt) or sharded embedding directory",
    )
    train_parser.add_argument(
        "--no-mmap",
        action="store_true",
        help="Load .npy embeddings fully into RAM instead of memory-mapping them",
    )
    train_parser.add_argument(
        "--no-normalize-embeddings",
        action="store_true",
        help="Do not L2-normalize loaded embeddings before SAE training",
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
    train_parser.add_argument(
        "--private",
        action="store_true",
        help="Create/upload to a private HuggingFace Hub repository",
    )

    # embed command
    embed_parser = subparsers.add_parser(
        "embed",
        help="Compute and save embeddings from a HuggingFace text dataset",
    )
    embed_parser.add_argument(
        "--dataset-name",
        type=str,
        default="HuggingFaceFW/finewiki",
        help="HuggingFace dataset name",
    )
    embed_parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="Optional HuggingFace dataset subset/config name",
    )
    embed_parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split",
    )
    embed_parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Column containing text to embed",
    )
    embed_parser.add_argument(
        "--model",
        type=str,
        default="jinaai/jina-embeddings-v5-text-small",
        help="SentenceTransformer model name",
    )
    embed_parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path. Use a directory/no extension for sharded embeddings, or .npy for a single file",
    )
    embed_parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of text chunks to embed",
    )
    embed_parser.add_argument(
        "--buffer-size",
        type=int,
        default=1000,
        help="Text shuffle buffer size",
    )
    embed_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for streaming buffer shuffling",
    )
    embed_parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=64,
        help="Text chunks to accumulate before yielding embeddings",
    )
    embed_parser.add_argument(
        "--encode-batch-size",
        type=int,
        default=8,
        help="Internal SentenceTransformer encode batch size",
    )
    embed_parser.add_argument(
        "--encode-device",
        action="append",
        default=None,
        help="Device for encode(); pass multiple times for multi-GPU, e.g. --encode-device cuda:0 --encode-device cuda:1",
    )
    embed_parser.add_argument(
        "--auto-multi-gpu",
        action="store_true",
        help="Use all visible CUDA devices for SentenceTransformers encode()",
    )
    embed_parser.add_argument(
        "--encode-chunk-size",
        type=int,
        default=None,
        help="SentenceTransformers multi-process work distribution chunk size",
    )
    embed_parser.add_argument(
        "--text-chunk-size",
        type=int,
        default=512,
        help="Tokenizer tokens per text chunk; set 0 to disable long-text chunking",
    )
    embed_parser.add_argument(
        "--text-chunk-overlap",
        type=int,
        default=32,
        help="Tokenizer token overlap between adjacent text chunks",
    )
    embed_parser.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
        help="SentenceTransformer max sequence length",
    )
    embed_parser.add_argument(
        "--truncate-dim",
        type=int,
        default=None,
        help="Optional Matryoshka embedding dimension truncation",
    )
    embed_parser.add_argument(
        "--save-chunk-size",
        type=int,
        default=10000,
        help="Embeddings per temporary save chunk",
    )
    embed_parser.add_argument(
        "--no-normalize-embeddings",
        action="store_true",
        help="Save raw embedder outputs instead of L2-normalized embeddings",
    )
    embed_parser.add_argument(
        "--task",
        type=str,
        default="clustering",
        help="Task argument passed to compatible embedding models such as Jina",
    )
    embed_parser.add_argument(
        "--no-bf16",
        action="store_true",
        help="Do not request bfloat16 model weights on CUDA",
    )
    embed_parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=True,
        help="Trust remote code when loading the SentenceTransformer model",
    )
    embed_parser.add_argument(
        "--no-trust-remote-code",
        dest="trust_remote_code",
        action="store_false",
        help="Disable trust_remote_code when loading the model",
    )

    # upload command
    upload_parser = subparsers.add_parser(
        "upload",
        help="Upload an existing self-contained SAE checkpoint to HuggingFace Hub",
    )
    upload_parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Checkpoint directory to upload, e.g. checkpoints/my-sae/final",
    )
    upload_parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HuggingFace repository ID, e.g. your-org/my-sae",
    )
    upload_parser.add_argument(
        "--create-repo",
        action="store_true",
        help="Create HF repository if it doesn't exist",
    )
    upload_parser.add_argument(
        "--private",
        action="store_true",
        help="Create/upload to a private HuggingFace Hub repository",
    )
    upload_parser.add_argument(
        "--commit-message",
        type=str,
        default=None,
        help="Optional HuggingFace Hub commit message",
    )

    args = parser.parse_args()

    if args.command == "train":
        train_sae_from_args(args)
    elif args.command == "embed":
        compute_embeddings_from_args(args)
    elif args.command == "upload":
        upload_checkpoint_from_args(args)
    else:
        parser.error("a command is required: choose from train, embed, upload")


def train_sae_from_args(args: argparse.Namespace) -> None:
    """Train SAE from CLI arguments."""
    from saetopic.training import train_sae

    # Load dataset first to detect input_dim
    from saetopic.training.data import EmbeddingDataset
    from saetopic.training.train_sae import TrainingConfig

    mmap_mode: Literal["r"] | None = None if args.no_mmap else "r"
    dataset = EmbeddingDataset.from_file(
        args.embeddings,
        normalize=not args.no_normalize_embeddings,
        mmap_mode=mmap_mode,
    )

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

    print("Training SAE with config:")
    print(f"  Input dim: {input_dim}")
    print(f"  Features: {config.n_features or input_dim * config.expansion_factor}")
    print(f"  Top-K: {config.top_k}")
    print(f"  Architecture: {config.architecture}")
    print(f"  Epochs: {config.n_epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Output: {config.output_dir}")

    # Train
    train_sae(
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
            private=args.private,
        )


def compute_embeddings_from_args(args: argparse.Namespace) -> None:
    """Compute embeddings from a HuggingFace dataset from CLI arguments."""
    import torch
    from sentence_transformers import SentenceTransformer

    from saetopic.training import compute_and_save_embeddings, create_streaming_dataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encode_device = args.encode_device
    if args.auto_multi_gpu and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        encode_device = [f"cuda:{i}" for i in range(torch.cuda.device_count())]

    model_kwargs = {}
    if device.type == "cuda" and not args.no_bf16:
        model_kwargs["dtype"] = torch.bfloat16

    sentence_transformer_kwargs = {
        "trust_remote_code": args.trust_remote_code,
        "device": device,
    }
    if model_kwargs:
        sentence_transformer_kwargs["model_kwargs"] = model_kwargs
    if args.truncate_dim is not None:
        sentence_transformer_kwargs["truncate_dim"] = args.truncate_dim

    embedder = SentenceTransformer(args.model, **sentence_transformer_kwargs)
    if args.max_seq_length:
        embedder.max_seq_length = args.max_seq_length

    text_chunk_size = args.text_chunk_size or None
    streaming_dataset = create_streaming_dataset(
        dataset_name=args.dataset_name,
        subset=args.subset,
        split=args.split,
        embedder=embedder,
        text_column=args.text_column,
        buffer_size=args.buffer_size,
        embedding_batch_size=args.embedding_batch_size,
        encode_batch_size=args.encode_batch_size,
        encode_device=encode_device,
        encode_chunk_size=args.encode_chunk_size,
        text_chunk_size=text_chunk_size,
        text_chunk_overlap=args.text_chunk_overlap,
        normalize=not args.no_normalize_embeddings,
        seed=args.seed,
        max_samples=args.max_samples,
        task=args.task,
    )

    n_embeddings, embedding_dim = compute_and_save_embeddings(
        dataset=streaming_dataset,
        output_path=args.output,
        chunk_size=args.save_chunk_size,
    )
    print(f"Saved {n_embeddings} embeddings of dimension {embedding_dim} to {args.output}")


def upload_checkpoint_from_args(args: argparse.Namespace) -> None:
    """Upload an existing SAE checkpoint from CLI arguments."""
    from saetopic.hf_utils import upload_checkpoint

    upload_checkpoint(
        args.checkpoint_dir,
        args.repo_id,
        create_repo=args.create_repo,
        private=args.private,
        commit_message=args.commit_message,
    )


if __name__ == "__main__":
    main()
