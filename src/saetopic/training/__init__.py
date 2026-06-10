"""
Training infrastructure for Sparse Autoencoders.

This package provides utilities for training SAE models on embedding datasets.
"""

from saetopic.training.data import (
    EmbeddingDataset,
    StreamingEmbeddingDataset,
    create_streaming_dataset,
    load_embeddings_from_hf,
)
from saetopic.training.train_sae import SAEOptimizer, SAETrainer, train_sae

__all__ = [
    "SAETrainer",
    "SAEOptimizer",
    "train_sae",
    "EmbeddingDataset",
    "StreamingEmbeddingDataset",
    "create_streaming_dataset",
    "load_embeddings_from_hf",
]
