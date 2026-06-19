"""
Sparse Autoencoder (SAE) module for topic atom extraction.

This package provides SAE architecture definitions, checkpoint loading,
and activation extraction.
"""

from saetopic.sae.activations import extract_activations
from saetopic.sae.loaders import SAECheckpoint, load_sae_weights
from saetopic.sae.modules import (
    BatchTopKSAE,
    JumpReLUSAE,
    MatryoshkaBatchTopKSAE,
    StandardSAE,
    TopKSAE,
    create_sae,
)

__all__ = [
    "BatchTopKSAE",
    "JumpReLUSAE",
    "MatryoshkaBatchTopKSAE",
    "StandardSAE",
    "TopKSAE",
    "create_sae",
    "extract_activations",
    "SAECheckpoint",
    "load_sae_weights",
]
