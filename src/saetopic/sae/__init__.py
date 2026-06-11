"""
Sparse Autoencoder (SAE) module for topic atom extraction.

This package provides SAE architecture definitions, checkpoint loading,
and activation extraction.
"""

from saetopic.sae.modules import BatchTopKSAE, JumpReLUSAE, StandardSAE, TopKSAE, create_sae

__all__ = [
    "BatchTopKSAE",
    "JumpReLUSAE",
    "StandardSAE",
    "TopKSAE",
    "create_sae",
]
