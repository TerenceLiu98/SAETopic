"""
SAETopic: BERTopic-style topic modeling with sparse autoencoder topic atoms.

This is an unofficial clean-room implementation inspired by:
"Sparse Autoencoders are Topic Models" (SAE-TM).
"""

__version__ = "0.1.0"

from saetopic.model import SAETopicModel

__all__ = ["SAETopicModel"]
