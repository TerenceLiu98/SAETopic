"""
SAETopic: sparse autoencoder topic-atom training.

The current package focuses on memory-aware SAE training. The topic inference
API is planned and exposed as stubs while it is under development.
"""

__version__ = "0.1.0"

from saetopic.model import SAETopicModel

__all__ = ["SAETopicModel"]
