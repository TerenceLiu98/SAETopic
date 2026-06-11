"""
Corpus interpretation module for feature-to-word mapping.

This package adapts pretrained SAE topic atoms to user-specific
corpora by learning feature-to-word distributions.
"""

from saetopic.interpretation.corpus_adapter import CorpusAdapter

__all__ = ["CorpusAdapter"]
