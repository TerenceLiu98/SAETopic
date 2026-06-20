"""
SAETopic: topic modeling with sparse autoencoder topic atoms.

The public package exposes a BERTopic-style ``SAETopicModel`` interface for
fitting topics, transforming documents, inspecting topic words, and changing
topic granularity without retraining the SAE. Training utilities live under
``saetopic.training`` and are exposed through the ``saetopic-train`` command.
"""

__version__ = "0.1.0"

from saetopic.interpretation import CorpusAdapter
from saetopic.merging import TopicMerger
from saetopic.model import SAETopicModel

__all__ = ["SAETopicModel", "CorpusAdapter", "TopicMerger"]
