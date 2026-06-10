"""
Evaluation metrics for topic models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def compute_diversity(
    topic_words: dict[int, list[tuple[str, float]]],
) -> float:
    """
    Compute topic diversity score.

    Measures how unique top words are across topics.

    Parameters
    ----------
    topic_words : dict
        Mapping from topic_id to list of (word, score)

    Returns
    -------
    float
        Diversity score (higher is more diverse)
    """
    # TODO: Implement diversity computation (Week 5)
    raise NotImplementedError("compute_diversity will be implemented in Week 5")


def compute_coherence(
    docs: list[str],
    topic_words: dict[int, list[tuple[str, float]]],
    metric: str = "npmi",
) -> dict[int, float]:
    """
    Compute topic coherence scores.

    Measures how semantically coherent top words are.

    Parameters
    ----------
    docs : list of str
        Corpus documents
    topic_words : dict
        Mapping from topic_id to list of (word, score)
    metric : str, default="npmi"
        Coherence metric ("npmi", "c_v", "u_mass")

    Returns
    -------
    dict
        Mapping from topic_id to coherence score
    """
    # TODO: Implement coherence computation (Week 5)
    raise NotImplementedError("compute_coherence will be implemented in Week 5")


def compute_stability(
    model,
    docs: list[str],
    n_runs: int = 5,
) -> float:
    """
    Compute topic stability across runs.

    Parameters
    ----------
    model : SAETopicModel
        Model to evaluate
    docs : list of str
        Test documents
    n_runs : int, default=5
        Number of runs with different random seeds

    Returns
    -------
    float
        Stability score (higher is more stable)
    """
    # TODO: Implement stability computation (Week 5)
    raise NotImplementedError("compute_stability will be implemented in Week 5")
