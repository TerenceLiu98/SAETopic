"""
Feature-to-word interpretation utilities.

Helper functions for extracting and interpreting word associations
for SAE features.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def get_top_words_for_feature(
    feature_id: int,
    feature_word_matrix: np.ndarray,
    vocab: list[str],
    top_n: int = 10,
) -> list[tuple[str, float]]:
    """
    Get top words associated with a specific SAE feature.

    Parameters
    ----------
    feature_id : int
        SAE feature identifier
    feature_word_matrix : np.ndarray
        Feature-to-word association matrix (n_features x vocab_size)
    vocab : list of str
        Vocabulary list
    top_n : int, default=10
        Number of top words to return

    Returns
    -------
    list of (str, float)
        Top words with association scores
    """
    # TODO: Implement word extraction (Week 3)
    raise NotImplementedError("get_top_words_for_feature will be implemented in Week 3")
