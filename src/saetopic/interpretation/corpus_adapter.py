"""
CorpusAdapter: Maps SAE features to corpus-specific word distributions.

This module learns a feature-to-word matrix that adapts pretrained
topic atoms to a user's corpus vocabulary.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class CorpusAdapter:
    """
    Adapts SAE topic atoms to corpus-specific word distributions.

    This component learns a matrix B where B[f, w] represents the
    association between feature f and word w in the corpus vocabulary.

    Parameters
    ----------
    vocab_size : int
        Size of corpus vocabulary
    n_features : int
        Number of SAE features (topic atoms)
    idf_weighting : bool, default=True
        Whether to use IDF weighting in the learning objective
    """

    def __init__(
        self,
        vocab_size: int,
        n_features: int,
        idf_weighting: bool = True,
    ):
        self.vocab_size = vocab_size
        self.n_features = n_features
        self.idf_weighting = idf_weighting

        # Learned feature-to-word matrix
        self.feature_word_matrix_: np.ndarray | None = None

    def fit(
        self,
        docs: list[str],
        feature_activations: np.ndarray,
        vectorizer,
    ) -> "CorpusAdapter":
        """
        Learn feature-to-word matrix from corpus.

        Parameters
        ----------
        docs : list of str
            Corpus documents
        feature_activations : np.ndarray
            SAE activations (n_docs x n_features)
        vectorizer : vectorizer
            Fitted vectorizer with vocabulary

        Returns
        -------
        CorpusAdapter
            Fitted adapter instance
        """
        # TODO: Implement corpus adaptation
        raise NotImplementedError("CorpusAdapter.fit is not implemented yet")

    def transform(
        self,
        feature_activations: np.ndarray,
    ) -> np.ndarray:
        """
        Transform features to word distributions.

        Parameters
        ----------
        feature_activations : np.ndarray
            SAE activations (n_docs x n_features)

        Returns
        -------
        np.ndarray
            Word distributions (n_docs x vocab_size)
        """
        # TODO: Implement transform
        raise NotImplementedError("CorpusAdapter.transform is not implemented yet")

    def fit_transform(
        self,
        docs: list[str],
        feature_activations: np.ndarray,
        vectorizer,
    ) -> np.ndarray:
        """
        Fit and transform in one step.

        Parameters
        ----------
        docs : list of str
            Corpus documents
        feature_activations : np.ndarray
            SAE activations (n_docs x n_features)
        vectorizer : vectorizer
            Fitted vectorizer with vocabulary

        Returns
        -------
        np.ndarray
            Word distributions (n_docs x vocab_size)
        """
        return self.fit(docs, feature_activations, vectorizer).transform(
            feature_activations
        )
