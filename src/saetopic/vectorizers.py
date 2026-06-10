"""
Vectorizer for corpus vocabulary and bag-of-words construction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from scipy import sparse


class CorpusVectorizer:
    """
    Build corpus vocabulary and bag-of-words representations.

    Parameters
    ----------
    vocabulary_size : int or None, default=None
        Maximum vocabulary size (None for unlimited)
    min_df : int, default=2
        Minimum document frequency for vocabulary terms
    max_df : float, default=0.95
        Maximum document frequency (ratio) for vocabulary terms
    idf_weighting : bool, default=True
        Whether to compute IDF weights
    """

    def __init__(
        self,
        vocabulary_size: int | None = None,
        min_df: int = 2,
        max_df: float = 0.95,
        idf_weighting: bool = True,
    ):
        self.vocabulary_size = vocabulary_size
        self.min_df = min_df
        self.max_df = max_df
        self.idf_weighting = idf_weighting

        # Fitted attributes
        self.vocab_: list[str] | None = None
        self.idf_: np.ndarray | None = None

    def fit(
        self,
        docs: list[str],
    ) -> "CorpusVectorizer":
        """
        Build vocabulary from documents.

        Parameters
        ----------
        docs : list of str
            Corpus documents

        Returns
        -------
        CorpusVectorizer
            Fitted vectorizer instance
        """
        # TODO: Implement vocabulary building (Week 3)
        raise NotImplementedError("CorpusVectorizer.fit will be implemented in Week 3")

    def transform(
        self,
        docs: list[str],
    ) -> sparse.csr_matrix:
        """
        Transform documents to bag-of-words.

        Parameters
        ----------
        docs : list of str
            Documents to vectorize

        Returns
        -------
        sparse.csr_matrix
            Bag-of-words matrix (n_docs x vocab_size)
        """
        # TODO: Implement BoW transformation (Week 3)
        raise NotImplementedError("CorpusVectorizer.transform will be implemented in Week 3")

    def fit_transform(
        self,
        docs: list[str],
    ) -> sparse.csr_matrix:
        """
        Fit and transform in one step.

        Parameters
        ----------
        docs : list of str
            Corpus documents

        Returns
        -------
        sparse.csr_matrix
            Bag-of-words matrix (n_docs x vocab_size)
        """
        return self.fit(docs).transform(docs)
