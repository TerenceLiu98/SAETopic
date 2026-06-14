"""
Topic representation and labeling.

TopicRepresentation extracts top words / labels from a learned
topic-word matrix. It is a stateless helper around the matrices produced by
``TopicMerger`` and the vocab produced by ``CorpusVectorizer``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    from scipy import sparse


def top_words_from_distribution(
    distribution: np.ndarray,
    vocab: list[str],
    top_n: int = 10,
) -> list[tuple[str, float]]:
    """Return the top_n (word, score) pairs from a word distribution."""
    top_n = min(top_n, len(vocab))
    # argsort ascending → take the last top_n and reverse
    top_indices = np.argsort(distribution)[-top_n:][::-1]
    return [(vocab[i], float(distribution[i])) for i in top_indices]


def compute_ctfidf(
    document_topic_matrix: np.ndarray,
    bow,
) -> np.ndarray:
    """
    Compute class-based TF-IDF (BERTopic-style) topic-word scores.

    Unlike a raw aggregated emission distribution, c-TF-IDF down-weights
    words that are common across *many* topics (e.g. ``events``, ``born``,
    month names on Wikipedia) and boosts words distinctive to each topic —
    which is what makes topic words readable on noisy corpora.

    Uses the SOFT document-topic distribution: the expected word count of
    word ``w`` in topic ``c`` is ``sum_d p(d,c) * bow(d,w)``.

    Parameters
    ----------
    document_topic_matrix : np.ndarray
        Soft document-topic probabilities (n_docs x n_topics)
    bow : scipy.sparse.csr_matrix or np.ndarray
        Bag-of-words counts (n_docs x vocab_size)

    Returns
    -------
    np.ndarray
        c-TF-IDF scores (n_topics x vocab_size), float32
    """
    from scipy import sparse

    dtm = np.asarray(document_topic_matrix, dtype=np.float32)  # (n_docs x T)

    # Expected per-topic word counts: T = dtm.T @ bow  → (n_topics x V).
    # Compute as (bow.T @ dtm).T to keep the sparse side efficient.
    if sparse.issparse(bow):
        counts = (bow.T.astype(np.float32) @ dtm).T  # (n_topics x V) dense
    else:
        counts = (np.asarray(bow, dtype=np.float32).T @ dtm).T
    counts = np.asarray(counts, dtype=np.float32)

    # tf: normalize each topic's word counts by its total mass
    topic_totals = counts.sum(axis=1, keepdims=True)  # (T x 1)
    tf = counts / np.clip(topic_totals, 1e-12, None)

    # idf: rare-across-topics words score higher (BERTopic form)
    word_totals = counts.sum(axis=0)  # (V,)
    m = float(topic_totals.mean())  # average total mass per topic
    idf = np.log(1.0 + (1.0 + m / np.clip(word_totals, 1e-12, None)))  # (V,)

    ctfidf = tf * idf[None, :]
    return ctfidf.astype(np.float32)



class TopicRepresentation:
    """
    Generate and manage topic representations.

    Holds the learned topic-word matrix and vocabulary and exposes helpers to
    extract top words, build topic-info tables, and derive simple labels.

    Parameters
    ----------
    topic_word_matrix : np.ndarray
        Topic-word distributions (n_topics x vocab_size)
    vocab : list of str
        Vocabulary mapping column indices to words
    document_topic_matrix : np.ndarray or None, default=None
        Document-topic distributions (n_docs x n_topics), used for counts
    """

    def __init__(
        self,
        topic_word_matrix: np.ndarray,
        vocab: list[str],
        document_topic_matrix: np.ndarray | None = None,
    ):
        self.topic_word_matrix = np.asarray(topic_word_matrix)
        self.vocab = list(vocab)
        self.document_topic_matrix = (
            None if document_topic_matrix is None else np.asarray(document_topic_matrix)
        )
        self.topic_labels_: dict[int, str] | None = None

    def get_topic_words(
        self,
        topic_id: int,
        top_n: int = 10,
    ) -> list[tuple[str, float]]:
        """
        Get top words for a topic.

        Parameters
        ----------
        topic_id : int
            Topic identifier
        top_n : int, default=10
            Number of top words to return

        Returns
        -------
        list of (str, float)
            Top words with their scores
        """
        if topic_id < 0 or topic_id >= self.topic_word_matrix.shape[0]:
            raise ValueError(f"Invalid topic_id: {topic_id}")
        return top_words_from_distribution(
            self.topic_word_matrix[topic_id], self.vocab, top_n=top_n
        )

    def get_topic_info(self) -> "pd.DataFrame":
        """
        Get information about all topics.

        Returns
        -------
        pd.DataFrame
            Topic information with Topic, Count, and Name (top words)
        """
        import pandas as pd

        n_topics = self.topic_word_matrix.shape[0]
        if self.document_topic_matrix is not None:
            assignments = self.document_topic_matrix.argmax(axis=1)
            counts = np.bincount(assignments, minlength=n_topics)
        else:
            counts = np.zeros(n_topics, dtype=int)

        rows = []
        for topic_id in range(n_topics):
            words = [w for w, _ in self.get_topic_words(topic_id, top_n=10)]
            label = self.topic_labels_.get(topic_id) if self.topic_labels_ else None
            name = label if label else f"{topic_id}_" + "_".join(words[:4])
            rows.append(
                {
                    "Topic": topic_id,
                    "Count": int(counts[topic_id]),
                    "Name": name,
                    "Top_Words": ", ".join(words),
                }
            )
        return pd.DataFrame(rows)

    def generate_topic_labels(
        self,
        method: str = "words",
        llm=None,
    ) -> dict[int, str]:
        """
        Generate human-readable topic labels.

        Parameters
        ----------
        method : str, default="words"
            Labeling method. Only "words" is supported (joins top words).
        llm : Any, default=None
            Reserved for future LLM-based labeling.

        Returns
        -------
        dict
            Mapping from topic_id to label string
        """
        if method == "words":
            labels = {}
            for topic_id in range(self.topic_word_matrix.shape[0]):
                words = [w for w, _ in self.get_topic_words(topic_id, top_n=4)]
                labels[topic_id] = "_".join(words)
            self.topic_labels_ = labels
            return labels
        raise NotImplementedError(f"Labeling method {method!r} is not supported")
