"""
Topic representation and labeling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


class TopicRepresentation:
    """
    Generate and manage topic representations.

    Handles topic words, labels, and representative documents.
    """

    def __init__(self):
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
            Top words with scores
        """
        # TODO: Implement word extraction (Week 3)
        raise NotImplementedError("get_topic_words will be implemented in Week 3")

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
            Labeling method ("words", "llm")
        llm : LLM, default=None
            LLM for label generation

        Returns
        -------
        dict
            Mapping from topic_id to label
        """
        # TODO: Implement label generation (Week 5)
        raise NotImplementedError("generate_topic_labels will be implemented in Week 5")

    def get_topic_info(self) -> pd.DataFrame:
        """
        Get information about all topics.

        Returns
        -------
        pd.DataFrame
            Topic information with counts, names, top words
        """
        # TODO: Implement topic info (Week 3)
        raise NotImplementedError("get_topic_info will be implemented in Week 3")
