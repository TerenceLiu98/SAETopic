"""
Visualization utilities for topics and documents.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def visualize_topics(
    topic_embeddings: np.ndarray,
    topic_labels: dict[int, str],
):
    """
    Create 2D visualization of topics.

    Parameters
    ----------
    topic_embeddings : np.ndarray
        Topic embeddings (n_topics x dim)
    topic_labels : dict
        Topic labels

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive 2D visualization
    """
    # TODO: Implement topic visualization (Week 4)
    raise NotImplementedError("visualize_topics will be implemented in Week 4")


def visualize_documents(
    doc_embeddings: np.ndarray,
    doc_topics: list[int],
    topic_labels: dict[int, str],
):
    """
    Create 2D visualization of documents colored by topic.

    Parameters
    ----------
    doc_embeddings : np.ndarray
        Document embeddings (n_docs x dim)
    doc_topics : list of int
        Topic assignments
    topic_labels : dict
        Topic labels

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive 2D visualization
    """
    # TODO: Implement document visualization (Week 4)
    raise NotImplementedError("visualize_documents will be implemented in Week 4")


def visualize_hierarchy(
    merge_tree: dict,
    topic_labels: dict[int, str],
):
    """
    Visualize topic merge hierarchy.

    Parameters
    ----------
    merge_tree : dict
        Tree structure of topic merges
    topic_labels : dict
        Topic labels

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive hierarchical visualization
    """
    # TODO: Implement hierarchy visualization (Week 4)
    raise NotImplementedError("visualize_hierarchy will be implemented in Week 4")
