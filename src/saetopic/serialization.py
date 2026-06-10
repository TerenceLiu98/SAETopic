"""
Model serialization utilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from saetopic.model import SAETopicModel


def save_model(
    model: "SAETopicModel",
    path: str,
    serialization: str = "safetensors",
) -> None:
    """
    Save a fitted SAETopic model to disk.

    Parameters
    ----------
    model : SAETopicModel
        Model to save
    path : str
        Directory path for saving
    serialization : str, default="safetensors"
        Serialization format ("safetensors", "pickle")
    """
    # TODO: Implement model saving (Week 4)
    raise NotImplementedError("save_model will be implemented in Week 4")


def load_model(
    path: str,
) -> "SAETopicModel":
    """
    Load a saved SAETopic model from disk.

    Parameters
    ----------
    path : str
        Directory path of saved model

    Returns
    -------
    SAETopicModel
        Loaded model instance
    """
    # TODO: Implement model loading (Week 4)
    raise NotImplementedError("load_model will be implemented in Week 4")
