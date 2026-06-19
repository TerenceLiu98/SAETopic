"""
SAE checkpoint loading utilities.

This module handles downloading and loading pretrained SAE checkpoints
from Hugging Face Hub or local paths.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _is_local_dir(target: str | Path) -> bool:
    p = Path(target)
    return p.is_dir() and (p / "config.json").exists()


def _resolve_checkpoint_dir(target: str | Path) -> Path:
    """
    Resolve a checkpoint directory from a local path or Hugging Face repo id.

    A local path must contain ``config.json`` and either ``model.safetensors``
    or ``model.pt``. A Hugging Face repo id is fetched via snapshot_download.
    """
    target = Path(target)
    if _is_local_dir(target):
        return target

    # Allow the "best" sub-directory of a checkpoint output dir
    if target.is_dir():
        for sub in ("best", "final"):
            if _is_local_dir(target / sub):
                return target / sub

    # Fall back to Hugging Face Hub
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "huggingface_hub is required to load SAE checkpoints from the Hub. "
            "Install it with `pip install huggingface_hub` or pass a local path."
        ) from exc

    repo_path = snapshot_download(
        repo_id=str(target),
        allow_patterns=["config.json", "model.safetensors", "model.pt"],
    )
    return Path(repo_path)


def _build_and_load_sae(checkpoint_dir: Path, config: dict[str, Any]):
    """Instantiate an SAE from config.json and load the saved weights."""
    import torch

    from saetopic.sae.modules import create_sae

    architecture = config.get("architecture", "batch_topk")
    model_kwargs = {
        "decoder_bias": config.get("decoder_bias", True),
        "encoder_bias": config.get("encoder_bias", False),
        "normalization": config.get("normalization"),
    }
    if architecture == "matryoshka_batch_topk":
        model_kwargs.update(
            {
                "group_sizes": config.get("matryoshka_group_sizes"),
                "group_fractions": config.get("matryoshka_group_fractions"),
                "group_weights": config.get("matryoshka_group_weights"),
                "active_groups": config.get("matryoshka_active_groups"),
            }
        )
    elif architecture == "ort_batch_topk":
        model_kwargs.update(
            {
                "orthogonality_weight": config.get("orthogonality_weight", 0.25),
                "orthogonality_chunk_size": config.get("orthogonality_chunk_size", 8192),
                "orthogonality_freq": config.get("orthogonality_freq", 10),
            }
        )

    model = create_sae(
        input_dim=config["input_dim"],
        architecture=architecture,
        n_features=config.get("n_features"),
        expansion_factor=config.get("expansion_factor", 32),
        top_k=config.get("top_k", 32),
        **model_kwargs,
    )

    safetensors_path = checkpoint_dir / "model.safetensors"
    if safetensors_path.exists():
        from safetensors.torch import load_file

        state_dict = load_file(str(safetensors_path))
    else:
        pt_path = checkpoint_dir / "model.pt"
        if not pt_path.exists():
            raise FileNotFoundError(
                f"No model weights found in {checkpoint_dir} "
                "(expected model.safetensors or model.pt)"
            )
        state_dict = torch.load(pt_path, map_location="cpu")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning("Missing keys when loading SAE: %s", missing)
    if unexpected:
        logger.warning("Unexpected keys when loading SAE: %s", unexpected)

    model.eval()
    logger.info(
        "Loaded SAE (%s, input_dim=%s, n_features=%s) from %s",
        config.get("architecture"),
        config.get("input_dim"),
        config.get("n_features"),
        checkpoint_dir,
    )
    return model


@dataclass
class SAECheckpoint:
    """
    Container for SAE checkpoint metadata and weights.

    Attributes
    ----------
    repo_id : str
        Hugging Face repository id or local path used for loading
    embedding_model : str
        Embedding model used for training
    embedding_task : str
        Task type for embeddings (e.g., "clustering")
    embedding_dim : int
        Dimension of embedding space (= SAE input_dim)
    sae_architecture : str
        SAE architecture type ("topk", "batch_topk", ...)
    expansion_factor : int
        Ratio of features to input dimension
    n_features : int
        Number of SAE features (topic atoms)
    top_k : int
        Number of features activated per input
    config : dict
        Full checkpoint configuration
    model : nn.Module or None
        The loaded SAE model (populated after from_pretrained)
    """

    repo_id: str
    embedding_model: str
    embedding_task: str
    embedding_dim: int
    sae_architecture: str
    expansion_factor: int
    n_features: int
    top_k: int
    config: dict[str, Any] = field(default_factory=dict)
    model: Any = None

    @classmethod
    def from_pretrained(cls, repo_id: str | Path, **kwargs) -> "SAECheckpoint":
        """
        Load checkpoint metadata and weights from a local path or HF Hub.

        Parameters
        ----------
        repo_id : str or Path
            Hugging Face model id (e.g., "saetopic/jina-v5-sae-small") or a
            local checkpoint directory containing ``config.json`` and weights.

        Returns
        -------
        SAECheckpoint
            Loaded checkpoint with metadata and model
        """
        checkpoint_dir = _resolve_checkpoint_dir(repo_id)
        config_path = checkpoint_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"config.json not found in {checkpoint_dir}")

        with open(config_path) as f:
            config = json.load(f)

        model = _build_and_load_sae(checkpoint_dir, config)

        return cls(
            repo_id=str(repo_id),
            embedding_model=kwargs.get("embedding_model", ""),
            embedding_task=kwargs.get("embedding_task", "clustering"),
            embedding_dim=config["input_dim"],
            sae_architecture=config.get("architecture", "batch_topk"),
            expansion_factor=config.get("expansion_factor", 32),
            n_features=config.get("n_features", config["input_dim"] * config.get("expansion_factor", 32)),
            top_k=config.get("top_k", 32),
            config=config,
            model=model,
        )

    def get_model(self):
        """Return the loaded SAE model."""
        return self.model


def load_sae_weights(repo_id: str | Path, local_cache: str | None = None):
    """
    Load an SAE model from a checkpoint (local path or HF Hub).

    Parameters
    ----------
    repo_id : str or Path
        Hugging Face model id or local checkpoint directory
    local_cache : str or None, default=None
        Optional local cache path (unused for local paths)

    Returns
    -------
    nn.Module
        Loaded SAE model ready for inference
    """
    del local_cache
    checkpoint = SAECheckpoint.from_pretrained(repo_id)
    return checkpoint.get_model()


def estimate_feature_weights(
    sae_model,
    embeddings: np.ndarray,
    batch_size: int = 512,
    device: str = "auto",
) -> np.ndarray:
    """
    Estimate per-feature corpus activation weights for topic merging.

    Convenience wrapper around :func:`extract_activations` returning the
    mean activation per feature across the corpus.

    Returns
    -------
    np.ndarray
        Mean activation per feature (n_features,)
    """
    from saetopic.sae.activations import extract_activations

    activations = extract_activations(embeddings, sae_model, batch_size=batch_size, device=device)
    return np.asarray(activations.mean(axis=0)).ravel()
