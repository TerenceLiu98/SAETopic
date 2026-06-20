"""Model serialization utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import sparse

if TYPE_CHECKING:
    from saetopic.model import SAETopicModel


_ARRAY_KEYS = [
    "embeddings_",
    "feature_activations_",
    "theta_avg_",
    "feature_word_matrix_",
    "topic_atom_clusters_",
    "topic_word_matrix_",
    "document_topic_matrix_",
    "topic_embeddings_",
    "word_embeddings_",
    "idf_",
    "ctfidf_",
]


def _json_safe(value: Any) -> Any:
    """Return a JSON-safe representation for constructor metadata."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return None


def _sae_architecture(sae: Any) -> str:
    from saetopic.sae.modules import (
        BatchTopKSAE,
        JumpReLUSAE,
        MatryoshkaBatchTopKSAE,
        OrtBatchTopKSAE,
        StandardSAE,
        TopKSAE,
    )

    if isinstance(sae, MatryoshkaBatchTopKSAE):
        return "matryoshka_batch_topk"
    if isinstance(sae, OrtBatchTopKSAE):
        return "ort_batch_topk"
    if isinstance(sae, BatchTopKSAE):
        return "batch_topk"
    if isinstance(sae, TopKSAE):
        return "topk"
    if isinstance(sae, JumpReLUSAE):
        return "jumprelu"
    if isinstance(sae, StandardSAE):
        return "standard"
    raise TypeError(f"Unsupported SAE type for serialization: {type(sae)!r}")


def _sae_config(model: "SAETopicModel") -> dict[str, Any]:
    sae = model.sae_
    if sae is None:
        raise RuntimeError("Model must have a loaded SAE before saving")

    config: dict[str, Any] = {
        "architecture": _sae_architecture(sae),
        "input_dim": int(getattr(sae, "input_dim", model.sae_input_dim_)),
        "n_features": int(getattr(sae, "n_features", model.sae_n_features_)),
        "expansion_factor": int(getattr(sae, "expansion_factor", 32)),
        "top_k": int(getattr(sae, "top_k", model.top_k_features)),
        "decoder_bias": bool(getattr(sae, "decoder_bias", True)),
        "encoder_bias": bool(getattr(sae, "encoder_bias", False)),
        "normalization": getattr(sae, "normalization", None),
    }
    if config["architecture"] == "matryoshka_batch_topk":
        config["matryoshka_group_sizes"] = [
            int(v) for v in getattr(sae, "group_sizes").detach().cpu().tolist()
        ]
        config["matryoshka_group_weights"] = [
            float(v) for v in getattr(sae, "group_weights").detach().cpu().tolist()
        ]
        config["matryoshka_active_groups"] = int(getattr(sae, "active_groups"))
    elif config["architecture"] == "ort_batch_topk":
        config["orthogonality_weight"] = float(getattr(sae, "orthogonality_weight", 0.25))
        config["orthogonality_chunk_size"] = int(
            getattr(sae, "orthogonality_chunk_size", 8192)
        )
        config["orthogonality_freq"] = int(getattr(sae, "orthogonality_freq", 10))
    return config


def _constructor_config(model: "SAETopicModel") -> dict[str, Any]:
    keys = [
        "embedding_model",
        "embedding_task",
        "merge_embedding_model",
        "n_topics",
        "top_k_features",
        "min_topic_size",
        "idf_weighting",
        "device",
        "random_state",
        "corpus_adapter_epochs",
        "corpus_adapter_batch_size",
        "activation_batch_size",
        "embedding_batch_size",
        "cluster_method",
        "sparsity_threshold",
        "vocabulary_size",
        "min_df",
        "max_df",
        "max_seq_length",
        "use_ctfidf",
        "drop_empty_topics",
        "stop_words",
        "theta_mode",
    ]
    return {key: _json_safe(getattr(model, key)) for key in keys}


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
        SAE weight serialization format ("safetensors", "torch").
    """
    if model.representation_ is None:
        raise RuntimeError("Model must be fitted before saving")
    if serialization not in {"safetensors", "torch"}:
        raise ValueError("serialization must be 'safetensors' or 'torch'")

    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)

    arrays = {}
    for key in _ARRAY_KEYS:
        value = getattr(model, key, None)
        if value is not None:
            arrays[key] = np.asarray(value)
    np.savez_compressed(path_obj / "arrays.npz", **arrays)

    if model.bow_ is not None:
        if sparse.issparse(model.bow_):
            sparse.save_npz(path_obj / "bow.npz", model.bow_.tocsr())
        else:
            np.save(path_obj / "bow.npy", np.asarray(model.bow_))

    (path_obj / "vocab.json").write_text(
        json.dumps(model.vocab_ or [], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (path_obj / "docs.json").write_text(
        json.dumps(model.docs_ or [], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    cluster_info = model.merger_.cluster_info_ if model.merger_ is not None else None
    topic_labels = (
        model.representation_.topic_labels_ if model.representation_ is not None else None
    )
    metadata = {
        "format": "saetopic.model.v1",
        "serialization": serialization,
        "constructor": _constructor_config(model),
        "sae_config": _sae_config(model),
        "cluster_info": cluster_info,
        "topic_labels": topic_labels,
    }
    (path_obj / "config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    sae_dir = path_obj / "sae"
    sae_dir.mkdir(exist_ok=True)
    (sae_dir / "config.json").write_text(
        json.dumps(metadata["sae_config"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    state_dict = {
        key: value.detach().cpu().contiguous()
        for key, value in model.sae_.state_dict().items()
    }
    if serialization == "safetensors":
        from safetensors.torch import save_file

        save_file(state_dict, str(sae_dir / "model.safetensors"))
    else:
        import torch

        torch.save(state_dict, sae_dir / "model.pt")


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
    from saetopic.merging import TopicMerger
    from saetopic.model import SAETopicModel
    from saetopic.representation import TopicRepresentation
    from saetopic.sae.loaders import SAECheckpoint

    path_obj = Path(path)
    metadata_path = path_obj / "config.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"SAETopic config not found: {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format") != "saetopic.model.v1":
        raise ValueError(f"Unsupported SAETopic model format: {metadata.get('format')!r}")

    constructor = dict(metadata.get("constructor") or {})
    constructor["sae_model"] = str(path_obj / "sae")
    model = SAETopicModel(**constructor)

    checkpoint = SAECheckpoint.from_pretrained(path_obj / "sae")
    model.sae_ = checkpoint.get_model()
    model.sae_input_dim_ = checkpoint.embedding_dim
    model.sae_n_features_ = checkpoint.n_features
    model.top_k_features = checkpoint.top_k
    model.sae_model = str(path_obj / "sae")

    arrays_path = path_obj / "arrays.npz"
    if not arrays_path.exists():
        raise FileNotFoundError(f"SAETopic arrays not found: {arrays_path}")
    with np.load(arrays_path, allow_pickle=False) as arrays:
        for key in _ARRAY_KEYS:
            setattr(model, key, arrays[key] if key in arrays else None)

    model.vocab_ = json.loads((path_obj / "vocab.json").read_text(encoding="utf-8"))
    model.docs_ = json.loads((path_obj / "docs.json").read_text(encoding="utf-8"))

    bow_npz = path_obj / "bow.npz"
    bow_npy = path_obj / "bow.npy"
    if bow_npz.exists():
        model.bow_ = sparse.load_npz(bow_npz)
    elif bow_npy.exists():
        model.bow_ = np.load(bow_npy, allow_pickle=False)
    else:
        model.bow_ = None

    model.merger_ = TopicMerger(
        n_topics=int(model.n_topics),
        method=model.cluster_method,
        random_state=model.random_state,
        sparsity_threshold=model.sparsity_threshold,
        embedding_model=model.merge_embedding_model,
        word_embeddings=model.word_embeddings_,
        allow_random_word_embeddings=model.merge_embedding_model is None,
    )
    model.merger_.feature_clusters_ = model.topic_atom_clusters_
    model.merger_.topic_word_matrix_ = model.topic_word_matrix_
    model.merger_.cluster_info_ = metadata.get("cluster_info")
    model.merger_._is_fitted = True

    if model.topic_word_matrix_ is not None:
        model.n_topics = int(model.topic_word_matrix_.shape[0])
    if model.document_topic_matrix_ is not None:
        model.topics_ = np.asarray(model.document_topic_matrix_).argmax(axis=1).tolist()

    display_matrix = model.ctfidf_ if model.ctfidf_ is not None else model.topic_word_matrix_
    model.representation_ = TopicRepresentation(
        display_matrix,
        model.vocab_ or [],
        model.document_topic_matrix_,
    )
    topic_labels = metadata.get("topic_labels")
    if topic_labels is not None:
        model.representation_.topic_labels_ = {int(k): v for k, v in topic_labels.items()}

    return model
