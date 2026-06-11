"""
HuggingFace Hub utilities for uploading and downloading SAE checkpoints.

This module provides utilities for interacting with the HuggingFace Hub,
including downloading pretrained weights and uploading trained checkpoints.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    pass


def download_checkpoint(
    repo_id: str,
    local_cache: str | Path | None = None,
    filename: str = "model.safetensors",
    **kwargs,
) -> Path:
    """
    Download SAE checkpoint from HuggingFace Hub.

    Parameters
    ----------
    repo_id : str
        HuggingFace repository ID (e.g., "saetopic/jina-v5-sae-small")
    local_cache : str or Path or None
        Local cache directory. If None, uses HF Hub default cache.
    filename : str
        Name of the checkpoint file to download
    **kwargs
        Additional arguments passed to hf_hub_download

    Returns
    -------
    Path
        Local path to downloaded file

    Examples
    --------
    >>> from saetopic.hf_utils import download_checkpoint
    >>> path = download_checkpoint("saetopic/jina-v5-sae-small")
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required. Install with: pip install huggingface_hub"
        )

    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=local_cache,
        **kwargs,
    )

    return Path(local_path)


def download_config(
    repo_id: str,
    local_cache: str | Path | None = None,
    filename: str = "config.json",
) -> dict[str, Any]:
    """
    Download and parse SAE config from HuggingFace Hub.

    Parameters
    ----------
    repo_id : str
        HuggingFace repository ID
    local_cache : str or Path or None
        Local cache directory
    filename : str
        Config filename

    Returns
    -------
    dict
        Parsed configuration dictionary
    """
    config_path = download_checkpoint(repo_id, local_cache, filename)
    with open(config_path) as f:
        return cast(dict[str, Any], json.load(f))


def verify_checksum(
    file_path: str | Path,
    expected_checksum: str | None = None,
    checksums_file: str | Path | None = None,
) -> bool:
    """
    Verify SHA256 checksum of a file.

    Parameters
    ----------
    file_path : str or Path
        Path to file to verify
    expected_checksum : str or None
        Expected SHA256 checksum (hex string)
    checksums_file : str or Path or None
        Path to checksums file (format: "sha256  filename")

    Returns
    -------
    bool
        True if checksum matches
    """
    file_path = Path(file_path)

    # Compute file checksum
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    computed_checksum = sha256.hexdigest()

    # Get expected checksum
    if expected_checksum is None and checksums_file is not None:
        checksums_path = Path(checksums_file)
        with open(checksums_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2 and parts[1] == file_path.name:
                    expected_checksum = parts[0]
                    break

    if expected_checksum is None:
        raise ValueError("Either expected_checksum or checksums_file must be provided")

    return computed_checksum == expected_checksum


def upload_checkpoint(
    checkpoint_dir: str | Path,
    repo_id: str,
    create_repo: bool = False,
    private: bool = False,
    commit_message: str | None = None,
) -> None:
    """
    Upload SAE checkpoint to HuggingFace Hub.

    Parameters
    ----------
    checkpoint_dir : str or Path
        Directory containing checkpoint files
    repo_id : str
        HuggingFace repository ID (e.g., "saetopic/jina-v5-sae-small")
    create_repo : bool, default=False
        Whether to create the repository if it doesn't exist
    private : bool, default=False
        Whether the repository should be private
    commit_message : str or None
        Commit message for the upload

    Examples
    --------
    >>> from saetopic.hf_utils import upload_checkpoint
    >>> upload_checkpoint(
    ...     "checkpoints/my_sae/final",
    ...     "saetopic/my-sae",
    ...     create_repo=True,
    ... )
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise ImportError(
            "huggingface_hub is required. Install with: pip install huggingface_hub"
        )

    api = HfApi()
    checkpoint_dir = Path(checkpoint_dir)

    # Create repository if needed
    if create_repo:
        try:
            api.create_repo(repo_id=repo_id, private=private, repo_type="model")
            print(f"Created repository: {repo_id}")
        except Exception as e:
            if "already exists" not in str(e):
                raise

    # Upload folder
    if commit_message is None:
        commit_message = f"Upload SAE checkpoint from {checkpoint_dir.name}"

    api.upload_folder(
        folder_path=str(checkpoint_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
    )

    print(f"Uploaded checkpoint to {repo_id}")


def upload_file(
    file_path: str | Path,
    repo_id: str,
    path_in_repo: str,
    create_repo: bool = False,
    private: bool = False,
    commit_message: str | None = None,
) -> str:
    """
    Upload a single file to HuggingFace Hub.

    Parameters
    ----------
    file_path : str or Path
        Path to file to upload
    repo_id : str
        HuggingFace repository ID
    path_in_repo : str
        Destination path within the repository
    create_repo : bool, default=False
        Whether to create the repository if it doesn't exist
    private : bool, default=False
        Whether the repository should be private
    commit_message : str or None
        Commit message for the upload

    Returns
    -------
    str
        URL to the uploaded file
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise ImportError(
            "huggingface_hub is required. Install with: pip install huggingface_hub"
        )

    api = HfApi()
    file_path = Path(file_path)

    # Create repository if needed
    if create_repo:
        try:
            api.create_repo(repo_id=repo_id, private=private, repo_type="model")
            print(f"Created repository: {repo_id}")
        except Exception as e:
            if "already exists" not in str(e):
                raise

    if commit_message is None:
        commit_message = f"Upload {file_path.name}"

    api.upload_file(
        path_or_fileobj=str(file_path),
        repo_id=repo_id,
        path_in_repo=path_in_repo,
        repo_type="model",
        commit_message=commit_message,
    )

    return f"https://huggingface.co/{repo_id}/blob/main/{path_in_repo}"


def get_model_card(
    repo_id: str,
    local_cache: str | Path | None = None,
) -> str:
    """
    Fetch model card from HuggingFace Hub.

    Parameters
    ----------
    repo_id : str
        HuggingFace repository ID
    local_cache : str or Path or None
        Local cache directory

    Returns
    -------
    str
        Model card content as markdown string
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required. Install with: pip install huggingface_hub"
        )

    card_path = hf_hub_download(
        repo_id=repo_id,
        filename="model_card.md" if local_cache is None else "README.md",
        local_dir=local_cache,
    )

    with open(card_path) as f:
        return f.read()


def list_repo_files(
    repo_id: str,
) -> list[str]:
    """
    List files in a HuggingFace repository.

    Parameters
    ----------
    repo_id : str
        HuggingFace repository ID

    Returns
    -------
    list of str
        List of file paths in the repository
    """
    try:
        from huggingface_hub import list_repo_tree
    except ImportError:
        raise ImportError(
            "huggingface_hub is required. Install with: pip install huggingface_hub"
        )

    tree = list_repo_tree(repo_id, repo_type="model", token=None)

    files = []
    for item in tree:
        if getattr(item, "type", None) == "file":
            files.append(getattr(item, "path"))

    return files


def create_checkpoint_config(
    checkpoint_dir: str | Path,
    embedding_model: str,
    embedding_task: str,
    embedding_dim: int,
    sae_architecture: str,
    expansion_factor: int,
    n_features: int,
    top_k: int,
    training_corpus: str = "",
    training_corpus_license: str = "",
    **kwargs,
) -> None:
    """
    Create config.json for a checkpoint.

    Parameters
    ----------
    checkpoint_dir : str or Path
        Checkpoint directory
    embedding_model : str
        Embedding model name
    embedding_task : str
        Embedding task type
    embedding_dim : int
        Embedding dimension
    sae_architecture : str
        SAE architecture type
    expansion_factor : int
        Expansion factor
    n_features : int
        Number of features
    top_k : int
        Top-k activation
    training_corpus : str
        Training corpus description
    training_corpus_license : str
        Training corpus license
    **kwargs
        Additional config fields
    """
    config = {
        "embedding_model": embedding_model,
        "embedding_task": embedding_task,
        "embedding_dim": embedding_dim,
        "sae_architecture": sae_architecture,
        "expansion_factor": expansion_factor,
        "n_features": n_features,
        "top_k": top_k,
        "normalization": kwargs.get("normalization", None),
        "license": "Apache-2.0",
        "training_corpus": training_corpus,
        "training_corpus_license": training_corpus_license,
        "created_by": "SAETopic contributors",
        "source": "clean-room trained checkpoint",
        **kwargs,
    }

    config_path = Path(checkpoint_dir) / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
