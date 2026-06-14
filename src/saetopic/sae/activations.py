"""
SAE activation extraction utilities.

This module handles computing sparse feature activations from
document embeddings using a trained SAE.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass


def _resolve_device(device: str):
    import torch

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def extract_activations(
    embeddings: np.ndarray,
    sae_model,
    batch_size: int = 512,
    device: str = "auto",
    sparse: bool = False,
) -> np.ndarray:
    """
    Extract SAE feature activations from document embeddings.

    Two modes (matching the SAE-TM framework's choice of θ):

    - ``sparse=False`` (default): use ``sae.encode()`` — the dense ReLU
      pre-activations. This is SAE-TM's θ for both the word-emission training
      and the document-topic matrix (all positive entries). Faithful default.
    - ``sparse=True``: reconstruct the true top-k sparse activation via
      ``activate()`` (only the fired features are non-zero per document).

    For Standard / JumpReLU SAEs the encoder output is already the feature
    vector and ``sparse`` is ignored.

    Parameters
    ----------
    embeddings : np.ndarray
        Document embeddings (n_docs x embedding_dim)
    sae_model : SAE model
        Trained sparse autoencoder with ``encode`` (and ``activate`` for TopK)
    batch_size : int, default=512
        Batch size for processing
    device : str, default="auto"
        Device for computation
    sparse : bool, default=False
        If True, return the top-k sparse activation; else dense ReLU encode.

    Returns
    -------
    np.ndarray
        Activations (n_docs x n_features), float32

    Notes
    -----
    The returned array is dense ``(n_docs x n_features)``. For very large
    corpora this can be memory-heavy; switch to a CSR representation when
    scaling beyond ~10k documents.
    """
    import torch

    dev = _resolve_device(device)
    sae_model = sae_model.to(dev).eval()
    # bf16 on GPU matches training dtype; CPU stays float32
    sae_dtype = torch.float32 if dev.type == "cpu" else torch.bfloat16

    n_features = getattr(sae_model, "n_features", None)
    if n_features is None:
        raise ValueError("sae_model has no n_features attribute")

    n_docs = len(embeddings)
    chunks: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, n_docs, batch_size):
            batch_np = np.asarray(embeddings[start : start + batch_size], dtype=np.float32)
            batch = torch.from_numpy(batch_np).to(dev, dtype=sae_dtype)

            if sparse and hasattr(sae_model, "activate"):
                # TopK / BatchTopK: reconstruct the sparse activation so only
                # the actually-fired features carry their activation values.
                h = sae_model.encode(batch)
                f, _ = sae_model.activate(h)
                act = f.float()
            else:
                # Dense ReLU encode (SAE-TM's θ); also handles Standard/JumpReLU.
                act = sae_model.encode(batch).float()

            chunks.append(act.cpu().numpy())

    return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
