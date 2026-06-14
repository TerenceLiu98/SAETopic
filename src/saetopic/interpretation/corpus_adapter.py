"""
CorpusAdapter: Maps SAE features to corpus-specific word distributions.

This module learns a feature-to-word matrix that adapts pretrained
topic atoms to a user's corpus vocabulary, following the SAE-TM framework.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.sparse import csr_matrix, vstack
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


def sparsify_and_renormalize(
    input_tensor: torch.Tensor,
    tau: float = 0.9,
) -> torch.Tensor:
    """
    Transform a tensor by keeping only the top-n entries per row and renormalizing.

    For each row, this keeps the minimum number of largest entries whose
    cumulative sum exceeds tau, then renormalizes to sum to 1.

    Args:
        input_tensor: A K x V tensor with non-negative entries summing to 1 per row
        tau: Cumulative sum threshold (default: 0.9)

    Returns:
        Transformed K x V tensor with sparse, renormalized rows
    """
    if not isinstance(input_tensor, torch.Tensor):
        raise TypeError(f"Input must be a torch.Tensor, got {type(input_tensor)}")
    if input_tensor.dim() != 2:
        raise ValueError(f"Input must be 2D, got {input_tensor.dim()}D")

    K, V = input_tensor.shape
    device = input_tensor.device

    # Sort values in descending order along each row
    sorted_values, sorted_indices = torch.sort(input_tensor, dim=-1, descending=True)

    # Cumulative sum of sorted values
    cumulative_sums = torch.cumsum(sorted_values, dim=-1)

    # Find n: minimum elements whose sum exceeds tau
    n_elements = torch.argmax((cumulative_sums > tau).int(), dim=-1) + 1

    # Handle rows that never exceed tau (keep all elements)
    never_exceeds_tau = (cumulative_sums > tau).sum(dim=-1) == 0
    n_elements[never_exceeds_tau] = V

    # Create mask for top-n elements in sorted positions
    arange_tensor = torch.arange(V, device=device).expand(K, -1)
    mask_sorted = arange_tensor < n_elements.unsqueeze(-1)

    # Un-sort mask to match original tensor structure
    final_mask = torch.zeros_like(input_tensor, dtype=torch.bool)
    final_mask.scatter_(dim=1, index=sorted_indices, src=mask_sorted)

    # Apply mask and renormalize
    transformed = input_tensor * final_mask
    row_sums = transformed.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    return transformed / row_sums


class TopicWordModel(nn.Module):
    """
    Neural module for learning feature-to-word associations.

    Implements the SAE-TM interpretation model with:
    - B_logits: Learnable feature-to-word emission matrix
    - bg_logits: Background word distribution
    - pi: Mixture weight between topic and background

    Forward pass computes:
        p(word|theta) = (1-pi) * softmax(theta @ softmax(B)) + pi * softmax(bg)

    Supports subset computation for efficient training with sparse BoW.
    """

    def __init__(
        self,
        n_features: int,
        vocab_size: int,
        init_pi: float = 0.3,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        # Feature-to-word emission logits (K x V)
        self.B_logits = nn.Parameter(
            0.01 * torch.randn(n_features, vocab_size, dtype=dtype)
        )
        # Background word distribution (V,)
        self.bg_logits = nn.Parameter(
            0.01 * torch.randn(vocab_size, dtype=dtype)
        )
        # Mixture parameter (convert logit to probability via sigmoid)
        self.register_buffer(
            "pi_logit",
            torch.tensor(np.log(init_pi / (1 - init_pi)), dtype=dtype)
        )

    def forward(
        self,
        theta: torch.Tensor,
        active_vocab_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute word probabilities from feature activations.

        Args:
            theta: Normalized feature activations (B x K)
            active_vocab_mask: Optional boolean mask (V,) for active vocabulary.
                If provided, only computes probabilities for active words.

        Returns:
            Word probabilities (B x V) or (B x V_active) if mask provided
        """
        if active_vocab_mask is None:
            # Full vocabulary computation
            B_probs = torch.softmax(self.B_logits, dim=1)
            topic_words = theta @ B_probs
            bg_probs = torch.softmax(self.bg_logits, dim=0)
        else:
            # Subset computation for efficiency
            # Compute log-normalizer over full vocab for numerical stability
            log_denoms = torch.logsumexp(self.B_logits, dim=1, keepdim=True)  # K x 1

            # Subset active columns
            B_logits_subset = self.B_logits[:, active_vocab_mask]  # K x V_active
            B_probs_subset = (B_logits_subset - log_denoms).exp()  # K x V_active

            topic_words = theta @ B_probs_subset  # B x V_active

            # Background subset
            bg_log_denom = torch.logsumexp(self.bg_logits, dim=0)
            bg_logits_subset = self.bg_logits[active_vocab_mask]
            bg_probs = (bg_logits_subset - bg_log_denom).exp()

        # Mixture: (1-pi) * topic + pi * background
        pi = torch.sigmoid(self.pi_logit)
        return (1 - pi) * topic_words + pi * bg_probs.unsqueeze(0)


class SparseBoWDataset(Dataset):
    """Dataset pairing embeddings with sparse bag-of-words."""

    def __init__(self, embeddings: torch.Tensor, bow: csr_matrix):
        self.embeddings = embeddings
        self.bow = bow
        assert embeddings.shape[0] == bow.shape[0]

    def __len__(self) -> int:
        return self.embeddings.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, csr_matrix]:
        return self.embeddings[idx], self.bow.getrow(idx)


def collate_sparse_bow(batch: list[tuple[torch.Tensor, csr_matrix]]) -> tuple:
    """Collate function that stacks embeddings and v-stacks sparse BoW."""
    embeddings = torch.stack([e for e, _ in batch])
    bow_rows = [b for _, b in batch]
    bow_batch = vstack(bow_rows)
    return embeddings, bow_batch


class CorpusAdapter:
    """
    Adapts SAE topic atoms to corpus-specific word distributions.

    This learns a feature-to-word matrix B where B[f, w] represents
    the association between feature f and word w in the corpus vocabulary.

    The learning objective maximizes the bag-of-words likelihood:
        p(words|theta) where theta = SAE.encode(embeddings)

    Parameters
    ----------
    vocab_size : int
        Size of corpus vocabulary
    n_features : int
        Number of SAE features (topic atoms)
    idf_weighting : bool, default=True
        Whether to use IDF weighting in learning
    init_pi : float, default=0.3
        Initial background mixture parameter
    learning_rate : float, default=1e-2
        Learning rate for B matrix optimization
    device : str, default="auto"
        Device for computation ("auto", "cpu", "cuda")
    """

    def __init__(
        self,
        vocab_size: int,
        n_features: int,
        idf_weighting: bool = True,
        init_pi: float = 0.3,
        learning_rate: float = 1e-2,
        device: str = "auto",
        use_sparse_activation: bool = False,
    ):
        self.vocab_size = vocab_size
        self.n_features = n_features
        self.idf_weighting = idf_weighting
        self.init_pi = init_pi
        self.learning_rate = learning_rate
        self.use_sparse_activation = use_sparse_activation

        # Resolve device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Learned parameters
        self.feature_word_matrix_: np.ndarray | None = None
        self.background_distribution_: np.ndarray | None = None
        self.pi_: float | None = None
        self.idf_: np.ndarray | None = None

        # Training state
        self._model: TopicWordModel | None = None
        self._is_fitted = False

    def _validate_inputs(
        self,
        embeddings: torch.Tensor,
        bow: csr_matrix,
        sae,
    ) -> tuple:
        """Validate and prepare inputs for training."""
        n_docs = embeddings.shape[0]
        assert n_docs == bow.shape[0], "Embeddings and BoW must have same length"

        # Move SAE to device
        sae_dtype = torch.float32 if self.device.type == "cpu" else torch.bfloat16
        sae = sae.to(self.device, dtype=sae_dtype).eval()

        # Detect number of features from SAE
        with torch.no_grad():
            probe_emb = embeddings[:1].to(self.device, dtype=sae_dtype)
            n_features_detected = sae.encode(probe_emb).shape[1]

        logger.info(f"Detected {n_features_detected} features from SAE")
        self.n_features = n_features_detected

        return sae, sae_dtype

    def _compute_idf(self, bow: csr_matrix) -> torch.Tensor:
        """Compute IDF weights from bag-of-words."""
        n_docs = bow.shape[0]
        # Document frequency per word (how many docs contain each word)
        doc_freq = bow.astype(bool).sum(axis=0).A1.clip(min=1)
        idf = np.log(n_docs / doc_freq)
        idf = idf / idf.max()  # Normalize to [0, 1]
        return torch.from_numpy(idf).float().to(self.device)

    def _compute_theta(self, sae, emb: torch.Tensor) -> torch.Tensor:
        """SAE feature activations used as the BoW-emission mixture θ.

        By default (``use_sparse_activation=False``) uses ``sae.encode()`` —
        the dense ReLU pre-activations — which is SAE-TM's θ for word-emission
        training. Set ``use_sparse_activation=True`` to instead use the true
        top-k sparse activation (via ``activate``), giving sharper per-atom
        mixtures at the cost of diverging from SAE-TM.

        ``emb`` is expected to already be on the right device/dtype.
        """
        if self.use_sparse_activation and hasattr(sae, "activate"):
            h = sae.encode(emb)
            theta, _ = sae.activate(h)
        else:
            theta = sae.encode(emb)
        return theta.float()

    def fit(
        self,
        embeddings: torch.Tensor,
        bow: csr_matrix,
        sae,
        n_epochs: int = 50,
        batch_size: int = 1024,
        num_workers: int = 0,
        verbose: bool = True,
    ) -> "CorpusAdapter":
        """
        Learn feature-to-word matrix from corpus.

        Parameters
        ----------
        embeddings : torch.Tensor
            Document embeddings (n_docs x embedding_dim)
        bow : scipy.sparse.csr_matrix
            Bag-of-words matrix (n_docs x vocab_size)
        sae : nn.Module
            Trained SAE model with encode() method
        n_epochs : int, default=50
            Number of training epochs
        batch_size : int, default=1024
            Training batch size
        num_workers : int, default=0
            DataLoader workers
        verbose : bool, default=True
            Whether to show progress bars

        Returns
        -------
        CorpusAdapter
            Fitted adapter instance
        """
        logger.info(f"Fitting CorpusAdapter: n_epochs={n_epochs}, batch_size={batch_size}")

        # Validate and prepare
        sae, sae_dtype = self._validate_inputs(embeddings, bow, sae)

        # Compute IDF if needed
        if self.idf_weighting:
            self.idf_ = self._compute_idf(bow)

        # Create model and optimizer
        self._model = TopicWordModel(
            n_features=self.n_features,
            vocab_size=self.vocab_size,
            init_pi=self.init_pi,
        ).to(self.device)

        # No weight decay on logits
        optimizer = optim.AdamW([
            {"params": [self._model.B_logits, self._model.bg_logits], "weight_decay": 0.0},
            {"params": [self._model.pi_logit], "weight_decay": 1e-5},
        ], lr=self.learning_rate)

        # Create dataset and loader
        dataset = SparseBoWDataset(embeddings, bow)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_sparse_bow,
            drop_last=False,
        )

        # Training loop
        disable_pbar = not verbose
        for epoch in range(1, n_epochs + 1):
            epoch_loss = 0.0
            total_tokens = 0
            empty_batches = 0

            pbar = tqdm(loader, desc=f"Epoch {epoch}/{n_epochs}", disable=disable_pbar)

            for emb_batch, bow_batch in pbar:
                optimizer.zero_grad(set_to_none=True)

                # Get sparse columns that have non-zero entries in this batch
                bow_col_sums = np.asarray(bow_batch.sum(axis=0)).flatten()
                active_vocab_mask = bow_col_sums > 0

                if not active_vocab_mask.any():
                    empty_batches += 1
                    continue

                # Subset to active vocabulary
                bow_subset = bow_batch[:, active_vocab_mask].toarray()
                bow_subset = torch.from_numpy(bow_subset).float().to(self.device)

                # Apply IDF weighting if enabled
                if self.idf_weighting and self.idf_ is not None:
                    idf_subset = self.idf_[active_vocab_mask]
                    bow_subset = bow_subset * idf_subset.unsqueeze(0)

                # Compute SAE activations (no grad)
                with torch.no_grad():
                    theta = self._compute_theta(
                        sae, emb_batch.to(self.device, dtype=sae_dtype)
                    )

                # Normalize theta (row-wise)
                row_sums = theta.sum(dim=1, keepdim=True).clamp_min(1e-8)
                theta_normalized = theta / row_sums

                # Forward pass with active vocabulary mask
                active_mask_tensor = torch.from_numpy(active_vocab_mask).to(self.device)
                predicted = self._model(theta_normalized, active_vocab_mask=active_mask_tensor)
                predicted = predicted.clamp(min=1e-8)

                # Loss: negative log likelihood
                loss = (bow_subset * -predicted.log()).sum()

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                total_tokens += bow_subset.sum().item()

                pbar.set_postfix({"loss": f"{epoch_loss / max(1, (pbar.n + 1)):.4f}"})

            avg_loss = epoch_loss / max(total_tokens, 1.0)
            perplexity = np.exp(avg_loss)

            logger.info(
                f"Epoch {epoch}: nll/token={avg_loss:.4f}, ppl={perplexity:.3f}, "
                f"tokens={int(total_tokens):,}, empty_batches={empty_batches}"
            )

        # Extract learned parameters
        self._save_learned_parameters()

        self._is_fitted = True
        logger.info("CorpusAdapter fitting complete")

        return self

    def _save_learned_parameters(self):
        """Extract learned parameters from model."""
        if self._model is None:
            return

        with torch.no_grad():
            self._model.cpu()

            # Feature-to-word matrix (softmax of logits)
            B = torch.softmax(self._model.B_logits, dim=1)
            self.feature_word_matrix_ = B.numpy()

            # Background distribution
            bg = torch.softmax(self._model.bg_logits, dim=0)
            self.background_distribution_ = bg.numpy()

            # Mixture parameter
            self.pi_ = torch.sigmoid(self._model.pi_logit).item()

        logger.info(
            f"Extracted feature_word_matrix: shape={self.feature_word_matrix_.shape}, "
            f"pi={self.pi_:.3f}"
        )

    def transform(
        self,
        feature_activations: np.ndarray | torch.Tensor,
    ) -> np.ndarray:
        """
        Transform feature activations to word distributions.

        Parameters
        ----------
        feature_activations : np.ndarray or torch.Tensor
            SAE activations (n_docs x n_features)

        Returns
        -------
        np.ndarray
            Document-word distributions (n_docs x vocab_size)
        """
        if not self._is_fitted:
            raise RuntimeError("CorpusAdapter must be fitted before transform")

        if isinstance(feature_activations, np.ndarray):
            feature_activations = torch.from_numpy(feature_activations)

        # Ensure we're on CPU for numpy output
        feature_activations = feature_activations.float().cpu()

        with torch.no_grad():
            if self._model is not None:
                # Use learned model (requires normalized theta)
                self._model.cpu()
                # Normalize theta before passing to model
                row_sums = feature_activations.sum(dim=1, keepdim=True).clamp_min(1e-8)
                theta_normalized = feature_activations / row_sums
                predicted = self._model(theta_normalized)
                return predicted.numpy()
            else:
                # Use pre-computed matrix
                B = torch.from_numpy(self.feature_word_matrix_)
                bg = torch.from_numpy(self.background_distribution_)
                pi = self.pi_

                # Normalize activations
                theta = feature_activations
                row_sums = theta.sum(dim=1, keepdim=True).clamp_min(1e-8)
                theta_normalized = theta / row_sums

                # Compute predictions
                topic_words = theta_normalized @ B
                bg_probs = bg.unsqueeze(0)
                predicted = (1 - pi) * topic_words + pi * bg_probs

                return predicted.numpy()

    def fit_transform(
        self,
        embeddings: torch.Tensor,
        bow: csr_matrix,
        sae,
        n_epochs: int = 50,
        batch_size: int = 1024,
        num_workers: int = 0,
        verbose: bool = True,
    ) -> np.ndarray:
        """
        Fit and transform in one step.

        Parameters
        ----------
        embeddings : torch.Tensor
            Document embeddings (n_docs x embedding_dim)
        bow : scipy.sparse.csr_matrix
            Bag-of-words matrix (n_docs x vocab_size)
        sae : nn.Module
            Trained SAE model with encode() method
        n_epochs : int, default=50
            Number of training epochs
        batch_size : int, default=1024
            Training batch size
        num_workers : int, default=0
            DataLoader workers
        verbose : bool, default=True
            Whether to show progress bars

        Returns
        -------
        np.ndarray
            Document-word distributions (n_docs x vocab_size)
        """
        self.fit(
            embeddings=embeddings,
            bow=bow,
            sae=sae,
            n_epochs=n_epochs,
            batch_size=batch_size,
            num_workers=num_workers,
            verbose=verbose,
        )

        # Compute activations and transform
        sae_dtype = torch.float32 if self.device.type == "cpu" else torch.bfloat16
        with torch.no_grad():
            sae = sae.to(self.device, dtype=sae_dtype).eval()
            activations = self._compute_theta(
                sae, embeddings.to(self.device, dtype=sae_dtype)
            )

        return self.transform(activations.cpu().numpy())

    def get_feature_word_matrix(self) -> np.ndarray:
        """Return the learned feature-to-word matrix."""
        if not self._is_fitted:
            raise RuntimeError("CorpusAdapter must be fitted first")
        return self.feature_word_matrix_

    def get_top_words_for_feature(
        self,
        feature_idx: int,
        vocab: list[str],
        top_n: int = 20,
    ) -> list[tuple[str, float]]:
        """
        Get top words for a specific feature.

        Parameters
        ----------
        feature_idx : int
            Index of the feature
        vocab : list of str
            Vocabulary list mapping indices to words
        top_n : int, default=20
            Number of top words to return

        Returns
        -------
        list of (str, float)
            Top words and their probabilities
        """
        if not self._is_fitted:
            raise RuntimeError("CorpusAdapter must be fitted first")

        feature_probs = self.feature_word_matrix_[feature_idx]
        top_indices = np.argsort(feature_probs)[-top_n:][::-1]

        return [(vocab[i], float(feature_probs[i])) for i in top_indices]
