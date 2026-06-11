"""
Tests for CorpusAdapter and TopicMerger components.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from scipy.sparse import csr_matrix

from saetopic.interpretation import CorpusAdapter
from saetopic.merging import TopicMerger


class DummySAE(nn.Module):
    """Dummy SAE for testing."""

    def __init__(self, n_features: int = 128):
        super().__init__()
        self.encoder = nn.Linear(512, n_features)
        self.decoder = nn.Linear(n_features, 512)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode embeddings to sparse features."""
        # Use ReLU for sparse-like activations
        return torch.relu(self.encoder(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full autoencoder forward pass."""
        features = self.encode(x)
        return self.decoder(features)


@pytest.fixture
def dummy_sae():
    """Create a dummy SAE for testing."""
    return DummySAE(n_features=128)


@pytest.fixture
def sample_embeddings():
    """Create sample document embeddings."""
    np.random.seed(42)
    torch.manual_seed(42)
    return torch.randn(100, 512).float()


@pytest.fixture
def sample_bow():
    """Create sample bag-of-words matrix."""
    np.random.seed(42)
    # Create sparse-like data
    data = np.random.randint(1, 10, size=500).astype(np.float32)
    rows = np.random.randint(0, 100, size=500)
    cols = np.random.randint(0, 50, size=500)

    return csr_matrix((data, (rows, cols)), shape=(100, 50))


@pytest.fixture
def sample_vocab():
    """Create sample vocabulary."""
    return [f"word_{i}" for i in range(50)]


class TestCorpusAdapter:
    """Tests for CorpusAdapter."""

    def test_init(self):
        """Test CorpusAdapter initialization."""
        adapter = CorpusAdapter(
            vocab_size=50,
            n_features=128,
        )
        assert adapter.vocab_size == 50
        assert adapter.n_features == 128
        assert not adapter._is_fitted

    def test_fit(self, sample_embeddings, sample_bow, dummy_sae):
        """Test fitting CorpusAdapter."""
        adapter = CorpusAdapter(
            vocab_size=50,
            n_features=128,
            device="cpu",
        )

        adapter.fit(
            embeddings=sample_embeddings,
            bow=sample_bow,
            sae=dummy_sae,
            n_epochs=2,
            batch_size=32,
            verbose=False,
        )

        assert adapter._is_fitted
        assert adapter.feature_word_matrix_ is not None
        assert adapter.feature_word_matrix_.shape == (128, 50)

    def test_transform_after_fit(self, sample_embeddings, sample_bow, dummy_sae):
        """Test transform after fitting."""
        adapter = CorpusAdapter(
            vocab_size=50,
            n_features=128,
            device="cpu",
        )

        adapter.fit(
            embeddings=sample_embeddings,
            bow=sample_bow,
            sae=dummy_sae,
            n_epochs=2,
            batch_size=32,
            verbose=False,
        )

        # Get activations and transform
        dummy_sae.eval()
        with torch.no_grad():
            activations = dummy_sae.encode(sample_embeddings).numpy()

        result = adapter.transform(activations)

        assert result.shape == (100, 50)
        assert np.allclose(result.sum(axis=1), 1.0, atol=1e-5)  # Rows sum to ~1

    def test_fit_transform(self, sample_embeddings, sample_bow, dummy_sae):
        """Test combined fit_transform method."""
        adapter = CorpusAdapter(
            vocab_size=50,
            n_features=128,
            device="cpu",
        )

        result = adapter.fit_transform(
            embeddings=sample_embeddings,
            bow=sample_bow,
            sae=dummy_sae,
            n_epochs=2,
            batch_size=32,
            verbose=False,
        )

        assert result.shape == (100, 50)
        assert adapter._is_fitted

    def test_get_feature_word_matrix(self, sample_embeddings, sample_bow, dummy_sae):
        """Test getting feature-to-word matrix."""
        adapter = CorpusAdapter(
            vocab_size=50,
            n_features=128,
            device="cpu",
        )

        adapter.fit(
            embeddings=sample_embeddings,
            bow=sample_bow,
            sae=dummy_sae,
            n_epochs=2,
            batch_size=32,
            verbose=False,
        )

        B = adapter.get_feature_word_matrix()
        assert B.shape == (128, 50)
        assert np.allclose(B.sum(axis=1), 1.0, atol=1e-5)  # Rows sum to ~1

    def test_get_top_words_for_feature(
        self, sample_embeddings, sample_bow, dummy_sae, sample_vocab
    ):
        """Test getting top words for a feature."""
        adapter = CorpusAdapter(
            vocab_size=50,
            n_features=128,
            device="cpu",
        )

        adapter.fit(
            embeddings=sample_embeddings,
            bow=sample_bow,
            sae=dummy_sae,
            n_epochs=2,
            batch_size=32,
            verbose=False,
        )

        top_words = adapter.get_top_words_for_feature(0, sample_vocab, top_n=10)

        assert len(top_words) == 10
        assert all(isinstance(w, str) and isinstance(p, float) for w, p in top_words)
        # Check descending order
        probs = [p for _, p in top_words]
        assert probs == sorted(probs, reverse=True)

    def test_transform_before_fit_raises(self):
        """Test that transform before fit raises an error."""
        adapter = CorpusAdapter(vocab_size=50, n_features=128)

        with pytest.raises(RuntimeError, match="must be fitted"):
            adapter.transform(np.random.randn(10, 128))


class TestTopicMerger:
    """Tests for TopicMerger."""

    @pytest.fixture
    def feature_word_matrix(self):
        """Create sample feature-to-word matrix."""
        np.random.seed(42)
        K, V = 128, 50
        # Create probability distributions
        B = np.random.rand(K, V).astype(np.float32)
        B = B / B.sum(axis=1, keepdims=True)
        return B

    @pytest.fixture
    def feature_weights(self):
        """Create sample feature activation weights."""
        np.random.seed(42)
        # Non-zero weights for most features
        weights = np.random.rand(128).astype(np.float32) * 0.1
        return weights

    @pytest.fixture
    def feature_activations(self):
        """Create sample document-feature activations."""
        np.random.seed(42)
        # Sparse-like activations
        data = np.random.rand(500).astype(np.float32) * 0.5
        rows = np.random.randint(0, 100, size=500)
        cols = np.random.randint(0, 128, size=500)
        return csr_matrix((data, (rows, cols)), shape=(100, 128))

    def test_init(self):
        """Test TopicMerger initialization."""
        merger = TopicMerger(n_topics=10)
        assert merger.n_topics == 10
        assert merger.method == "kmeans"
        assert not merger._is_fitted

    def test_fit(
        self, feature_word_matrix, feature_weights, sample_vocab
    ):
        """Test fitting TopicMerger."""
        merger = TopicMerger(n_topics=10, random_state=42)

        merger.fit(
            feature_word_matrix=feature_word_matrix,
            feature_weights=feature_weights,
            vocab=sample_vocab,
        )

        assert merger._is_fitted
        assert merger.feature_clusters_ is not None
        assert len(merger.feature_clusters_) == 128
        assert merger.topic_word_matrix_ is not None
        assert merger.topic_word_matrix_.shape == (10, 50)

    def test_fit_transform(
        self, feature_word_matrix, feature_weights, feature_activations, sample_vocab
    ):
        """Test combined fit_transform method."""
        merger = TopicMerger(n_topics=10, random_state=42)

        result = merger.fit_transform(
            feature_word_matrix=feature_word_matrix,
            feature_weights=feature_weights,
            feature_activations=feature_activations,
            vocab=sample_vocab,
        )

        assert result.shape == (100, 10)
        assert np.allclose(result.sum(axis=1), 1.0, atol=1e-5)  # Rows sum to ~1
        assert merger._is_fitted

    def test_transform(
        self, feature_word_matrix, feature_weights, feature_activations, sample_vocab
    ):
        """Test transform after fitting."""
        merger = TopicMerger(n_topics=10, random_state=42)

        merger.fit(
            feature_word_matrix=feature_word_matrix,
            feature_weights=feature_weights,
            vocab=sample_vocab,
        )

        result = merger.transform(feature_activations)

        assert result.shape == (100, 10)
        assert np.allclose(result.sum(axis=1), 1.0, atol=1e-5)

    def test_get_topic_info(
        self, feature_word_matrix, feature_weights, sample_vocab
    ):
        """Test getting topic information."""
        merger = TopicMerger(n_topics=10, random_state=42)

        merger.fit(
            feature_word_matrix=feature_word_matrix,
            feature_weights=feature_weights,
            vocab=sample_vocab,
        )

        info = merger.get_topic_info()

        assert len(info) == 10
        assert all("cluster_id" in topic for topic in info)
        assert all("top_words" in topic for topic in info)

    def test_get_topic_words(
        self, feature_word_matrix, feature_weights, sample_vocab
    ):
        """Test getting top words for a topic."""
        merger = TopicMerger(n_topics=10, random_state=42)

        merger.fit(
            feature_word_matrix=feature_word_matrix,
            feature_weights=feature_weights,
            vocab=sample_vocab,
        )

        top_words = merger.get_topic_words(0, sample_vocab, top_n=10)

        assert len(top_words) == 10
        assert all(isinstance(w, str) and isinstance(p, float) for w, p in top_words)
        # Check descending order
        probs = [p for _, p in top_words]
        assert probs == sorted(probs, reverse=True)

    def test_agglomerative_method(
        self, feature_word_matrix, feature_weights, sample_vocab
    ):
        """Test agglomerative clustering method."""
        merger = TopicMerger(
            n_topics=10, method="agglomerative", random_state=42
        )

        merger.fit(
            feature_word_matrix=feature_word_matrix,
            feature_weights=feature_weights,
            vocab=sample_vocab,
        )

        assert merger._is_fitted
        assert merger.topic_word_matrix_ is not None

    def test_transform_before_fit_raises(self):
        """Test that transform before fit raises an error."""
        merger = TopicMerger(n_topics=10)

        with pytest.raises(RuntimeError, match="must be fitted"):
            merger.transform(np.random.randn(10, 128))
