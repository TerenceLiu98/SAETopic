"""
Tests for configuration module.
"""

from saetopic.config import HFHubConfig, SAETopicConfig, SAETrainingConfig


def test_default_saetopic_config():
    """Test that default SAETopicConfig has expected values."""
    config = SAETopicConfig()

    assert config.embedding_model == "jinaai/jina-embeddings-v5-text-small"
    assert config.embedding_task == "clustering"
    assert config.sae_model == "saetopic/jina-v5-sae-small"
    assert config.n_topics == 50
    assert config.top_k_features == 32
    assert config.random_state == 42


def test_custom_saetopic_config():
    """Test that SAETopicConfig can be customized."""
    config = SAETopicConfig(
        n_topics=100,
        embedding_model="custom-model",
        random_state=123,
    )

    assert config.n_topics == 100
    assert config.embedding_model == "custom-model"
    assert config.random_state == 123


def test_default_sae_training_config():
    """Test that default SAETrainingConfig has expected values."""
    config = SAETrainingConfig()

    assert config.input_dim == 1024
    assert config.expansion_factor == 32
    assert config.top_k == 32
    assert config.learning_rate == 1e-4
    assert config.batch_size == 256


def test_default_hf_hub_config():
    """Test that default HFHubConfig has expected values."""
    config = HFHubConfig()

    assert config.cache_dir is None
    assert config.offline is False
    assert config.timeout == 30
