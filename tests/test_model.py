"""
Tests for SAETopicModel.
"""

import pytest

from saetopic import SAETopicModel


def test_model_initialization():
    """Test that SAETopicModel can be initialized with defaults."""
    model = SAETopicModel()

    assert model.embedding_model == "jinaai/jina-embeddings-v5-text-small"
    assert model.embedding_task == "clustering"
    assert model.sae_model == "saetopic/jina-v5-sae-small"
    assert model.n_topics == 50
    assert model.top_k_features == 32


def test_model_custom_params():
    """Test that SAETopicModel can be initialized with custom parameters."""
    model = SAETopicModel(
        embedding_model="custom-model",
        n_topics=100,
        random_state=123,
    )

    assert model.embedding_model == "custom-model"
    assert model.n_topics == 100
    assert model.random_state == 123


def test_from_pretrained_not_implemented():
    """Test that from_pretrained raises NotImplementedError."""
    with pytest.raises(NotImplementedError, match="from_pretrained is not implemented yet"):
        SAETopicModel.from_pretrained("saetopic/jina-v5-sae-small")


def test_fit_not_implemented():
    """Test that fit raises NotImplementedError."""
    model = SAETopicModel()
    docs = ["test document"]

    with pytest.raises(NotImplementedError, match="fit is not implemented yet"):
        model.fit(docs)


def test_fit_transform_not_implemented():
    """Test that fit_transform raises NotImplementedError."""
    model = SAETopicModel()
    docs = ["test document"]

    with pytest.raises(NotImplementedError, match="fit_transform is not implemented yet"):
        model.fit_transform(docs)


def test_retopic_not_implemented():
    """Test that retopic raises NotImplementedError."""
    model = SAETopicModel()

    with pytest.raises(NotImplementedError, match="retopic is not implemented yet"):
        model.retopic(n_topics=30)


def test_reduce_topics_alias():
    """Test that reduce_topics is an alias for retopic."""
    model = SAETopicModel()

    with pytest.raises(NotImplementedError, match="retopic is not implemented yet"):
        model.reduce_topics(nr_topics=30)


def test_get_topic_info_not_implemented():
    """Test that get_topic_info raises NotImplementedError."""
    model = SAETopicModel()

    with pytest.raises(NotImplementedError, match="get_topic_info is not implemented yet"):
        model.get_topic_info()


def test_visualize_topics_not_implemented():
    """Test that visualize_topics raises NotImplementedError."""
    model = SAETopicModel()

    with pytest.raises(NotImplementedError, match="visualize_topics is not implemented yet"):
        model.visualize_topics()


def test_save_not_implemented():
    """Test that save raises NotImplementedError."""
    model = SAETopicModel()

    with pytest.raises(NotImplementedError, match="save is not implemented yet"):
        model.save("/tmp/model")


def test_load_not_implemented():
    """Test that load raises NotImplementedError."""
    with pytest.raises(NotImplementedError, match="load is not implemented yet"):
        SAETopicModel.load("/tmp/model")
