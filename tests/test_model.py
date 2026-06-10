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
    """Test that from_pretrained raises NotImplementedError (Week 3)."""
    with pytest.raises(NotImplementedError, match="from_pretrained will be implemented in Week 3"):
        SAETopicModel.from_pretrained("saetopic/jina-v5-sae-small")


def test_fit_not_implemented():
    """Test that fit raises NotImplementedError (Week 3)."""
    model = SAETopicModel()
    docs = ["test document"]

    with pytest.raises(NotImplementedError, match="fit will be implemented in Week 3"):
        model.fit(docs)


def test_fit_transform_not_implemented():
    """Test that fit_transform raises NotImplementedError (Week 3)."""
    model = SAETopicModel()
    docs = ["test document"]

    with pytest.raises(NotImplementedError, match="fit_transform will be implemented in Week 3"):
        model.fit_transform(docs)


def test_retopic_not_implemented():
    """Test that retopic raises NotImplementedError (Week 3)."""
    model = SAETopicModel()

    with pytest.raises(NotImplementedError, match="retopic will be implemented in Week 3"):
        model.retopic(n_topics=30)


def test_reduce_topics_alias():
    """Test that reduce_topics is an alias for retopic."""
    model = SAETopicModel()

    with pytest.raises(NotImplementedError, match="retopic will be implemented in Week 3"):
        model.reduce_topics(nr_topics=30)


def test_get_topic_info_not_implemented():
    """Test that get_topic_info raises NotImplementedError (Week 3)."""
    model = SAETopicModel()

    with pytest.raises(NotImplementedError, match="get_topic_info will be implemented in Week 3"):
        model.get_topic_info()


def test_visualize_topics_not_implemented():
    """Test that visualize_topics raises NotImplementedError (Week 4)."""
    model = SAETopicModel()

    with pytest.raises(NotImplementedError, match="visualize_topics will be implemented in Week 4"):
        model.visualize_topics()


def test_save_not_implemented():
    """Test that save raises NotImplementedError (Week 4)."""
    model = SAETopicModel()

    with pytest.raises(NotImplementedError, match="save will be implemented in Week 4"):
        model.save("/tmp/model")


def test_load_not_implemented():
    """Test that load raises NotImplementedError (Week 4)."""
    with pytest.raises(NotImplementedError, match="load will be implemented in Week 4"):
        SAETopicModel.load("/tmp/model")
