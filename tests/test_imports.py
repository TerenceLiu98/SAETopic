"""
Tests for package imports.
"""

def test_import_saetopic():
    """Test that the main package can be imported."""
    import saetopic

    assert saetopic.__version__ is not None


def test_import_model():
    """Test that SAETopicModel can be imported."""
    from saetopic import SAETopicModel

    assert SAETopicModel is not None


def test_import_config():
    """Test that config can be imported."""
    from saetopic.config import SAETopicConfig, SAETrainingConfig, HFHubConfig

    assert SAETopicConfig is not None
    assert SAETrainingConfig is not None
    assert HFHubConfig is not None


def test_import_sae_modules():
    """Test that SAE modules can be imported."""
    from saetopic.sae import modules, loaders, activations

    assert modules is not None
    assert loaders is not None
    assert activations is not None


def test_import_interpretation():
    """Test that interpretation modules can be imported."""
    from saetopic.interpretation import corpus_adapter, feature_words

    assert corpus_adapter is not None
    assert feature_words is not None


def test_import_other_modules():
    """Test that other modules can be imported."""
    from saetopic import (
        embeddings,
        vectorizers,
        merging,
        representation,
        visualization,
        evaluation,
        serialization,
        cli,
        config,
    )

    assert embeddings is not None
    assert vectorizers is not None
    assert merging is not None
    assert representation is not None
    assert visualization is not None
    assert evaluation is not None
    assert serialization is not None
    assert cli is not None
    assert config is not None
