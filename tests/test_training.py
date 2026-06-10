"""
Tests for SAE training infrastructure.
"""

import pytest
import torch

from saetopic.sae.modules import BatchTopKSAE, TopKSAE, create_sae
from saetopic.training.data import EmbeddingDataset
from saetopic.training.train_sae import SAEOptimizer, SAETrainer, TrainingConfig


def test_create_sae_topk():
    """Test creating a TopKSAE model."""
    model = create_sae(
        input_dim=128,
        architecture="topk",
        n_features=256,
        top_k=8,
    )

    assert isinstance(model, TopKSAE)
    assert model.input_dim == 128
    assert model.n_features == 256
    assert model.top_k == 8


def test_create_sae_batch_topk():
    """Test creating a BatchTopKSAE model."""
    model = create_sae(
        input_dim=128,
        architecture="batch_topk",
        expansion_factor=16,
        top_k=8,
    )

    assert isinstance(model, BatchTopKSAE)
    assert model.input_dim == 128
    assert model.n_features == 128 * 16
    assert model.top_k == 8


def test_topk_sae_forward():
    """Test TopKSAE forward pass."""
    model = TopKSAE(input_dim=128, n_features=256, top_k=8)
    x = torch.randn(4, 128)

    x_recon, h, f, topk_indices = model(x)

    assert x_recon.shape == (4, 128)
    assert h.shape == (4, 256)
    assert f.shape == (4, 256)
    assert topk_indices.shape == (4, 8)

    # Check sparsity: most features should be zero
    assert (f == 0).sum() > f.numel() // 2


def test_batch_topk_sae_forward():
    """Test BatchTopKSAE forward pass."""
    model = BatchTopKSAE(input_dim=128, n_features=256, top_k=8)
    x = torch.randn(4, 128)

    x_recon, h, f, topk_indices = model(x)

    assert x_recon.shape == (4, 128)
    assert h.shape == (4, 256)
    assert f.shape == (4, 256)
    assert topk_indices.shape == (4, 8)


def test_sae_compute_loss():
    """Test loss computation."""
    model = BatchTopKSAE(input_dim=128, n_features=256, top_k=8)
    x = torch.randn(4, 128)

    x_recon, h, f, _ = model(x)
    loss, losses = model.compute_loss(x, x_recon, h, f)

    assert "total" in losses
    assert "reconstruction" in losses
    assert "sparsity" in losses
    assert "auxiliary" in losses

    # All losses should be non-negative
    for value in losses.values():
        assert value.item() >= 0


def test_feature_stats():
    """Test feature activation statistics."""
    model = BatchTopKSAE(input_dim=128, n_features=256, top_k=8)
    x = torch.randn(16, 128)

    # Update stats
    with torch.no_grad():
        _, _, f, _ = model(x)
        model.update_feature_stats(f)

    usage = model.get_feature_usage()
    assert usage.shape == (256,)
    assert (usage >= 0).all()

    # Reset stats
    model.reset_feature_stats()
    usage_after = model.get_feature_usage()
    assert (usage_after == 0).all()


def test_dead_features():
    """Test dead feature detection."""
    model = BatchTopKSAE(input_dim=128, n_features=256, top_k=8)

    # Initially all features are "dead" (no activations yet)
    dead = model.get_dead_features(threshold=0.01)
    assert dead.sum() == 256  # All dead initially


def test_embedding_dataset():
    """Test EmbeddingDataset."""
    embeddings = torch.randn(100, 128)
    dataset = EmbeddingDataset(embeddings, normalize=True)

    assert len(dataset) == 100
    assert dataset.embedding_dim == 128

    # Test __getitem__
    x = dataset[0]
    assert x.shape == (128,)


def test_embedding_dataset_normalize():
    """Test that EmbeddingDataset normalizes embeddings."""
    embeddings = torch.randn(10, 128)
    dataset = EmbeddingDataset(embeddings, normalize=True)

    # Check normalized
    for i in range(len(dataset)):
        x = dataset[i]
        assert torch.allclose(x.norm(p=2), torch.tensor(1.0), atol=1e-6)


def test_training_config():
    """Test TrainingConfig."""
    config = TrainingConfig(
        input_dim=128,
        n_features=256,
        top_k=8,
        batch_size=32,
    )

    assert config.input_dim == 128
    assert config.n_features == 256
    assert config.top_k == 8
    assert config.batch_size == 32

    # Test to_dict
    config_dict = config.to_dict()
    assert config_dict["input_dim"] == 128
    assert config_dict["n_features"] == 256


def test_sae_optimizer():
    """Test SAEOptimizer."""
    model = TopKSAE(input_dim=128, n_features=256, top_k=8)
    optimizer = SAEOptimizer(model, learning_rate=1e-4)

    # Create dummy loss
    x = torch.randn(4, 128)
    x_recon, h, f, _ = model(x)
    loss, _ = model.compute_loss(x, x_recon, h, f)

    # Optimizer step
    optimizer.step(loss)

    # Check state dict
    state_dict = optimizer.state_dict()
    assert "optimizer" in state_dict


def test_training_state():
    """Test TrainingState."""
    from saetopic.training.train_sae import TrainingState

    state = TrainingState()

    assert state.epoch == 0
    assert state.global_step == 0
    assert state.best_loss == float("inf")

    # Update state
    state.update({"total": 1.5, "reconstruction": 1.0})
    assert state.global_step == 1
    assert state.best_loss == 1.5


def test_sae_trainer_init():
    """Test SAETrainer initialization."""
    model = TopKSAE(input_dim=128, n_features=256, top_k=8)
    config = TrainingConfig(input_dim=128, n_features=256, top_k=8, n_epochs=1)

    trainer = SAETrainer(model, config, output_dir="/tmp/test_sae")

    assert trainer.model is model
    assert trainer.config is config
    assert trainer.output_dir.name == "test_sae"


def test_train_sae_function():
    """Test train_sae function with dummy data."""
    embeddings = torch.randn(100, 128)
    dataset = EmbeddingDataset(embeddings)

    # Train for 1 epoch
    trainer = train_sae(
        dataset=dataset,
        n_epochs=1,
        batch_size=16,
        output_dir="/tmp/test_sae_function",
        save_frequency=100,  # Don't save during short test
    )

    assert trainer.state.epoch == 1
    assert trainer.state.global_step > 0


def test_unknown_architecture():
    """Test that unknown architecture raises error."""
    with pytest.raises(ValueError, match="Unknown architecture"):
        create_sae(input_dim=128, architecture="unknown")
