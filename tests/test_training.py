"""
Tests for SAE training infrastructure.
"""

import json

import numpy as np
import pytest
import torch

from saetopic.sae.modules import BatchTopKSAE, TopKSAE, create_sae
from saetopic.training.data import EmbeddingDataset
from saetopic.training.train_sae import SAEOptimizer, SAETrainer, TrainingConfig, train_sae


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


def test_sparse_forward_matches_dense_forward():
    """Test memory-efficient sparse forward matches dense forward outputs."""
    model = TopKSAE(input_dim=32, n_features=64, top_k=4)
    x = torch.randn(8, 32)

    dense_x_recon, dense_h, dense_f, dense_indices = model(x)
    sparse_x_recon, sparse_h, topk_values, sparse_indices = model.forward_sparse(x)

    assert torch.allclose(sparse_h, dense_h)
    assert torch.equal(sparse_indices, dense_indices)
    assert torch.allclose(
        topk_values,
        dense_f.gather(dim=-1, index=dense_indices),
    )
    assert torch.allclose(sparse_x_recon, dense_x_recon, atol=1e-6)


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


def test_sparse_loss_matches_dense_loss():
    """Test sparse loss matches dense loss without materializing dense f."""
    model = BatchTopKSAE(input_dim=32, n_features=64, top_k=4)
    x = torch.randn(8, 32)

    dense_x_recon, dense_h, dense_f, dense_indices = model(x)
    dense_loss, dense_losses = model.compute_loss(x, dense_x_recon, dense_h, dense_f)

    sparse_x_recon, sparse_h, topk_values, sparse_indices = model.forward_sparse(x)
    sparse_loss, sparse_losses = model.compute_loss_sparse(
        x,
        sparse_x_recon,
        sparse_h,
        topk_values,
        sparse_indices,
    )

    assert torch.equal(sparse_indices, dense_indices)
    assert torch.allclose(sparse_loss, dense_loss, atol=1e-6)
    for key in dense_losses:
        assert torch.allclose(sparse_losses[key], dense_losses[key], atol=1e-6)


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


def test_sparse_feature_stats_match_dense_stats():
    """Test sparse feature statistics match dense feature statistics."""
    dense_model = BatchTopKSAE(input_dim=32, n_features=64, top_k=4)
    sparse_model = BatchTopKSAE(input_dim=32, n_features=64, top_k=4)
    sparse_model.load_state_dict(dense_model.state_dict())
    x = torch.randn(8, 32)

    with torch.no_grad():
        _, _, dense_f, _ = dense_model(x)
        dense_model.update_feature_stats(dense_f)

        _, _, topk_values, topk_indices = sparse_model.forward_sparse(x)
        sparse_model.update_feature_stats_sparse(topk_values, topk_indices)

    assert torch.allclose(sparse_model.feature_counts, dense_model.feature_counts)
    assert torch.equal(sparse_model.update_count, dense_model.update_count)


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


def test_embedding_dataset_from_file_mmap(tmp_path):
    """Test memory-mapped .npy loading with lazy per-sample normalization."""
    embeddings = np.random.randn(10, 8).astype(np.float32)
    path = tmp_path / "embeddings.npy"
    np.save(path, embeddings)

    dataset = EmbeddingDataset.from_file(path, normalize=True, mmap_mode="r")

    assert dataset.lazy is True
    assert isinstance(dataset.embeddings, np.memmap)
    assert len(dataset) == 10
    assert dataset.embedding_dim == 8
    assert torch.allclose(dataset[0].norm(p=2), torch.tensor(1.0), atol=1e-6)


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


def test_train_sae_respects_config_output_dir(tmp_path):
    """Test config.output_dir is used when output_dir is not explicitly passed."""
    embeddings = torch.randn(32, 16)
    dataset = EmbeddingDataset(embeddings)
    config_output_dir = tmp_path / "config_output"

    config = TrainingConfig(
        input_dim=16,
        n_features=32,
        top_k=4,
        n_epochs=1,
        batch_size=8,
        output_dir=str(config_output_dir),
        save_frequency=100,
    )

    trainer = train_sae(dataset=dataset, config=config)

    assert trainer.output_dir == config_output_dir
    final_dir = config_output_dir / "final"
    assert final_dir.exists()
    assert (final_dir / "config.json").exists()
    assert (final_dir / "model_card.md").exists()
    assert (final_dir / "README.md").exists()
    model_card = (final_dir / "model_card.md").read_text()
    assert "Checkpoint Contents" in model_card
    assert "fit_transform" not in model_card

    checksums = (final_dir / "checksums.txt").read_text()
    assert "optimizer.pt" in checksums
    assert "training_state.pt" in checksums
    assert "model.safetensors" in checksums or "model.pt" in checksums


def test_train_sae_from_embeddings_path_uses_mmap(tmp_path):
    """Test train_sae can train directly from a .npy path using mmap loading."""
    embeddings = np.random.randn(32, 16).astype(np.float32)
    embeddings_path = tmp_path / "embeddings.npy"
    np.save(embeddings_path, embeddings)

    trainer = train_sae(
        embeddings_path=embeddings_path,
        input_dim=16,
        n_features=32,
        top_k=4,
        n_epochs=1,
        batch_size=8,
        output_dir=str(tmp_path / "checkpoints"),
        save_frequency=100,
    )

    assert trainer.state.epoch == 1
    assert trainer.state.global_step > 0


def test_train_sae_from_sharded_embeddings_path(tmp_path):
    """Test train_sae can train directly from a sharded embedding directory."""
    embeddings_dir = tmp_path / "embeddings"
    embeddings_dir.mkdir()
    shard_0 = np.random.randn(16, 8).astype(np.float32)
    shard_1 = np.random.randn(16, 8).astype(np.float32)
    np.save(embeddings_dir / "shard_000000.npy", shard_0)
    np.save(embeddings_dir / "shard_000001.npy", shard_1)
    (embeddings_dir / "manifest.json").write_text(
        json.dumps(
            {
                "format": "saetopic.sharded_embeddings.v1",
                "dtype": "float32",
                "shape": [32, 8],
                "shard_size": 16,
                "shards": [
                    {"file": "shard_000000.npy", "shape": [16, 8]},
                    {"file": "shard_000001.npy", "shape": [16, 8]},
                ],
            }
        )
    )

    trainer = train_sae(
        embeddings_path=embeddings_dir,
        input_dim=8,
        n_features=16,
        top_k=4,
        n_epochs=1,
        batch_size=8,
        output_dir=str(tmp_path / "checkpoints"),
        save_frequency=100,
    )

    assert trainer.state.epoch == 1
    assert trainer.state.global_step > 0


def test_train_sae_from_embeddings_path_can_skip_normalize(tmp_path):
    """Test train_sae can skip re-normalizing pre-normalized embedding files."""
    embeddings = np.ones((16, 8), dtype=np.float32) * 2.0
    embeddings_path = tmp_path / "embeddings.npy"
    np.save(embeddings_path, embeddings)

    trainer = train_sae(
        embeddings_path=embeddings_path,
        normalize_embeddings=False,
        input_dim=8,
        n_features=16,
        top_k=4,
        n_epochs=1,
        batch_size=8,
        output_dir=str(tmp_path / "checkpoints_no_normalize"),
        save_frequency=100,
    )

    assert trainer.state.epoch == 1

    from saetopic.training.data import EmbeddingDataset

    dataset = EmbeddingDataset.from_file(
        embeddings_path,
        normalize=False,
        mmap_mode="r",
    )
    assert torch.equal(dataset[0], torch.full((8,), 2.0))


def test_cli_train_passes_mmap_and_normalize_flags(monkeypatch):
    """Test CLI train forwards mmap and normalize flags to EmbeddingDataset."""
    import argparse

    import saetopic.training as training_package
    import saetopic.training.cli as training_cli
    import saetopic.training.data as data_module

    class DummyDataset:
        embedding_dim = 8

    captured_from_file = {}
    captured_train = {}
    captured_upload = {}

    def fake_from_file(path, normalize=True, mmap_mode=None):
        captured_from_file.update(
            {
                "path": path,
                "normalize": normalize,
                "mmap_mode": mmap_mode,
            }
        )
        return DummyDataset()

    def fake_train_sae(*, dataset, config):
        captured_train.update({"dataset": dataset, "config": config})
        return object()

    def fake_upload_checkpoint(checkpoint_dir, repo_id, create_repo=False, private=False):
        captured_upload.update(
            {
                "checkpoint_dir": checkpoint_dir,
                "repo_id": repo_id,
                "create_repo": create_repo,
                "private": private,
            }
        )

    monkeypatch.setattr(data_module.EmbeddingDataset, "from_file", fake_from_file)
    monkeypatch.setattr(training_package, "train_sae", fake_train_sae)
    monkeypatch.setattr("saetopic.hf_utils.upload_checkpoint", fake_upload_checkpoint)

    args = argparse.Namespace(
        embeddings="embeddings.npy",
        no_mmap=True,
        no_normalize_embeddings=True,
        input_dim=None,
        n_features=None,
        expansion_factor=32,
        top_k=16,
        architecture="batch_topk",
        decoder_bias=True,
        encoder_bias=False,
        normalization=None,
        learning_rate=1e-4,
        batch_size=32,
        n_epochs=1,
        device="cpu",
        seed=42,
        save_frequency=100,
        recon_loss_weight=1.0,
        sparsity_loss_weight=1.0,
        aux_loss_weight=0.001,
        output="checkpoints/test",
        checkpoint_name="test",
        dataset_name="dataset",
        dataset_license="license",
        upload_to_hf="org/repo",
        create_repo=True,
        private=True,
    )

    training_cli.train_sae_from_args(args)

    assert captured_from_file == {
        "path": "embeddings.npy",
        "normalize": False,
        "mmap_mode": None,
    }
    assert captured_train["dataset"].embedding_dim == 8
    assert captured_train["config"].input_dim == 8
    assert captured_train["config"].top_k == 16
    assert captured_upload == {
        "checkpoint_dir": "checkpoints/test/final",
        "repo_id": "org/repo",
        "create_repo": True,
        "private": True,
    }


def test_cli_embed_passes_streaming_and_sentence_transformer_args(monkeypatch):
    """Test CLI embed forwards model, multi-GPU, chunking, and save arguments."""
    import argparse

    import sentence_transformers
    import torch as torch_module

    import saetopic.training as training_package
    import saetopic.training.cli as training_cli

    captured_model = {}
    captured_streaming = {}
    captured_save = {}

    class FakeSentenceTransformer:
        def __init__(self, model_name, **kwargs):
            captured_model.update({"model_name": model_name, "kwargs": kwargs})
            self.max_seq_length = None

    def fake_create_streaming_dataset(**kwargs):
        captured_streaming.update(kwargs)
        return "streaming-dataset"

    def fake_compute_and_save_embeddings(*, dataset, output_path, chunk_size):
        captured_save.update(
            {
                "dataset": dataset,
                "output_path": output_path,
                "chunk_size": chunk_size,
            }
        )
        return 123, 512

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", FakeSentenceTransformer)
    monkeypatch.setattr(torch_module.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch_module.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(training_package, "create_streaming_dataset", fake_create_streaming_dataset)
    monkeypatch.setattr(
        training_package,
        "compute_and_save_embeddings",
        fake_compute_and_save_embeddings,
    )

    args = argparse.Namespace(
        dataset_name="HuggingFaceFW/finewiki",
        subset=None,
        split="train",
        text_column="text",
        model="jinaai/jina-embeddings-v5-text-small",
        output="data/finewiki_embeddings.npy",
        max_samples=1000,
        buffer_size=100,
        seed=123,
        embedding_batch_size=64,
        encode_batch_size=8,
        encode_device=["cuda:7"],
        auto_multi_gpu=True,
        encode_chunk_size=128,
        text_chunk_size=0,
        text_chunk_overlap=32,
        max_seq_length=1024,
        truncate_dim=512,
        save_chunk_size=5000,
        no_normalize_embeddings=True,
        task="clustering",
        no_bf16=False,
        trust_remote_code=True,
    )

    training_cli.compute_embeddings_from_args(args)

    assert captured_model["model_name"] == "jinaai/jina-embeddings-v5-text-small"
    assert captured_model["kwargs"]["trust_remote_code"] is True
    assert captured_model["kwargs"]["device"] == torch_module.device("cuda")
    assert captured_model["kwargs"]["model_kwargs"] == {"dtype": torch_module.bfloat16}
    assert captured_model["kwargs"]["truncate_dim"] == 512

    embedder = captured_streaming["embedder"]
    assert embedder.max_seq_length == 1024
    assert captured_streaming["dataset_name"] == "HuggingFaceFW/finewiki"
    assert captured_streaming["encode_device"] == ["cuda:0", "cuda:1"]
    assert captured_streaming["encode_chunk_size"] == 128
    assert captured_streaming["text_chunk_size"] is None
    assert captured_streaming["text_chunk_overlap"] == 32
    assert captured_streaming["normalize"] is False
    assert captured_streaming["seed"] == 123
    assert captured_streaming["max_samples"] == 1000
    assert captured_streaming["task"] == "clustering"

    assert captured_save == {
        "dataset": "streaming-dataset",
        "output_path": "data/finewiki_embeddings.npy",
        "chunk_size": 5000,
    }


def test_cli_upload_existing_checkpoint(monkeypatch):
    """Test CLI upload forwards an existing checkpoint without training."""
    import argparse

    import saetopic.training.cli as training_cli

    captured_upload = {}

    def fake_upload_checkpoint(
        checkpoint_dir,
        repo_id,
        create_repo=False,
        private=False,
        commit_message=None,
    ):
        captured_upload.update(
            {
                "checkpoint_dir": checkpoint_dir,
                "repo_id": repo_id,
                "create_repo": create_repo,
                "private": private,
                "commit_message": commit_message,
            }
        )

    monkeypatch.setattr("saetopic.hf_utils.upload_checkpoint", fake_upload_checkpoint)

    args = argparse.Namespace(
        checkpoint_dir="checkpoints/jina-v5-sae-small/final",
        repo_id="org/repo",
        create_repo=True,
        private=True,
        commit_message="Upload final checkpoint",
    )

    training_cli.upload_checkpoint_from_args(args)

    assert captured_upload == {
        "checkpoint_dir": "checkpoints/jina-v5-sae-small/final",
        "repo_id": "org/repo",
        "create_repo": True,
        "private": True,
        "commit_message": "Upload final checkpoint",
    }


def test_training_cli_requires_command(monkeypatch, capsys):
    """Test training CLI exits non-zero when no subcommand is provided."""
    import saetopic.training.cli as training_cli

    monkeypatch.setattr("sys.argv", ["saetopic-train"])

    with pytest.raises(SystemExit) as exc_info:
        training_cli.main()

    assert exc_info.value.code == 2
    assert "a command is required" in capsys.readouterr().err


def test_standard_dataloader_pins_memory_only_for_cuda(tmp_path):
    """Test non-CUDA standard training does not request pinned memory."""
    model = TopKSAE(input_dim=16, n_features=32, top_k=4)
    config = TrainingConfig(
        input_dim=16,
        n_features=32,
        top_k=4,
        n_epochs=1,
        batch_size=8,
        device="cpu",
        output_dir=str(tmp_path / "pin_memory"),
    )
    trainer = SAETrainer(model, config)
    dataset = EmbeddingDataset(torch.randn(16, 16))

    import importlib

    train_sae_module = importlib.import_module("saetopic.training.train_sae")
    original_dataloader = train_sae_module.DataLoader
    captured_kwargs = {}

    class CapturingDataLoader(original_dataloader):
        def __init__(self, *args, **kwargs):
            captured_kwargs.update(kwargs)
            super().__init__(*args, **kwargs)

    try:
        train_sae_module.DataLoader = CapturingDataLoader
        trainer.fit(dataset)
    finally:
        train_sae_module.DataLoader = original_dataloader

    assert captured_kwargs["pin_memory"] is False


def test_unknown_architecture():
    """Test that unknown architecture raises error."""
    with pytest.raises(ValueError, match="Unknown architecture"):
        create_sae(input_dim=128, architecture="unknown")
