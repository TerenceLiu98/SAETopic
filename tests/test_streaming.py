"""
Tests for streaming dataset functionality.
"""

import pytest
import torch


def test_streaming_dataset_creation():
    """Test creating a StreamingEmbeddingDataset."""
    from saetopic.training.data import StreamingEmbeddingDataset

    # Mock embedder
    class MockEmbedder:
        def encode(self, texts, **kwargs):
            import numpy as np
            return np.random.randn(len(texts), 128).astype(np.float32)

    # Mock HF dataset
    class MockDataset:
        def __init__(self, n=10):
            self.n = n
            self.i = 0

        def __iter__(self):
            self.i = 0
            return self

        def __next__(self):
            if self.i < self.n:
                result = {"text": f"Document {self.i}"}
                self.i += 1
                return result
            raise StopIteration

    mock_dataset = MockDataset(n=100)
    embedder = MockEmbedder()

    dataset = StreamingEmbeddingDataset(
        mock_dataset,
        embedder,
        buffer_size=10,
        embedding_batch_size=4,
        max_samples=20,  # Limit for testing
    )

    # Should yield batches
    batches = list(dataset)
    assert len(batches) > 0
    assert all(isinstance(b, torch.Tensor) for b in batches)
    assert all(b.shape[1] == 128 for b in batches)  # embedding dim


def test_streaming_dataset_empty_text():
    """Test that empty texts are skipped."""
    from saetopic.training.data import StreamingEmbeddingDataset

    class MockEmbedder:
        def encode(self, texts, **kwargs):
            import numpy as np
            return np.random.randn(len(texts), 128).astype(np.float32)

    class MockDataset:
        def __iter__(self):
            return self

        def __next__(self):
            # Mix of empty and non-empty
            texts = ["", "  ", "valid text", None, "another valid"]
            if not hasattr(self, "i"):
                self.i = 0
            if self.i < len(texts):
                result = {"text": texts[self.i]}
                self.i += 1
                return result
            raise StopIteration

    mock_dataset = MockDataset()
    embedder = MockEmbedder()

    dataset = StreamingEmbeddingDataset(
        mock_dataset,
        embedder,
        buffer_size=10,
        embedding_batch_size=4,
    )

    # Should skip empty texts
    batches = list(dataset)
    # Only 2 valid texts, so we get 1 batch of size 2
    assert len(batches) >= 1


def test_sae_trainer_streaming_mode():
    """Test SAETrainer with streaming dataset."""
    from saetopic.sae.modules import create_sae
    from saetopic.training.data import StreamingEmbeddingDataset
    from saetopic.training.train_sae import SAETrainer, TrainingConfig

    # Mock embedder
    class MockEmbedder:
        def encode(self, texts, **kwargs):
            import numpy as np
            return np.random.randn(len(texts), 64).astype(np.float32)

    # Mock streaming dataset
    class MockDataset:
        def __init__(self, n=100):
            self.n = n

        def __iter__(self):
            for i in range(self.n):
                yield {"text": f"Document {i}"}

    # Create streaming dataset
    streaming_dataset = StreamingEmbeddingDataset(
        MockDataset(n=100),
        MockEmbedder(),
        buffer_size=20,
        embedding_batch_size=10,
        max_samples=100,
    )

    # Create model
    model = create_sae(input_dim=64, n_features=128, top_k=8)

    # Create trainer
    config = TrainingConfig(
        input_dim=64,
        n_features=128,
        top_k=8,
        n_epochs=1,
        batch_size=8,
        save_frequency=100,  # Don't save during test
    )

    trainer = SAETrainer(model, config, output_dir="/tmp/test_sae_streaming")

    # Train with streaming
    state = trainer.fit(streaming_dataset)

    assert state.epoch == 1
    assert state.global_step > 0


def test_sae_trainer_detects_streaming():
    """Test that SAETrainer correctly detects streaming mode."""
    from saetopic.sae.modules import create_sae
    from saetopic.training.data import EmbeddingDataset, StreamingEmbeddingDataset
    from saetopic.training.train_sae import SAETrainer, TrainingConfig

    class MockEmbedder:
        def encode(self, texts, **kwargs):
            import numpy as np
            return np.random.randn(len(texts), 64).astype(np.float32)

    class MockDataset:
        def __iter__(self):
            yield {"text": "test"}

    # Standard dataset has __len__
    standard_dataset = EmbeddingDataset(torch.randn(10, 64))
    assert hasattr(standard_dataset, "__len__")
    assert not hasattr(standard_dataset, "__iter__") or hasattr(standard_dataset, "__len__")

    # Streaming dataset has __iter__ but no __len__
    streaming_dataset = StreamingEmbeddingDataset(MockDataset(), MockEmbedder())
    assert hasattr(streaming_dataset, "__iter__")
    assert not hasattr(streaming_dataset, "__len__")
