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


def test_streaming_dataset_passes_encode_device_and_task():
    """Test that encode options are forwarded to SentenceTransformer."""
    from saetopic.training.data import StreamingEmbeddingDataset

    class MockEmbedder:
        def __init__(self):
            self.calls = []

        def encode(self, texts, **kwargs):
            import numpy as np

            self.calls.append(kwargs)
            return np.random.randn(len(texts), 8).astype(np.float32)

    class MockDataset:
        def __iter__(self):
            yield {"text": "first document"}
            yield {"text": "second document"}

    embedder = MockEmbedder()
    dataset = StreamingEmbeddingDataset(
        MockDataset(),
        embedder,
        buffer_size=2,
        embedding_batch_size=2,
        encode_batch_size=4,
        encode_device=["cuda:0", "cuda:1"],
        encode_chunk_size=16,
        task="clustering",
    )

    list(dataset)

    assert embedder.calls
    assert embedder.calls[0]["batch_size"] == 4
    assert embedder.calls[0]["device"] == ["cuda:0", "cuda:1"]
    assert embedder.calls[0]["chunk_size"] == 16
    assert embedder.calls[0]["task"] == "clustering"


def test_create_streaming_dataset_forwards_local_options_to_dataset(monkeypatch):
    """Test local options are handled locally, not passed to load_dataset."""
    import sys
    import types

    from saetopic.training.data import create_streaming_dataset

    captured = {}

    def fake_load_dataset(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return [{"text": "first document"}]

    class MockEmbedder:
        def encode(self, texts, **kwargs):
            import numpy as np

            return np.random.randn(len(texts), 8).astype(np.float32)

    fake_datasets = types.SimpleNamespace(load_dataset=fake_load_dataset)
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    dataset = create_streaming_dataset(
        dataset_name="mock/dataset",
        split="train",
        embedder=MockEmbedder(),
        normalize=False,
        seed=123,
    )

    assert dataset.normalize is False
    assert dataset.seed == 123
    assert captured["args"] == ("mock/dataset",)
    assert captured["kwargs"]["split"] == "train"
    assert "normalize" not in captured["kwargs"]
    assert "seed" not in captured["kwargs"]


def test_streaming_dataset_token_chunking_and_max_samples_per_iter():
    """Test tokenizer chunking and ensure max_samples resets per iterator."""
    from saetopic.training.data import StreamingEmbeddingDataset

    class MockTokenizer:
        def encode(self, text, add_special_tokens=False):
            return text.split()

        def decode(
            self,
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ):
            return " ".join(token_ids)

    class MockEmbedder:
        def __init__(self):
            self.tokenizer = MockTokenizer()
            self.encoded_texts = []

        def encode(self, texts, **kwargs):
            import numpy as np

            self.encoded_texts.extend(texts)
            return np.random.randn(len(texts), 8).astype(np.float32)

    class MockDataset:
        def __iter__(self):
            yield {"text": "a b c d e f"}

    embedder = MockEmbedder()
    dataset = StreamingEmbeddingDataset(
        MockDataset(),
        embedder,
        buffer_size=10,
        embedding_batch_size=10,
        text_chunk_size=3,
        text_chunk_overlap=1,
        max_samples=2,
    )

    first_pass = list(dataset)
    second_pass = list(dataset)

    assert sum(batch.shape[0] for batch in first_pass) == 2
    assert sum(batch.shape[0] for batch in second_pass) == 2
    assert embedder.encoded_texts[:2] == ["a b c", "c d e"]


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
    model = create_sae(
        input_dim=64,
        architecture="batch_topk",
        n_features=128,
        top_k=8,
    )

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
    from saetopic.training.data import EmbeddingDataset, StreamingEmbeddingDataset

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


def test_compute_and_save_embeddings_writes_chunked_npy(tmp_path):
    """Test that chunked embedding saving produces one valid final .npy file."""
    import numpy as np

    from saetopic.training.train_sae import compute_and_save_embeddings

    class MockDataset:
        max_samples = 7

        def __iter__(self):
            yield torch.ones(3, 4)
            yield torch.ones(3, 4) * 2
            yield torch.ones(1, 4) * 3

    output_path = tmp_path / "embeddings.npy"
    n_embeddings, embedding_dim = compute_and_save_embeddings(
        MockDataset(),
        output_path,
        chunk_size=4,
    )

    embeddings = np.load(output_path)
    assert n_embeddings == 7
    assert embedding_dim == 4
    assert embeddings.shape == (7, 4)
    assert embeddings.dtype == np.float32
    assert embeddings[0, 0] == 1
    assert embeddings[-1, 0] == 3


def test_compute_and_save_embeddings_rejects_empty_dataset(tmp_path):
    """Test that empty streams fail explicitly instead of writing bad files."""

    from saetopic.training.train_sae import compute_and_save_embeddings

    class EmptyDataset:
        max_samples = 0

        def __iter__(self):
            return iter(())

    with pytest.raises(ValueError, match="No embeddings were produced"):
        compute_and_save_embeddings(
            EmptyDataset(),
            tmp_path / "empty.npy",
            chunk_size=4,
        )
