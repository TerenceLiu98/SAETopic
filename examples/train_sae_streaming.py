"""
Example: Training SAE with streaming embeddings from HuggingFace.

This example shows how to train an SAE without pre-computing all embeddings.
The embeddings are computed on-the-fly during training using HF Datasets streaming mode.
"""

import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from saetopic.training import create_streaming_dataset, train_sae
from saetopic.training.train_sae import TrainingConfig

# Example 1: Basic streaming training
# -------------------------------------------------

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load embedding model
# Note: jina-embeddings-v5-text-small is smaller and faster for testing
# For production, use jinaai/jina-embeddings-v5-text-small
embedder = SentenceTransformer(
    "jinaai/jina-embeddings-v5-text-small",
    trust_remote_code=True,
    device=device,
)

# Create streaming dataset
streaming_dataset = create_streaming_dataset(
    dataset_name="HuggingFaceFW/finewiki",
    split="train",
    embedder=embedder,
    text_column="text",
    buffer_size=50000,  # Shuffle buffer size
    embedding_batch_size=256,  # Batch size for encoding
    max_samples=500000,  # Limit samples for testing (None for unlimited)
)

print(f"Streaming dataset created (embedding_dim will be detected on first batch)")

# Create training config
config = TrainingConfig(
    input_dim=1024,  # Will be auto-detected from embedder
    expansion_factor=32,
    top_k=32,
    architecture="batch_topk",
    learning_rate=1e-4,
    batch_size=256,
    n_epochs=5,  # Fewer epochs for testing
    device="auto",
    seed=42,
    save_frequency=2,
    output_dir="checkpoints/jina-v5-sae-streaming",
    checkpoint_name="jina-v5-sae-small",
    dataset_name="HuggingFaceFW/finewiki",
    dataset_license="CC-BY-SA 4.0 / Apache 2.0",
)

# Train with streaming
trainer = train_sae(
    dataset=streaming_dataset,
    config=config,
)


# Example 2: Custom streaming setup
# -------------------------------------------------

from saetopic.training.data import StreamingEmbeddingDataset

# Load HF dataset with custom settings
hf_ds = load_dataset(
    "HuggingFaceFW/finewiki",
    split="train",
    streaming=True,
)

# Add shuffle buffer at HF level (recommended)
# This shuffles before encoding
hf_ds = hf_ds.shuffle(buffer_size=10000, seed=42)

# Create streaming dataset
streaming_dataset = StreamingEmbeddingDataset(
    hf_dataset=hf_ds,
    embedder=embedder,
    text_column="text",
    buffer_size=10000,  # Additional buffer after HF shuffle
    embedding_batch_size=128,
    normalize=True,
    max_samples=100000,  # Limit for testing
)

# Train
trainer = train_sae(
    dataset=streaming_dataset,
    config=config,
)


# Example 3: CLI usage for streaming training
# -------------------------------------------------

# You can also use the CLI for streaming training:
#
# First, create a simple embedding script:
#   (see examples/compute_embeddings_streaming.py)
#
# Then train:
#   saetopic-train train --embeddings embeddings.npy --output checkpoints/sae
#
# Or for true streaming, modify the CLI to accept embedder directly


# Tips for streaming training:
# -------------------------------------------------

"""
1. Buffer Size:
   - Larger buffer = better shuffling but more memory
   - HF shuffle buffer + dataset buffer = total shuffling capacity
   - Recommended: buffer_size >= 10000

2. Batch Sizes:
   - embedding_batch_size: for encoding (can be larger, limited by GPU/CPU)
   - training batch_size: for SAE training (in config)
   - They can be different!

3. Memory:
   - Streaming uses much less disk space (no pre-computed embeddings)
   - But embedder model stays in memory
   - Use smaller embedding model (nano/small) if GPU memory is limited

4. Performance:
   - Encoding is done on CPU by SentenceTransformers
   - Training is done on GPU if available
   - Consider doing encoding on GPU too for very large datasets

5. Reproducibility:
   - Set seed for both HF shuffle and dataset
   - Each epoch will have different order due to buffering
   - For exact reproducibility, pre-compute and save embeddings
"""


# Example 4: GPU-based encoding for faster streaming
# -------------------------------------------------

# If you have a powerful GPU, encode on GPU too:
"""
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer(
    "jinaai/jina-embeddings-v5-text-small",
    device="cuda",
    model_kwargs={"torch_dtype": torch.float16},  # Use half precision
)

# In StreamingEmbeddingDataset, the encode call will use GPU
# This is much faster but requires more GPU memory
"""
