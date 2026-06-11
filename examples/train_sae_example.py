"""
Example: Training a Sparse Autoencoder for SAETopic.

This example shows how to train an SAE on pre-computed embeddings.
For a complete pipeline including embedding computation, see the
full training example.
"""
# ruff: noqa: E402

import numpy as np
import torch

# Example 1: Complete pipeline - HuggingFace dataset → embeddings → training
# -------------------------------------------------
# Recommended workflow for large-scale training
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from saetopic.training import compute_and_save_embeddings, train_sae
from saetopic.training.data import EmbeddingDataset
from saetopic.training.train_sae import TrainingConfig

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encode_device = (
    [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    if torch.cuda.is_available() and torch.cuda.device_count() > 1
    else None
)
output_dir = "/home/jovyan/helloworld-datavol-1/SAETopic/embeddings"
embeddings_path = f"{output_dir}/finewiki_embeddings.npy"

# Step 1: Load embedding model
# -----------------------------
# jina-embeddings-v5-text-nano: 768-dim, fastest for testing
# jina-embeddings-v5-text-small: 1024-dim, good balance
# jina-embeddings-v5-text-base: 1024-dim, best quality
embedder = SentenceTransformer(
    "jinaai/jina-embeddings-v5-text-small",
    trust_remote_code=True,
    device=device,
    model_kwargs={"dtype": torch.bfloat16} if device.type == "cuda" else {},
    truncate_dim=512            # for matryoshka, please check: https://www.sbert.net/docs/package_reference/sentence_transformer/model.html
)

# FineWiki articles can be long. Keep each embedded chunk bounded so
# encode_batch_size controls memory predictably.
embedder.max_seq_length = 512

# Step 2: Load and stream HuggingFace dataset
# --------------------------------------------
from saetopic.training import create_streaming_dataset

streaming_dataset = create_streaming_dataset(
    dataset_name="HuggingFaceFW/finewiki",
    split="train",
    embedder=embedder,
    buffer_size=1000,           # Shuffle buffer size
    embedding_batch_size=128,   # Text chunks to accumulate before yielding
    encode_batch_size=128,      # Internal batch for embedder.encode() (lower if OOM)
    encode_device=encode_device,# e.g. ["cuda:0", "cuda:1"] for multi-GPU
    encode_chunk_size=128,      # Work distribution size for multi-process encode
    text_chunk_size=512,        # Split long FineWiki articles before embedding
    text_chunk_overlap=32,
    max_samples=100000,         # Adjust based on your needs
)

# Step 3: Compute and save embeddings (one-time setup)
# -----------------------------------------------------
# This saves embeddings to disk so you can train multiple times
# without recomputing embeddings each run.
n_embeddings, embedding_dim = compute_and_save_embeddings(
    dataset=streaming_dataset,
    output_path=embeddings_path,
    chunk_size=10000,
)
print(f"Saved {n_embeddings} embeddings of dimension {embedding_dim}")

# Step 4: Train SAE on saved embeddings
# --------------------------------------
# Now you can experiment with different hyperparameters
# without recomputing embeddings!
config = TrainingConfig(
    input_dim=embedding_dim,  # Auto-detected from saved embeddings
    expansion_factor=32,
    top_k=32,
    architecture="batch_topk",
    learning_rate=1e-4,
    batch_size=256,
    n_epochs=100,
    device="auto",
    seed=42,
    output_dir="checkpoints/jina-v5-sae-small",
    checkpoint_name="jina-v5-sae-small",
    dataset_name="HuggingFaceFW/finewiki",
    dataset_license="CC-BY-SA 4.0 / Apache 2.0",
)

trainer = train_sae(
    embeddings_path=embeddings_path,
    config=config,
    normalize_embeddings=False,  # create_streaming_dataset normalizes before saving
)

# The trained model is now available at:
# trainer.model  # The trained SAE
# trainer.state  # Training history


# Example 2: Train from dataset object
# -------------------------------------------------
# If you have your embeddings in memory or want more control

# Load embeddings into memory
embeddings = np.random.randn(10000, 1024).astype(np.float32)  # Dummy data

# Create dataset
dataset = EmbeddingDataset(embeddings, normalize=True)

# Train with custom config
config = TrainingConfig(
    input_dim=1024,
    n_features=16384,  # Specify directly instead of using expansion_factor
    top_k=16,
    architecture="batch_topk",
    n_epochs=50,
    output_dir="checkpoints/my-sae",
)

trainer = train_sae(dataset=dataset, config=config)


# Example 3: Upload to HuggingFace Hub
# -------------------------------------------------
# After training, upload the checkpoint

from saetopic.hf_utils import upload_checkpoint

# First, make sure you're logged in:
# In terminal: huggingface-cli login

upload_checkpoint(
    checkpoint_dir="checkpoints/jina-v5-sae-small/final",
    repo_id="saetopic/jina-v5-sae-small",
    create_repo=True,
)

# The uploaded folder is a self-contained SAE training checkpoint.
# SAETopicModel.from_pretrained() and end-to-end inference are planned, but not
# implemented yet; use the checkpoint as a training artifact for now.


# Example 4: Compute embeddings from text first
# -------------------------------------------------
# Complete pipeline from text to trained SAE

import torch
from sentence_transformers import SentenceTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load embedding model
embedder = SentenceTransformer(
    "jinaai/jina-embeddings-v5-text-small",
    trust_remote_code=True,
    device=device,
    model_kwargs={"dtype": torch.bfloat16}, # Use bf16 if your GPU supports it
    # config_kwargs={"_attn_implementation": "flash_attention_2"}, # use 'fa2' if you GPU supports it
)

# Load your text corpus
texts = [
    "This is a document about machine learning.",
    "This is a document about climate change.",
    # ... more documents
]

# Compute embeddings
embeddings = embedder.encode(
    texts,
    task="clustering",  # Important for Jina v5
    batch_size=32,
    show_progress_bar=True,
)

# Save embeddings
np.save("my_embeddings.npy", embeddings)

# Now train SAE on these embeddings
config = TrainingConfig(
    input_dim=embeddings.shape[1],
    expansion_factor=32,
    top_k=32,
    n_epochs=100,
    output_dir="checkpoints/my-corpus-sae",
)

trainer = train_sae(
    embeddings_path="my_embeddings.npy",
    config=config,
)


# Example 5: Streaming embeddings from HuggingFace (no pre-computation)
# -------------------------------------------------
# Train an SAE without pre-computing all embeddings.
# The embeddings are computed on-the-fly during training using HF Datasets streaming mode.

import torch
from sentence_transformers import SentenceTransformer

from saetopic.training import compute_and_save_embeddings, create_streaming_dataset, train_sae
from saetopic.training.data import StreamingEmbeddingDataset
from saetopic.training.train_sae import TrainingConfig

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load embedding model
# Note: jina-embeddings-v5-text-nano is fastest for testing
# For production, use jinaai/jina-embeddings-v5-text-small
model_kwargs = {"dtype": torch.bfloat16} if device.type == "cuda" else {}
embedder = SentenceTransformer(
    "jinaai/jina-embeddings-v5-text-nano",
    trust_remote_code=True,
    device=device,
    model_kwargs=model_kwargs,
)

# Option A: Basic streaming training (simple, but slower)
# --------------------------------------------------------
streaming_dataset = create_streaming_dataset(
    dataset_name="HuggingFaceFW/finewiki",
    split="train",
    embedder=embedder,
    text_column="text",
    buffer_size=1000,  # Shuffle buffer size
    embedding_batch_size=64,  # Text chunks to accumulate before yielding
    encode_batch_size=8,  # Internal batch for embedder.encode() (lower if OOM)
    text_chunk_size=512,
    text_chunk_overlap=32,
    max_samples=50000,  # 50k samples for testing
)

print("Streaming dataset created (embedding_dim will be detected on first batch)")

config = TrainingConfig(
    input_dim=768,  # nano model is 768-dim (will be auto-detected)
    expansion_factor=16,
    top_k=16,
    architecture="batch_topk",
    learning_rate=1e-4,
    batch_size=256,
    n_epochs=5,
    device="auto",
    output_dir="checkpoints/jina-v5-sae-nano-test",
)

trainer = train_sae(dataset=streaming_dataset, config=config)


# Option B: Custom streaming setup (more control)
# ------------------------------------------------
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
    buffer_size=1000,
    embedding_batch_size=64,
    encode_batch_size=8,  # Internal batch for embedder.encode() (lower if OOM)
    text_chunk_size=512,
    text_chunk_overlap=32,
    normalize=True,
    max_samples=50000,
    task="clustering",  # Required for Jina v5
)


# Option C: Save embeddings first, then train (RECOMMENDED for development)
# --------------------------------------------------------------------------
# This approach saves embeddings once, then you can train multiple times
# with different hyperparameters without recomputing embeddings.

streaming_dataset = create_streaming_dataset(
    dataset_name="HuggingFaceFW/finewiki",
    split="train",
    embedder=embedder,
    buffer_size=1000,
    embedding_batch_size=64,
    encode_batch_size=8,  # Internal batch for embedder.encode() (lower if OOM)
    text_chunk_size=512,
    text_chunk_overlap=32,
    max_samples=50000,
)

# Step 1: Compute and save embeddings (do this once!)
n_embeddings, embedding_dim = compute_and_save_embeddings(
    dataset=streaming_dataset,
    output_path="data/finewiki_embeddings.npy",
    chunk_size=10000,
)
print(f"Saved {n_embeddings} embeddings of dimension {embedding_dim}")

# Step 2: Train from saved embeddings (fast, repeatable)
# Now you can experiment with different hyperparameters!
config = TrainingConfig(
    input_dim=embedding_dim,  # Auto-detected
    expansion_factor=16,
    top_k=16,
    batch_size=256,  # Larger batch OK since no encoding overhead
    n_epochs=20,  # More epochs since setup is fast
    output_dir="checkpoints/jina-v5-sae-nano",
)

trainer = train_sae(
    embeddings_path="data/finewiki_embeddings.npy",
    config=config,
    normalize_embeddings=False,  # streaming dataset normalized before saving
)


# Tips for streaming training:
# -------------------------------------------------

"""
1. Buffer Size:
   - Larger buffer = better shuffling but more memory
   - HF shuffle buffer + dataset buffer = total shuffling capacity
   - Start with buffer_size=1000 for FineWiki; increase only if memory allows

2. Batch Sizes:
   - embedding_batch_size: for encoding (can be larger, limited by GPU/CPU)
   - encode_batch_size: internal SentenceTransformer batch size; lower if OOM
   - training batch_size: for SAE training (in config)
   - They can be different!

3. Memory:
   - Direct streaming training uses less disk space but keeps embedder in memory
   - Pre-computing embeddings releases the embedder before SAE training
   - Use smaller embedding model (nano/small) if GPU memory is limited
   - Use text_chunk_size/max_seq_length to bound long FineWiki articles

4. Performance:
   - Encoding is done by SentenceTransformers on the device you specify
   - Training is done on GPU if available
   - For large datasets, use encode_device=["cuda:0", "cuda:1"] or CLI --auto-multi-gpu

5. Reproducibility:
   - Set seed for both HF shuffle and create_streaming_dataset(seed=...)
   - Each epoch will have different order due to buffering
   - For exact reproducibility, pre-compute and save embeddings

6. Save embeddings for faster iteration:
   - Use compute_and_save_embeddings() to pre-compute once
   - Then train from memory-mapped .npy for fast hyperparameter tuning
   - If embeddings were saved from create_streaming_dataset(normalize=True),
     pass normalize_embeddings=False to train_sae()
"""


# Example 6: GPU-based encoding for faster streaming
# ----------------------------------------------------
# If you have a powerful GPU, encode on GPU too:
"""
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer(
    "jinaai/jina-embeddings-v5-text-small",
    device="cuda",
    model_kwargs={"dtype": torch.float16},  # Use half precision
)

# In StreamingEmbeddingDataset, the encode call will use GPU
# This is much faster but requires more GPU memory
"""
