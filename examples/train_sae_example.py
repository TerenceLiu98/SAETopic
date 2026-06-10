"""
Example: Training a Sparse Autoencoder for SAETopic.

This example shows how to train an SAE on pre-computed embeddings.
For a complete pipeline including embedding computation, see the
full training example.
"""

import numpy as np
from saetopic.training import train_sae
from saetopic.training.train_sae import TrainingConfig
from saetopic.training.data import EmbeddingDataset

# Example 1: Train from embeddings file
# -------------------------------------------------
# This assumes you have pre-computed embeddings saved as .npy or .pt

# Path to your embeddings file
embeddings_path = "path/to/embeddings.npy"

# Create training configuration
config = TrainingConfig(
    input_dim=1024,  # Jina v5 embedding dimension
    expansion_factor=32,  # n_features = 1024 * 32 = 32768
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

# Train the model
trainer = train_sae(
    embeddings_path=embeddings_path,
    config=config,
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

# Now others can use your checkpoint:
# from saetopic import SAETopicModel
# model = SAETopicModel.from_pretrained("saetopic/jina-v5-sae-small")


# Example 4: Compute embeddings from text first
# -------------------------------------------------
# Complete pipeline from text to trained SAE

from sentence_transformers import SentenceTransformer
import torch 

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


## testing spec
# PYTHONPATH=./src python
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from saetopic.training import create_streaming_dataset, train_sae
from saetopic.training.train_sae import TrainingConfig
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. load embedder
embedder = SentenceTransformer("jinaai/jina-embeddings-v5-text-nano",
    trust_remote_code=True,
    device=device,
    model_kwargs={"dtype": torch.bfloat16} if device.type == "cuda" else {},
    truncate_dim=128          #For mathyoshka embedding, please check: https://sbert.net/examples/sentence_transformer/training/matryoshka/README.html#inference
)

# 2. create streaming dataset
streaming_dataset = create_streaming_dataset(
    dataset_name="HuggingFaceFW/finewiki",
    split="train",
    embedder=embedder,
    buffer_size=1000,         
    embedding_batch_size=32,  
    max_samples=5000,
)

# 3. Training Config
config = TrainingConfig(
    input_dim=128,            # same as the truncate_dim or model's dim
    expansion_factor=4,       
    top_k=16,
    architecture="batch_topk",
    learning_rate=1e-4,
    batch_size=32,            
    n_epochs=1, 
    device="auto",
    seed=42,
    save_frequency=5,
    output_dir="/home/jovyan/helloworld-datavol-1/SAETopic/checkpoints/jina-v5-nano-test",
    checkpoint_name="jina-v5-sae-nano-test",
    dataset_name="HuggingFaceFW/finewiki",
    dataset_license="CC-BY-SA 4.0 / Apache 2.0",
)

trainer = train_sae(dataset=streaming_dataset, config=config)