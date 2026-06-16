
import numpy as np
import torch

# jina-embeddings-v5-text-small(1024) with 1B
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
output_dir = "/home/jovyan/helloworld-datavol-1/SAETopic"
embeddings_path = f"{output_dir}/embeddings/finewiki_embeddings"
# Step 1: Load embedding model
# -----------------------------
# jina-embeddings-v5-text-small: {32, 64, 128, 256, 512, 768, 1024}-dim
embedder = SentenceTransformer(
    "jinaai/jina-embeddings-v5-text-small",
    trust_remote_code=True,
    device=device,
    model_kwargs={"dtype": torch.bfloat16},
    #config_kwargs={"_attn_implementation": "flash_attention_2"},
    truncate_dim=768            # for matryoshka, please check: https://www.sbert.net/docs/package_reference/sentence_transformer/model.html
)

# FineWiki articles can be long. Keep each embedded chunk bounded so
# encode_batch_size controls memory predictably.
embedder.max_seq_length = 1024

# Step 2: Load and stream HuggingFace dataset
# --------------------------------------------
from saetopic.training import create_streaming_dataset

streaming_dataset = create_streaming_dataset(
    dataset_name="HuggingFaceFW/finewiki",
    subset="en",
    split="train",
    embedder=embedder,
    buffer_size=200000,         # Shuffle buffer size
    embedding_batch_size=2048,  # Text chunks to accumulate before yielding
    encode_batch_size=128,      # Internal batch for embedder.encode() (lower if OOM)
    encode_device=encode_device,       
    encode_chunk_size=128,      # Work distribution size for multi-process encode
    text_chunk_size=1024,       # Split long FineWiki articles before embedding
    text_chunk_overlap=32,
    max_samples=None,           # Adjust based on your needs
)

# Step 3: Compute and save embeddings (one-time setup)
# -----------------------------------------------------
# This saves sharded embeddings to disk so you can train multiple times
# without recomputing embeddings each run. The output is a directory with
# manifest.json and shard_*.npy files.
n_embeddings, embedding_dim = compute_and_save_embeddings(
    dataset=streaming_dataset,
    output_path=embeddings_path,
    chunk_size=50000,
)
print(f"Saved {n_embeddings} embeddings of dimension {embedding_dim}")

# Step 4: Train SAE on saved embeddings
# --------------------------------------
# Now you can experiment with different hyperparameters
# without recomputing embeddings!
full_dataset = EmbeddingDataset.from_file(
    embeddings_path,
    normalize=False,  # create_streaming_dataset normalizes before saving
    mmap_mode="r",
)
val_size = max(1, int(0.05 * len(full_dataset)))
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(
    full_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42),
)

config = TrainingConfig(
    input_dim=embedding_dim,  # Auto-detected from saved embeddings
    expansion_factor=32,
    top_k=32,
    architecture="batch_topk",
    learning_rate=1e-3,
    batch_size=256,
    n_epochs=100,
    save_frequency=1,
    warmup_ratio=0.1,
    aux_loss_weight=1 / 32,
    early_stopping=True,
    early_stopping_patience=5,
    early_stopping_min_delta=1e-4,
    early_stopping_metric="val_reconstruction",
    device="auto",
    seed=42,
    output_dir=f"{output_dir}/checkpoints/jina-v5-sae-small",
    checkpoint_name="jina-v5-sae-small",
    dataset_name="HuggingFaceFW/finewiki",
    dataset_license="CC-BY-SA 4.0 / Apache 2.0",
)
torch.cuda.empty_cache()
trainer = train_sae(
    dataset=train_dataset,
    val_dataset=val_dataset,
    config=config,
)