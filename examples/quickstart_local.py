"""Offline SAETopic quickstart.

This example is intentionally tiny and self-contained: it builds an in-memory
SAE and a deterministic toy embedder so the public API can be exercised without
downloading a checkpoint or embedding model.

Run:

    python examples/quickstart_local.py
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from saetopic import SAETopicModel
from saetopic.sae.modules import BatchTopKSAE

DIM = 16
N_FEATURES = 64
TOP_K = 4


DOCS = [
    "mars rover satellite orbit launch telescope galaxy",
    "space mission rocket orbit astronaut lunar telescope",
    "galaxy telescope satellite space mission launch orbit",
    "football coach striker league match goal championship",
    "team coach football goal striker league tournament",
    "championship match goal team tactics football coach",
    "neural network model training dataset classifier",
    "machine learning model dataset neural classifier",
    "training examples neural network model prediction",
]


def toy_embed(docs: list[str]) -> np.ndarray:
    """Embed documents deterministically from token hashes."""
    vectors = np.zeros((len(docs), DIM), dtype=np.float32)
    for row, doc in enumerate(docs):
        for token in doc.split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:2], "little") % DIM
            sign = 1.0 if digest[2] % 2 == 0 else -1.0
            vectors[row, index] += sign
        norm = np.linalg.norm(vectors[row])
        if norm > 0:
            vectors[row] /= norm
    return vectors


def main() -> None:
    sae = BatchTopKSAE(input_dim=DIM, n_features=N_FEATURES, top_k=TOP_K)
    sae.eval()

    model = SAETopicModel(
        embedding_model=toy_embed,
        sae_model=sae,
        n_topics=3,
        min_df=1,
        corpus_adapter_epochs=2,
        corpus_adapter_batch_size=16,
        activation_batch_size=16,
        device="cpu",
        random_state=42,
    )

    topics, probs = model.fit_transform(DOCS)
    print("topics:", topics)
    print("probabilities shape:", probs.shape)
    print(model.get_topic_info())
    print("topic 0:", model.get_topic(0, top_n=5))

    model.retopic(n_topics=2)
    print("after retopic:")
    print(model.get_topic_info())

    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "saetopic-demo"
        model.save(path)
        loaded = SAETopicModel.load(path)
        print("loaded topics:", loaded.get_topic_info()[["Topic", "Count", "Name"]])


if __name__ == "__main__":
    main()
