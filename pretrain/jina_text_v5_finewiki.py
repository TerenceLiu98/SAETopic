#!/usr/bin/env python3
"""Compatibility entrypoint for FineWiki/Jina SAE pretraining.

Prefer:

    PYTHONPATH=src python pretrain/run.py --stages embeddings train_sae
"""

from run import main

if __name__ == "__main__":
    main(default_stages=["embeddings", "train_sae"])
