"""Tests for pretrain workflow helpers."""

from __future__ import annotations

import sys
import types

import numpy as np


def test_build_embedder_vllm_backend(monkeypatch):
    """vLLM embedding backend should expose a SentenceTransformer-like API."""
    import pretrain.run as pretrain_run

    captured: dict[str, object] = {}

    class FakePoolingParams:
        def __init__(self, dimensions=None):
            self.dimensions = dimensions

    class FakeLLM:
        def __init__(self, **kwargs):
            captured["llm_kwargs"] = kwargs
            self.requests = []
            self.pooling_params = None

        def embed(self, requests, pooling_params=None):
            self.requests.append(requests)
            self.pooling_params = pooling_params
            dim = pooling_params.dimensions if pooling_params is not None else 4
            return [
                types.SimpleNamespace(
                    outputs=types.SimpleNamespace(
                        embedding=np.full(dim, index, dtype=np.float32).tolist()
                    )
                )
                for index, _ in enumerate(requests)
            ]

    monkeypatch.setitem(
        sys.modules,
        "vllm",
        types.SimpleNamespace(LLM=FakeLLM, PoolingParams=FakePoolingParams),
    )

    embedder = pretrain_run.build_embedder(
        {
            "embedding_model": {
                "name": "jinaai/jina-embeddings-v5-omni-nano",
                "inference_backend": "vllm",
                "trust_remote_code": True,
                "dtype": "bfloat16",
                "truncate_dim": 3,
                "max_seq_length": 1024,
                "model_kwargs": {
                    "default_task": "clustering",
                    "modality": "text",
                },
                "vllm": {
                    "tensor_parallel_size": 1,
                    "gpu_memory_utilization": 0.8,
                    "disable_custom_all_reduce": True,
                },
            }
        }
    )

    embeddings = embedder.encode_document(
        ["alpha", "Document: beta"],
        batch_size=128,
        device=["cuda:0", "cuda:1"],
        chunk_size=512,
    )

    llm_kwargs = captured["llm_kwargs"]
    assert llm_kwargs["model"] == "jinaai/jina-embeddings-v5-omni-nano"
    assert llm_kwargs["runner"] == "pooling"
    assert llm_kwargs["trust_remote_code"] is True
    assert llm_kwargs["tensor_parallel_size"] == 1
    assert llm_kwargs["gpu_memory_utilization"] == 0.8
    assert llm_kwargs["disable_custom_all_reduce"] is True
    assert llm_kwargs["language_model_only"] is True
    assert llm_kwargs["skip_mm_profiling"] is True
    assert llm_kwargs["dtype"] == "bfloat16"
    assert llm_kwargs["max_model_len"] == 1024
    assert llm_kwargs["hf_overrides"] == {
        "task": "clustering",
    }

    assert embedder.llm.requests == [
        [
            {"prompt": "Document: alpha"},
            {"prompt": "Document: beta"},
        ]
    ]
    assert embedder.llm.pooling_params.dimensions == 3
    assert embeddings.shape == (2, 3)
    assert embeddings.dtype == np.float32
