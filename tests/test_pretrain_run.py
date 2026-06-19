"""Tests for pretraining pipeline orchestration helpers."""

import json
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

import pretrain.run as pretrain_run


def test_build_topic_model_forwards_topics_idf_weighting(monkeypatch, tmp_path):
    captured = {}

    class FakeSAETopicModel:
        @classmethod
        def from_pretrained(cls, checkpoint, **kwargs):
            captured["checkpoint"] = checkpoint
            captured["kwargs"] = kwargs
            return cls()

    monkeypatch.setattr(pretrain_run, "SAETopicModel", FakeSAETopicModel)

    config = {
        "project": {"seed": 42},
        "sae": {"training": {"output_dir": str(tmp_path / "checkpoint")}},
        "topics": {
            "checkpoint_path": None,
            "corpus_adapter_epochs": 30,
            "corpus_adapter_batch_size": 512,
            "activation_batch_size": 256,
            "embedding_batch_size": 64,
            "idf_weighting": True,
            "vocabulary_size": 5000,
            "min_df": 5,
            "max_df": 1.0,
            "stop_words": "saetm",
            "theta_mode": "dense",
            "max_seq_length": 1024,
            "use_ctfidf": False,
            "merge_embedding_model": "word2vec-google-news-300",
            "device": "cpu",
        },
    }

    pretrain_run.build_topic_model(config, n_topics=100)

    assert captured["kwargs"]["idf_weighting"] is True


def test_save_topic_outputs_orders_topics_by_cluster_size(tmp_path):
    class FakeModel:
        n_topics = 3
        vocab_ = ["alpha", "beta"]
        embeddings_ = np.zeros((2, 4), dtype=np.float32)
        feature_activations_ = np.zeros((2, 3), dtype=np.float32)
        theta_avg_ = np.ones(3, dtype=np.float32) / 3
        merge_embedding_model = "word2vec-google-news-300"
        topics_ = None

        def get_topics(self, top_n=20):
            return {
                0: [(f"zero_{i}", 1.0) for i in range(top_n)],
                1: [(f"one_{i}", 1.0) for i in range(top_n)],
                2: [(f"two_{i}", 1.0) for i in range(top_n)],
            }

        def get_topic_info(self):
            return pd.DataFrame(
                {
                    "Topic": [0, 1, 2],
                    "Count": [1, 1, 1],
                    "Name": ["zero", "one", "two"],
                    "Top_Words": ["zero", "one", "two"],
                }
            )

        def get_cluster_info(self):
            return pd.DataFrame(
                {
                    "cluster_id": [0, 1, 2],
                    "cluster_size": [2, 5, 3],
                    "cluster_prob": [0.2, 0.5, 0.3],
                    "cluster_words": ["zero", "one", "two"],
                    "cluster_ratio": [0.1, 0.2, 0.3],
                }
            )

        def get_cluster_to_feature_indices(self):
            return {0: [0], 1: [1], 2: [2]}

    pretrain_run.save_topic_outputs(
        FakeModel(),
        docs=["a", "b"],
        labels=None,
        output_dir=tmp_path,
        elapsed=1.0,
        save_theta_topic=False,
    )

    top_words = (tmp_path / "top_words.txt").read_text(encoding="utf-8").splitlines()
    clusters = pd.read_csv(tmp_path / "clusters.csv")
    topic_info = pd.read_csv(tmp_path / "topic_info.csv")
    mapping = json.loads((tmp_path / "cluster_to_feature_indices.json").read_text())

    assert top_words[0].startswith("one_0")
    assert top_words[1].startswith("two_0")
    assert top_words[2].startswith("zero_0")
    assert clusters["cluster_id"].tolist() == [1, 2, 0]
    assert topic_info["Topic"].tolist() == [1, 2, 0]
    assert mapping == {"0": [0], "1": [1], "2": [2]}


def test_run_vision_probe_writes_sample_and_feature_outputs(monkeypatch, tmp_path):
    class FakeEmbedder:
        def encode_document(self, payloads, **kwargs):
            del kwargs
            assert payloads == ["image-a.jpg", ("caption", "image-b.jpg")]
            return np.asarray(
                [
                    [1.0, 0.0, 0.0, 9.0],
                    [0.0, 1.0, 0.0, 9.0],
                ],
                dtype=np.float32,
            )

    class FakeSAE(torch.nn.Module):
        n_features = 4

        def forward(self, x):
            h = torch.asarray(
                [
                    [0.5, 2.0, 0.1, 1.0],
                    [3.0, 0.2, 0.4, 0.1],
                ],
                device=x.device,
                dtype=torch.float32,
            )
            f = h
            return x.float(), h, f, torch.empty((0, 2), device=x.device)

    class FakeCheckpoint:
        embedding_dim = 3

        def get_model(self):
            return FakeSAE()

    monkeypatch.setattr(pretrain_run, "build_vision_probe_embedder", lambda config: FakeEmbedder())
    monkeypatch.setattr(
        pretrain_run.SAECheckpoint,
        "from_pretrained",
        classmethod(lambda cls, checkpoint: FakeCheckpoint()),
    )
    monkeypatch.setattr(pretrain_run, "checkpoint_path", lambda config: tmp_path / "checkpoint")

    config = {
        "embedding_model": {"batch_size": 2, "model_kwargs": {}},
        "topics": {"device": "cpu"},
        "vision_probe": {
            "out_dir": str(tmp_path / "vision_probe"),
            "inputs": [
                {"id": "a", "image": "image-a.jpg", "label": "first"},
                {"id": "b", "image": "image-b.jpg", "text": "caption"},
            ],
            "batch_size": 2,
            "activation_batch_size": 2,
            "top_k": 2,
            "truncate_dim": 3,
            "device": "cpu",
        },
    }

    pretrain_run.run_vision_probe(config)

    out_dir = tmp_path / "vision_probe"
    samples = [
        json.loads(line)
        for line in (out_dir / "samples.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    feature_summary = pd.read_csv(out_dir / "feature_summary.csv")
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))

    assert samples[0]["id"] == "a"
    assert samples[0]["top_features"] == [
        {"activation": 2.0, "feature": 1},
        {"activation": 1.0, "feature": 3},
    ]
    assert samples[1]["top_features"][0] == {"activation": 3.0, "feature": 0}
    assert feature_summary["feature"].tolist() == [0, 1, 3, 2]
    assert summary["n_images"] == 2
    assert summary["mean_reconstruction_cosine"] == 1.0


def test_load_vision_probe_inputs_reads_hf_dataset_cache(monkeypatch):
    calls = {}

    class FakeImage:
        def __init__(self, name):
            self.name = name
            self.converted = False

        def convert(self, mode):
            assert mode == "RGB"
            self.converted = True
            return self

    class FakeDataset(list):
        features = {"label": SimpleNamespace(names=["class_a", "class_b"])}

    def fake_load_dataset(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return FakeDataset(
            [
                {"image": FakeImage("a0"), "label": 0},
                {"image": FakeImage("a1"), "label": 0},
                {"image": FakeImage("b0"), "label": 1},
            ]
        )

    class FakeDownloadConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_dataset=fake_load_dataset, DownloadConfig=FakeDownloadConfig),
    )

    records = pretrain_run.load_vision_probe_inputs(
        {
            "hf_dataset": "timm/mini-imagenet",
            "hf_split": "test",
            "max_per_label": 1,
            "hf_local_files_only": True,
        }
    )

    assert calls["args"] == ("timm/mini-imagenet",)
    assert calls["kwargs"]["split"] == "test"
    assert calls["kwargs"]["download_config"].kwargs == {"local_files_only": True}
    assert [record["label"] for record in records] == ["class_a", "class_b"]
    assert [record["label_id"] for record in records] == [0, 1]
    assert records[0]["image"] == "hf://timm/mini-imagenet/test/0"
    assert records[0]["_image"].name == "a0"
    assert records[0]["_image"].converted is True
