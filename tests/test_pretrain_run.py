"""Tests for pretraining pipeline orchestration helpers."""

import json

import numpy as np
import pandas as pd

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
