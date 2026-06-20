"""Tests for pretraining pipeline orchestration helpers."""

import csv
import json
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix, load_npz, save_npz

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
                    [0.5, 2.0, 0.0, 1.0],
                    [3.0, 0.0, 0.0, 0.1],
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
            "top_k": 3,
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
    class_summary = pd.read_csv(out_dir / "class_summary.csv")
    class_distribution = pd.read_csv(out_dir / "class_feature_distribution.csv")
    visual_bow = load_npz(out_dir / "visual_bow.npz")
    visual_bow_meta = json.loads((out_dir / "visual_bow_meta.json").read_text(encoding="utf-8"))
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))

    assert samples[0]["id"] == "a"
    assert samples[0]["top_features"] == [
        {"activation": 2.0, "feature": 1},
        {"activation": 1.0, "feature": 3},
        {"activation": 0.5, "feature": 0},
    ]
    assert samples[1]["top_features"][0] == {"activation": 3.0, "feature": 0}
    assert all(item["activation"] > 0 for sample in samples for item in sample["top_features"])
    assert feature_summary["feature"].tolist() == [0, 3, 1]
    assert visual_bow.shape == (2, 4)
    assert visual_bow[0, 1] == 2.0
    assert visual_bow[1, 0] == 3.0
    assert visual_bow_meta["row_ids"] == ["a", "b"]
    assert visual_bow_meta["labels"] == ["first", None]
    assert set(class_summary["label"]) == {"first", "__unlabeled__"}
    first_rows = class_distribution[class_distribution["label"] == "first"]
    assert first_rows["feature"].tolist() == [1, 3, 0]
    assert first_rows["image_fraction"].tolist() == [1.0, 1.0, 1.0]
    assert summary["n_images"] == 2
    assert summary["mean_reconstruction_cosine"] == 1.0
    assert summary["outputs"]["visual_bow"].endswith("visual_bow.npz")
    assert summary["outputs"]["class_feature_distribution"].endswith(
        "class_feature_distribution.csv"
    )


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


def test_vision_vocab_and_bow_use_dinov2_patch_clusters(monkeypatch, tmp_path):
    records = [
        {"id": "a", "image": "a.jpg", "label": "alpha"},
        {"id": "b", "image": "b.jpg", "label": "beta"},
        {"id": "c", "image": "c.jpg", "label": "alpha"},
    ]

    def fake_patch_batches(records_arg, tokenizer_cfg):
        del tokenizer_cfg
        assert records_arg == records
        yield 0, np.asarray(
            [
                [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]],
                [[1.0, 0.0], [0.8, 0.2], [1.0, 0.0], [0.2, 0.8]],
            ],
            dtype=np.float32,
        )
        yield 2, np.asarray(
            [
                [[0.0, 1.0], [0.1, 0.9], [0.0, 1.0], [0.9, 0.1]],
            ],
            dtype=np.float32,
        )

    monkeypatch.setattr(pretrain_run, "load_vision_probe_inputs", lambda cfg: records)
    monkeypatch.setattr(pretrain_run, "_iter_dinov2_patch_batches", fake_patch_batches)

    config = {
        "project": {"seed": 7},
        "vision": {
            "out_dir": str(tmp_path / "vision"),
            "visual_tokenizer": {
                "codebook_size": 2,
                "kmeans_batch_size": 4,
                "max_patch_samples": 12,
                "seed": 7,
            },
        },
    }

    pretrain_run.run_vision_vocab(config)
    pretrain_run.run_vision_bow(config)

    out_dir = tmp_path / "vision"
    centroids = np.load(out_dir / "visual_vocab_centroids.npy")
    visual_bow = load_npz(out_dir / "visual_bow.npz")
    meta = json.loads((out_dir / "visual_bow_meta.json").read_text(encoding="utf-8"))
    ctfidf = pd.read_csv(out_dir / "class_visual_ctfidf.csv")

    assert centroids.shape == (2, 2)
    assert visual_bow.shape == (3, 2)
    assert visual_bow.sum(axis=1).A1.tolist() == [4.0, 4.0, 4.0]
    assert meta["source"] == "dinov2_patch_kmeans"
    assert meta["row_ids"] == ["a", "b", "c"]
    assert set(ctfidf["label"]) == {"alpha", "beta"}


def test_vision_emission_writes_b_vis_artifacts(monkeypatch, tmp_path):
    records = [
        {"id": "a", "image": "image-a.jpg", "label": "alpha"},
        {"id": "b", "image": "image-b.jpg", "label": "beta"},
    ]
    out_dir = tmp_path / "vision"
    out_dir.mkdir()
    save_npz(
        out_dir / "visual_bow.npz",
        csr_matrix(np.asarray([[2.0, 0.0, 1.0], [0.0, 3.0, 1.0]], dtype=np.float32)),
    )

    class FakeEmbedder:
        def encode_document(self, payloads, **kwargs):
            del kwargs
            assert payloads == ["image-a.jpg", "image-b.jpg"]
            return np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

    class FakeSAE(torch.nn.Module):
        n_features = 4

        def encode(self, x):
            return torch.asarray(
                [
                    [1.0, 0.0, 0.5, 0.0],
                    [0.0, 1.0, 0.0, 0.5],
                ],
                device=x.device,
                dtype=torch.float32,
            )[: x.shape[0]]

    class FakeCheckpoint:
        embedding_dim = 3

        def get_model(self):
            return FakeSAE()

    monkeypatch.setattr(pretrain_run, "load_vision_probe_inputs", lambda cfg: records)
    monkeypatch.setattr(pretrain_run, "build_vision_embedder", lambda config: FakeEmbedder())
    monkeypatch.setattr(
        pretrain_run.SAECheckpoint,
        "from_pretrained",
        classmethod(lambda cls, checkpoint: FakeCheckpoint()),
    )
    monkeypatch.setattr(pretrain_run, "checkpoint_path", lambda config: tmp_path / "checkpoint")

    config = {
        "project": {"seed": 7},
        "embedding_model": {"batch_size": 2, "model_kwargs": {}},
        "topics": {"device": "cpu"},
        "sae": {"training": {"output_dir": str(tmp_path / "checkpoint")}},
        "vision": {
            "out_dir": str(out_dir),
            "device": "cpu",
            "batch_size": 2,
            "normalize": False,
            "emission": {
                "corpus_adapter_epochs": 1,
                "corpus_adapter_batch_size": 2,
                "idf_weighting": True,
                "theta_batch_size": 2,
                "theta_top_k": 2,
                "verbose": False,
            },
        },
    }

    pretrain_run.run_vision_emission(config)

    emission = torch.load(out_dir / "visual_emission_probabilities.pt", map_location="cpu")
    feature_probs = torch.load(out_dir / "visual_feature_probabilities.pt", map_location="cpu")
    theta = load_npz(out_dir / "theta_sae_csr.npz")
    top_visual_words = pd.read_csv(out_dir / "feature_top_visual_words.csv")
    summary = json.loads((out_dir / "vision_emission_summary.json").read_text(encoding="utf-8"))

    assert emission["B"].shape == (4, 3)
    assert emission["vocab_type"] == "dinov2_kmeans_visual_words"
    assert feature_probs["theta_avg"].shape == (4,)
    assert theta.shape == (2, 4)
    assert top_visual_words["visual_word"].max() < 3
    assert summary["outputs"]["visual_emission_probabilities"].endswith(
        "visual_emission_probabilities.pt"
    )


def test_vision_topics_merges_visual_sae_atoms(tmp_path):
    out_dir = tmp_path / "vision"
    out_dir.mkdir()
    np.save(
        out_dir / "visual_vocab_centroids.npy",
        np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.8, 0.2],
            ],
            dtype=np.float32,
        ),
    )
    torch.save(
        {
            "B": torch.asarray(
                [
                    [0.7, 0.1, 0.2],
                    [0.6, 0.2, 0.2],
                    [0.1, 0.7, 0.2],
                    [0.2, 0.6, 0.2],
                ],
                dtype=torch.float32,
            )
        },
        out_dir / "visual_emission_probabilities.pt",
    )
    torch.save(
        {"theta_avg": torch.asarray([0.4, 0.3, 0.2, 0.1], dtype=torch.float32)},
        out_dir / "visual_feature_probabilities.pt",
    )
    save_npz(
        out_dir / "theta_sae_csr.npz",
        csr_matrix(
            np.asarray(
                [
                    [0.8, 0.2, 0.0, 0.0],
                    [0.0, 0.0, 0.6, 0.4],
                ],
                dtype=np.float32,
            )
        ),
    )

    config = {
        "project": {"seed": 7},
        "vision": {
            "out_dir": str(out_dir),
            "n_topics": [2],
            "topic_embedding_sparsity": 1.0,
            "top_visual_words": 2,
            "max_topic_features": 3,
            "min_emission_entropy_gap": 0.01,
        },
    }

    pretrain_run.run_vision_topics(config)

    topic_dir = out_dir / "topics_2"
    clusters = pd.read_csv(topic_dir / "clusters.csv")
    theta_topic = load_npz(topic_dir / "theta_topic_csr.npz")
    summary = json.loads((topic_dir / "vision_topics_summary.json").read_text(encoding="utf-8"))

    assert len(clusters) == 2
    assert sorted(clusters["cluster_size"].tolist()) == [1, 2]
    assert theta_topic.shape == (2, 2)
    assert summary["n_topics"] == 2
    assert summary["n_active_features"] == 4
    assert summary["n_topic_features"] == 3
    assert summary["feature_filtering"]["max_topic_features"] == 3
    assert summary["outputs"]["clusters"].endswith("clusters.csv")


def test_vision_visualize_writes_topic_contact_sheets(monkeypatch, tmp_path):
    image_paths = []
    for idx in range(3):
        path = tmp_path / f"image_{idx}.jpg"
        path.write_bytes(b"fake-image")
        image_paths.append(path)

    out_dir = tmp_path / "vision"
    topic_dir = out_dir / "topics_2"
    topic_dir.mkdir(parents=True)
    save_npz(
        out_dir / "visual_bow.npz",
        csr_matrix(
            np.asarray(
                [
                    [4.0, 0.0, 1.0],
                    [0.0, 3.0, 1.0],
                    [2.0, 1.0, 0.0],
                ],
                dtype=np.float32,
            )
        ),
    )
    (out_dir / "visual_bow_meta.json").write_text(
        json.dumps(
            {
                "row_ids": ["a", "b", "c"],
                "labels": ["alpha", "beta", "alpha"],
            }
        ),
        encoding="utf-8",
    )
    save_npz(
        topic_dir / "theta_topic_csr.npz",
        csr_matrix(np.asarray([[0.8, 0.1], [0.0, 0.9], [0.6, 0.0]], dtype=np.float32)),
    )
    with (topic_dir / "clusters.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "cluster_id",
                "cluster_size",
                "cluster_prob",
                "cluster_ratio",
                "top_visual_words",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "cluster_id": 0,
                "cluster_size": 2,
                "cluster_prob": 0.7,
                "cluster_ratio": 0.5,
                "top_visual_words": "0, 2",
            }
        )

    config = {
        "vision": {
            "out_dir": str(out_dir),
            "inputs": [
                {"id": "a", "image": str(image_paths[0]), "label": "alpha"},
                {"id": "b", "image": str(image_paths[1]), "label": "beta"},
                {"id": "c", "image": str(image_paths[2]), "label": "alpha"},
            ],
            "visualize": {
                "n_topics": 2,
                "top_topics": 1,
                "top_visual_words_per_topic": 2,
                "visual_word_examples": 2,
                "topic_image_examples": 2,
                "thumb_size": 32,
                "columns": 2,
                "patch_representatives": False,
            },
        },
    }

    class FakeSheet:
        def save(self, path, quality=90):
            del quality
            path.write_bytes(b"fake-jpeg")

    monkeypatch.setattr(pretrain_run, "_image_from_vision_record", lambda record: object())
    monkeypatch.setattr(pretrain_run, "_make_contact_sheet", lambda cells, **kwargs: FakeSheet())
    pretrain_run.run_vision_visualize(config)

    viz_dir = out_dir / "visualizations" / "topics_2"
    summary = json.loads((viz_dir / "vision_visualize_summary.json").read_text(encoding="utf-8"))

    assert (viz_dir / "index.html").exists()
    assert (viz_dir / "topic_0_images.jpg").exists()
    assert (viz_dir / "visual_words" / "visual_word_0.jpg").exists()
    assert (viz_dir / "visual_words" / "visual_word_2.jpg").exists()
    assert summary["top_topics"] == 1
    assert summary["patch_representatives"] is False
