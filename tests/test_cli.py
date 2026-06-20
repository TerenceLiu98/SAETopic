"""Tests for the public saetopic CLI."""

from __future__ import annotations

import numpy as np
import pandas as pd

from saetopic import cli


class _DummyModel:
    fit_docs: list[str] | None = None
    fit_embeddings_shape: tuple[int, int] | None = None
    saved_paths: list[str] = []
    retopiced_to: int | None = None

    @classmethod
    def from_pretrained(cls, model, **kwargs):
        cls.model = model
        cls.kwargs = kwargs
        return cls()

    @classmethod
    def load(cls, model):
        cls.loaded_model = model
        return cls()

    def fit_transform(self, docs, embeddings=None):
        type(self).fit_docs = list(docs)
        type(self).fit_embeddings_shape = None if embeddings is None else embeddings.shape
        return [0 for _ in docs], np.ones((len(docs), 1), dtype=np.float32)

    def save(self, path):
        from pathlib import Path

        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        type(self).saved_paths.append(str(out))

    def get_topic_info(self):
        return pd.DataFrame(
            {
                "Topic": [0],
                "Count": [2],
                "Name": ["0_alpha_beta"],
                "Top_Words": ["alpha, beta"],
            }
        )

    def get_document_info(self):
        return pd.DataFrame(
            {
                "Document": ["alpha beta", "gamma delta"],
                "Topic": [0, 0],
                "Probability": [1.0, 1.0],
            }
        )

    def get_topics(self, top_n=10):
        return {0: [("alpha", 1.0), ("beta", 0.5)][:top_n]}

    def retopic(self, n_topics):
        type(self).retopiced_to = n_topics
        return self


def test_cli_fit_reads_csv_and_saves_outputs(monkeypatch, tmp_path):
    """saetopic fit reads documents, optional embeddings, and writes artifacts."""
    monkeypatch.setattr(cli, "SAETopicModel", _DummyModel)
    _DummyModel.saved_paths = []

    input_path = tmp_path / "docs.csv"
    pd.DataFrame({"text": ["alpha beta", "gamma delta"]}).to_csv(input_path, index=False)
    embeddings_path = tmp_path / "embeddings.npy"
    np.save(embeddings_path, np.ones((2, 4), dtype=np.float32))
    output = tmp_path / "model"

    cli.main(
        [
            "fit",
            "--input",
            str(input_path),
            "--model",
            "checkpoint",
            "--embeddings",
            str(embeddings_path),
            "--n-topics",
            "1",
            "--output",
            str(output),
            "--min-df",
            "1",
        ]
    )

    assert _DummyModel.fit_docs == ["alpha beta", "gamma delta"]
    assert _DummyModel.fit_embeddings_shape == (2, 4)
    assert str(output) in _DummyModel.saved_paths
    assert (output / "topic_info.csv").exists()
    assert (output / "document_info.csv").exists()
    assert _DummyModel.kwargs["n_topics"] == 1
    assert _DummyModel.kwargs["min_df"] == 1


def test_cli_topics_writes_csv(monkeypatch, tmp_path):
    """saetopic topics exports topic info from a saved model."""
    monkeypatch.setattr(cli, "SAETopicModel", _DummyModel)
    output = tmp_path / "topics.csv"

    cli.main(["topics", "--model", "saved-model", "--output", str(output), "--top-n", "2"])

    frame = pd.read_csv(output)
    assert frame.loc[0, "Topic"] == 0
    assert frame.loc[0, "Top_Words"] == "alpha, beta"
    assert _DummyModel.loaded_model == "saved-model"


def test_cli_retopic_saves_new_model(monkeypatch, tmp_path):
    """saetopic retopic saves the requested topic granularity."""
    monkeypatch.setattr(cli, "SAETopicModel", _DummyModel)
    _DummyModel.saved_paths = []
    output = tmp_path / "retopiced"

    cli.main(
        [
            "retopic",
            "--model",
            "saved-model",
            "--n-topics",
            "3",
            "--output",
            str(output),
        ]
    )

    assert _DummyModel.retopiced_to == 3
    assert str(output) in _DummyModel.saved_paths
    assert (output / "topic_info.csv").exists()
