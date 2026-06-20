"""
Tests for SAETopicModel.

The implemented API (from_pretrained / fit / fit_transform / retopic /
transform / save / load / get_topic*) is exercised with a tiny in-memory SAE
and a fake embedding callable, so the full pipeline can be tested without a
GPU, network, or a real checkpoint. Visualization helpers remain
unimplemented and are still asserted to raise NotImplementedError.
"""

import numpy as np
import pytest

from saetopic import SAETopicModel
from saetopic.sae.modules import BatchTopKSAE

DIM, NFEAT, TOPK = 16, 64, 4


def _fake_embedder_factory():
    """Return a deterministic callable embedding backend (n_docs -> [n, DIM])."""
    rng = np.random.default_rng(0)

    def embed(docs):
        x = rng.standard_normal((len(docs), DIM)).astype(np.float32)
        x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
        return x

    return embed


def _make_model(n_topics: int = 5) -> SAETopicModel:
    """Build a SAETopicModel backed by a tiny in-memory SAE + fake embedder.

    drop_empty_topics=False keeps the requested n_topics exactly so the
    shape-based tests are deterministic; a dedicated test covers the
    empty-topic dropping behavior.
    """
    sae = BatchTopKSAE(input_dim=DIM, n_features=NFEAT, top_k=TOPK)
    sae.eval()
    return SAETopicModel(
        embedding_model=_fake_embedder_factory(),
        sae_model=sae,
        n_topics=n_topics,
        corpus_adapter_epochs=2,
        corpus_adapter_batch_size=32,
        activation_batch_size=32,
        min_df=1,
        drop_empty_topics=False,
        device="cpu",
    )


def _docs() -> list[str]:
    """A small 3-theme corpus with varied per-theme vocabulary."""
    themes = [
        "alpha beta gamma delta epsilon zeta",
        "eta theta iota kappa lambda mu",
        "nu xi omicron pi rho sigma",
    ]
    return [themes[i % 3] for i in range(60)]


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------
def test_model_initialization():
    """Test that SAETopicModel can be initialized with defaults."""
    model = SAETopicModel()

    assert model.embedding_model == "jinaai/jina-embeddings-v5-text-small"
    assert model.embedding_task == "clustering"
    assert model.sae_model == "saetopic/jina-v5-sae-small"
    assert model.merge_embedding_model is None
    assert model.n_topics == 50
    assert model.top_k_features == 32
    assert model.idf_weighting is False
    assert model.use_ctfidf is False
    assert model.drop_empty_topics is False


def test_model_custom_params():
    """Test that SAETopicModel can be initialized with custom parameters."""
    model = SAETopicModel(
        embedding_model="custom-model",
        merge_embedding_model="word2vec-google-news-300",
        n_topics=100,
        random_state=123,
    )

    assert model.embedding_model == "custom-model"
    assert model.merge_embedding_model == "word2vec-google-news-300"
    assert model.n_topics == 100
    assert model.random_state == 123


def test_from_pretrained_loads_sae():
    """from_pretrained loads the SAE and records its dimensions."""
    sae = BatchTopKSAE(input_dim=DIM, n_features=NFEAT, top_k=TOPK)
    model = SAETopicModel.from_pretrained(
        sae,
        embedding_model=_fake_embedder_factory(),
        n_topics=5,
        device="cpu",
    )

    assert model.sae_ is not None
    assert model.sae_input_dim_ == DIM
    assert model.sae_n_features_ == NFEAT
    assert model.top_k_features == TOPK


# --------------------------------------------------------------------------
# Fitting / inference
# --------------------------------------------------------------------------
def test_fit_transform_shapes():
    """fit_transform populates the pipeline matrices with correct shapes."""
    model = _make_model(n_topics=5)
    docs = _docs()

    topics, probs = model.fit_transform(docs)

    n_docs = len(docs)
    assert len(topics) == n_docs
    assert probs.shape == (n_docs, 5)
    assert model.embeddings_.shape == (n_docs, DIM)
    assert model.feature_activations_.shape == (n_docs, NFEAT)
    assert model.theta_avg_.shape == (NFEAT,)
    assert np.isclose(model.theta_avg_.sum(), 1.0, atol=1e-5)
    assert model.adapter_.random_state == model.random_state
    assert model.feature_word_matrix_.shape == (NFEAT, len(model.vocab_))
    assert model.topic_word_matrix_.shape == (5, len(model.vocab_))
    assert model.document_topic_matrix_.shape == (n_docs, 5)


def test_fit_returns_self():
    """fit returns the fitted model instance."""
    model = _make_model(n_topics=4)
    assert model.fit(_docs()) is model


def test_fit_rejects_empty_docs():
    """fit raises on an empty document list."""
    model = _make_model()
    with pytest.raises(ValueError):
        model.fit([])


def test_fit_rejects_embedding_dim_mismatch():
    """fit raises when supplied embeddings dim != SAE input_dim."""
    model = _make_model()
    docs = _docs()
    wrong = np.zeros((len(docs), DIM + 1), dtype=np.float32)
    with pytest.raises(ValueError, match="Embedding dim"):
        model.fit(docs, embeddings=wrong)


# --------------------------------------------------------------------------
# Inspection
# --------------------------------------------------------------------------
def test_get_topic_info_columns():
    """get_topic_info returns a DataFrame with the expected columns."""
    model = _make_model(n_topics=5)
    model.fit(_docs())

    info = model.get_topic_info()
    assert list(info.columns)[:3] == ["Topic", "Count", "Name"]
    assert len(info) == 5


def test_get_topic_returns_word_score_pairs():
    """get_topic returns (word, score) tuples."""
    model = _make_model(n_topics=5)
    model.fit(_docs())

    words = model.get_topic(0, top_n=5)
    assert len(words) == 5
    assert all(isinstance(w, str) and isinstance(s, float) for w, s in words)


def test_get_topics_returns_all():
    """get_topics returns an entry per topic."""
    model = _make_model(n_topics=5)
    model.fit(_docs())

    all_topics = model.get_topics()
    assert set(all_topics.keys()) == set(range(5))


def test_get_topics_respects_top_n():
    """get_topics forwards top_n to each topic."""
    model = _make_model(n_topics=5)
    model.fit(_docs())

    all_topics = model.get_topics(top_n=3)

    assert set(all_topics.keys()) == set(range(5))
    assert all(len(words) == 3 for words in all_topics.values())


def test_get_cluster_info_matches_saetm_columns():
    """SAE-TM-style cluster artifact exposes expected columns."""
    model = _make_model(n_topics=5)
    model.fit(_docs())

    clusters = model.get_cluster_info()

    assert list(clusters.columns) == [
        "cluster_id",
        "cluster_size",
        "cluster_prob",
        "cluster_words",
        "cluster_ratio",
    ]
    assert len(clusters) == 5
    assert clusters["cluster_size"].sum() == (model.topic_atom_clusters_ >= 0).sum()


def test_get_cluster_to_feature_indices_covers_assigned_features():
    """Cluster-to-feature mapping matches the fitted feature cluster labels."""
    model = _make_model(n_topics=5)
    model.fit(_docs())

    mapping = model.get_cluster_to_feature_indices()
    assigned = sorted(feature for features in mapping.values() for feature in features)
    expected = sorted(np.where(model.topic_atom_clusters_ >= 0)[0].tolist())

    assert set(mapping) == set(range(5))
    assert assigned == expected


def test_get_theta_topic_matrix_sparse_matches_dense():
    """SAE-TM-style theta-topic matrix can be exported as CSR."""
    model = _make_model(n_topics=5)
    model.fit(_docs())

    dense = model.get_theta_topic_matrix(normalize=False, sparse=False)
    sparse = model.get_theta_topic_matrix(normalize=False, sparse=True)

    assert sparse.shape == dense.shape == (len(_docs()), 5)
    assert np.allclose(sparse.toarray(), dense, atol=1e-6)


def test_unfitted_raises():
    """Inspection methods raise before fit."""
    model = _make_model()
    with pytest.raises(RuntimeError):
        model.get_topic_info()
    with pytest.raises(RuntimeError):
        model.get_topic(0)


def test_ctfidf_computed():
    """c-TF-IDF (distinctiveness-weighted) scores are computed after fit."""
    sae = BatchTopKSAE(input_dim=DIM, n_features=NFEAT, top_k=TOPK)
    sae.eval()
    model = SAETopicModel(
        embedding_model=_fake_embedder_factory(),
        sae_model=sae,
        n_topics=4,
        corpus_adapter_epochs=2,
        corpus_adapter_batch_size=32,
        activation_batch_size=32,
        min_df=1,
        use_ctfidf=True,
        drop_empty_topics=False,
        device="cpu",
    )
    model.fit(_docs())

    assert model.ctfidf_ is not None
    assert model.ctfidf_.shape == (4, len(model.vocab_))


def test_drop_empty_topics_removes_zeros():
    """Topics with no assigned documents are dropped by default."""
    sae = BatchTopKSAE(input_dim=DIM, n_features=NFEAT, top_k=TOPK)
    sae.eval()
    model = SAETopicModel(
        embedding_model=_fake_embedder_factory(),
        sae_model=sae,
        n_topics=12,  # more topics than the 3-theme corpus can populate
        corpus_adapter_epochs=2,
        corpus_adapter_batch_size=32,
        activation_batch_size=32,
        min_df=1,
        drop_empty_topics=True,
        device="cpu",
    )
    model.fit(_docs())

    info = model.get_topic_info()
    assert (info["Count"] > 0).all()  # no empty topics remain
    assert model.n_topics == len(info) <= 12


# --------------------------------------------------------------------------
# retopic / transform
# --------------------------------------------------------------------------
def test_retopic_changes_n_topics():
    """retopic updates n_topics and the topic matrices."""
    model = _make_model(n_topics=5)
    model.fit(_docs())

    model.retopic(n_topics=3)
    assert model.n_topics == 3
    assert model.topic_word_matrix_.shape[0] == 3
    assert model.document_topic_matrix_.shape[1] == 3


def test_reduce_topics_alias():
    """reduce_topics is an alias for retopic."""
    model = _make_model(n_topics=5)
    model.fit(_docs())

    model.reduce_topics(nr_topics=2)
    assert model.n_topics == 2


def test_retopic_requires_fit():
    """retopic raises before fit."""
    model = _make_model()
    with pytest.raises(RuntimeError):
        model.retopic(n_topics=3)


def test_transform_new_docs():
    """transform maps new documents to topics."""
    model = _make_model(n_topics=4)
    model.fit(_docs())

    new_docs = ["fresh document one", "fresh document two"]
    topics, probs = model.transform(new_docs)

    assert len(topics) == 2
    assert probs.shape == (2, 4)


# --------------------------------------------------------------------------
# Serialization and still-unimplemented visualization helpers
# --------------------------------------------------------------------------
def test_visualize_topics_not_implemented():
    """Test that visualize_topics raises NotImplementedError."""
    model = SAETopicModel()
    with pytest.raises(NotImplementedError, match="visualize_topics is not implemented yet"):
        model.visualize_topics()


def test_save_requires_fit(tmp_path):
    """save rejects unfitted models."""
    model = SAETopicModel()
    with pytest.raises(RuntimeError, match="fitted"):
        model.save(tmp_path / "model")


def test_save_load_roundtrip(tmp_path):
    """A saved fitted model can be loaded for inspection, transform, and retopic."""
    model = _make_model(n_topics=5)
    docs = _docs()
    model.fit(docs)

    path = tmp_path / "saetopic_model"
    model.save(path)

    loaded = SAETopicModel.load(path)

    assert loaded.n_topics == model.n_topics
    assert loaded.vocab_ == model.vocab_
    assert loaded.sae_input_dim_ == model.sae_input_dim_
    assert loaded.sae_n_features_ == model.sae_n_features_
    assert np.allclose(loaded.topic_word_matrix_, model.topic_word_matrix_)
    assert np.allclose(loaded.document_topic_matrix_, model.document_topic_matrix_)
    assert loaded.get_topic_info().equals(model.get_topic_info())
    assert loaded.get_topic(0, top_n=5) == model.get_topic(0, top_n=5)

    new_embeddings = _fake_embedder_factory()(["fresh document one", "fresh document two"])
    topics, probs = loaded.transform(
        ["fresh document one", "fresh document two"],
        embeddings=new_embeddings,
    )
    assert len(topics) == 2
    assert probs.shape == (2, 5)

    loaded.retopic(n_topics=3)
    assert loaded.n_topics == 3
    assert loaded.get_topic_info().shape[0] == 3
