"""Tests for corpus vectorization utilities."""

from saetopic import vectorizers
from saetopic.vectorizers import CorpusVectorizer


def test_saetm_preprocessing_uses_document_processor(monkeypatch):
    """The saetm mode delegates token filtering to the SAE-TM processor."""

    class FakeSAETMDocumentProcessor:
        def process(self, text):
            del text
            return ["encryption", "clipper", "hockey", "team", "car"]

    monkeypatch.setattr(
        vectorizers,
        "SAETMDocumentProcessor",
        FakeSAETMDocumentProcessor,
    )
    docs = [
        "I don't know just like thanks edu com encryption clipper chip",
        "People think just use edu com hockey team game season",
        "max pl output entry stream buf appears line graphics image",
    ]

    vectorizer = CorpusVectorizer(min_df=1, max_df=1.0, stop_words="saetm")
    vectorizer.fit(docs)

    vocab = set(vectorizer.vocab_)
    assert "encryption" in vocab
    assert "clipper" in vocab
    assert "hockey" in vocab
    assert "team" in vocab
    assert "car" in vocab
    assert "max" not in vocab
    assert "pl" not in vocab
    assert "output" not in vocab
    assert "don" not in vocab
    assert "know" not in vocab
    assert "just" not in vocab
    assert "thanks" not in vocab
    assert "edu" not in vocab
