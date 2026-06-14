"""Tests for corpus vectorization utilities."""

from saetopic.vectorizers import CorpusVectorizer


def test_news20k_stop_words_remove_email_and_conversation_fillers():
    """The news20k preset should keep topic words and drop common artifacts."""
    docs = [
        "I don't know just like thanks edu com encryption clipper chip",
        "People think just use edu com hockey team game season",
    ]

    vectorizer = CorpusVectorizer(min_df=1, stop_words="news20k")
    vectorizer.fit(docs)

    vocab = set(vectorizer.vocab_)
    assert "encryption" in vocab
    assert "clipper" in vocab
    assert "hockey" in vocab
    assert "team" in vocab
    assert "don" not in vocab
    assert "know" not in vocab
    assert "just" not in vocab
    assert "thanks" not in vocab
    assert "edu" not in vocab
