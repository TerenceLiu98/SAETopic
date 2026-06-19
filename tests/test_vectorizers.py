"""Tests for corpus vectorization utilities."""

import re
from types import SimpleNamespace

from examples.build_news20k_topic_model import _strip_20newsgroups_metadata
from saetopic import vectorizers
from saetopic.vectorizers import CorpusVectorizer, SAETMDocumentProcessor


def test_saetm_processor_pos_tags_before_filtering():
    """SAE-TM preprocessing POS-tags the full token sequence like the reference."""

    processor = SAETMDocumentProcessor.__new__(SAETMDocumentProcessor)
    observed = {}

    class FakeNltk:
        @staticmethod
        def pos_tag(tokens):
            observed["tokens"] = list(tokens)
            return [
                ("the", "DT"),
                ("cars", "NNS"),
                ("x1", "NN"),
                ("running", "VBG"),
            ]

    class FakeLemmatizer:
        @staticmethod
        def lemmatize(token, pos):
            return f"{token}:{pos}"

    processor.nltk = FakeNltk()
    processor.wordnet = SimpleNamespace(NOUN="n", ADJ="a", VERB="v", ADV="r")
    processor.word_tokenize = lambda text: text.split()
    processor.lemmatizer = FakeLemmatizer()
    processor.stop_words = {"the"}
    processor.wordnet_words = {"cars", "running"}
    processor.ascii_pattern = re.compile(r"^[a-z]+$")
    processor.tag_dict = {
        "J": processor.wordnet.ADJ,
        "N": processor.wordnet.NOUN,
        "V": processor.wordnet.VERB,
        "R": processor.wordnet.ADV,
    }

    lemmas = processor.process("the cars x1 running")

    assert observed["tokens"] == ["the", "cars", "x1", "running"]
    assert lemmas == ["cars:n", "running:v"]


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


def test_saetm_vocabulary_size_uses_document_frequency(monkeypatch):
    """SAE-TM vocab truncation ranks lemmas by document frequency, then token."""

    class FakeSAETMDocumentProcessor:
        def process(self, text):
            return text.split()

    monkeypatch.setattr(
        vectorizers,
        "SAETMDocumentProcessor",
        FakeSAETMDocumentProcessor,
    )
    docs = [
        "alpha alpha alpha rare",
        "beta gamma",
        "beta delta",
    ]

    vectorizer = CorpusVectorizer(
        vocabulary_size=2,
        min_df=1,
        max_df=1.0,
        stop_words="saetm",
    )
    vectorizer.fit(docs)

    assert vectorizer.vocab_ == ["beta", "alpha"]


def test_news20k_metadata_stripping_removes_headers_quotes_and_footer():
    text = """From: user@example.com
Subject: Re: graphics

This line should stay.
> quoted line should go
In article someone wrote something
Another useful line.
--
signature should go
"""

    cleaned = _strip_20newsgroups_metadata(text)

    assert "This line should stay." in cleaned
    assert "Another useful line." in cleaned
    assert "From:" not in cleaned
    assert "quoted line" not in cleaned
    assert "signature" not in cleaned
