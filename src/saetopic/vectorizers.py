"""
Vectorizer for corpus vocabulary and bag-of-words construction.
"""

from __future__ import annotations

import collections
import math
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scipy import sparse


class SAETMDocumentProcessor:
    """Tokenization/filtering/lemmatization used by the SAE-TM BoW cache."""

    def __init__(self):
        try:
            import nltk
            from nltk.corpus import stopwords, wordnet
            from nltk.stem import WordNetLemmatizer
            from nltk.tokenize import word_tokenize
        except ImportError as exc:
            raise ImportError(
                "nltk is required for stop_words='saetm'. Install with `pip install nltk`."
            ) from exc

        self.nltk = nltk
        self.wordnet = wordnet
        self.word_tokenize = word_tokenize
        self.lemmatizer = WordNetLemmatizer()
        self._ensure_nltk_data()
        self.stop_words = set(stopwords.words("english"))
        self.wordnet_words = set(wordnet.words())
        self.ascii_pattern = re.compile(r"^[a-z]+$")
        self.tag_dict = {
            "J": wordnet.ADJ,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV,
        }

    def _ensure_nltk_data(self) -> None:
        required_resources = [
            ("tokenizers/punkt", "punkt"),
            ("tokenizers/punkt_tab", "punkt_tab"),
            ("corpora/stopwords", "stopwords"),
            ("corpora/wordnet", "wordnet"),
            ("corpora/omw-1.4", "omw-1.4"),
            ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
            ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
        ]
        for resource, package in required_resources:
            try:
                self.nltk.data.find(resource)
                continue
            except LookupError:
                pass

            try:
                downloaded = self.nltk.download(package, quiet=True)
            except Exception as exc:
                raise RuntimeError(
                    f"Missing NLTK resource {resource!r}, and automatic download "
                    f"failed. Download NLTK data once with "
                    f"`python -m nltk.downloader {' '.join(pkg for _, pkg in required_resources)}` "
                    "or use `--stop-words english` / `--stop-words none`."
                ) from exc
            if not downloaded:
                raise RuntimeError(
                    f"Missing NLTK resource {resource!r}. Download NLTK data once with "
                    f"`python -m nltk.downloader {' '.join(pkg for _, pkg in required_resources)}` "
                    "or use `--stop-words english` / `--stop-words none`."
                )

        optional_packages = [
        ]
        for resource, package in optional_packages:
            try:
                self.nltk.data.find(resource)
            except LookupError:
                try:
                    self.nltk.download(package, quiet=True)
                except Exception:
                    pass

    def _get_pos(self, tag: str) -> str:
        tag_char = tag[0].upper()
        return self.tag_dict.get(tag_char, self.wordnet.NOUN)

    def process(self, text: str) -> list[str]:
        tokens = self.word_tokenize(text.lower())
        filtered_tokens = [
            token
            for token in tokens
            if (
                len(token) > 2
                and self.ascii_pattern.match(token)
                and token not in self.stop_words
                and token in self.wordnet_words
            )
        ]
        if not filtered_tokens:
            return []

        lemmas = []
        for token, tag in self.nltk.pos_tag(filtered_tokens):
            lemmas.append(self.lemmatizer.lemmatize(token, self._get_pos(tag)))
        return lemmas


class CorpusVectorizer:
    """
    Build corpus vocabulary and bag-of-words representations.

    A thin wrapper around scikit-learn's ``CountVectorizer`` that exposes a
    plain ``list[str]`` vocabulary (as expected by ``CorpusAdapter`` and
    ``TopicMerger``) and a sparse bag-of-words matrix.

    Parameters
    ----------
    vocabulary_size : int or None, default=None
        Maximum vocabulary size (maps to ``max_features``; None = unlimited)
    min_df : int, default=2
        Minimum document frequency for vocabulary terms
    max_df : float, default=0.95
        Maximum document frequency (ratio) for vocabulary terms
    idf_weighting : bool, default=False
        Whether to compute IDF weights (consumed by CorpusAdapter)
    stop_words : str or None, default="english"
        Stop-word/preprocessing mode. Use "saetm" to match the SAE-TM BoW
        preprocessing (NLTK stopwords + WordNet filtering + lemmatization),
        "english", None, or a custom list.
    ngram_range : tuple, default=(1, 1)
        N-gram range forwarded to CountVectorizer
    token_pattern : str or None, default=r"(?u)\\b[a-zA-Z][a-zA-Z]+\\b"
        Token pattern forwarded to CountVectorizer. The default keeps only
        alphabetic tokens of length >= 2, dropping pure-number tokens (dates,
        years) that otherwise dominate topic words on Wikipedia-style corpora.
    """

    # Drop pure-number tokens (dates/years) so they don't dominate topic words.
    DEFAULT_TOKEN_PATTERN = r"(?u)\b[a-zA-Z][a-zA-Z]+\b"

    def __init__(
        self,
        vocabulary_size: int | None = None,
        min_df: int = 2,
        max_df: float = 0.95,
        idf_weighting: bool = False,
        stop_words: str | None = "english",
        ngram_range: tuple[int, int] = (1, 1),
        token_pattern: str | None = None,
    ):
        self.vocabulary_size = vocabulary_size
        self.min_df = min_df
        self.max_df = max_df
        self.idf_weighting = idf_weighting
        self.stop_words = stop_words
        self.ngram_range = ngram_range
        self.token_pattern = token_pattern or self.DEFAULT_TOKEN_PATTERN

        # Fitted attributes
        self.vectorizer_ = None
        self.vocab_: list[str] | None = None

    def _build_count_vectorizer(self, vocabulary: list[str] | None = None):
        from sklearn.feature_extraction.text import CountVectorizer

        kwargs: dict = dict(
            min_df=self.min_df,
            max_df=self.max_df,
            ngram_range=self.ngram_range,
            binary=False,
        )
        if self.stop_words == "saetm":
            processor = SAETMDocumentProcessor()
            kwargs["tokenizer"] = processor.process
            kwargs["token_pattern"] = None
            kwargs["lowercase"] = False
            if vocabulary is not None:
                kwargs["vocabulary"] = vocabulary
        else:
            kwargs["token_pattern"] = self.token_pattern
            kwargs["stop_words"] = self.stop_words
        if self.vocabulary_size and vocabulary is None:
            kwargs["max_features"] = self.vocabulary_size
        return CountVectorizer(**kwargs)

    def _build_saetm_vocabulary(self, docs: list[str]) -> list[str] | None:
        if not self.vocabulary_size:
            return None

        processor = SAETMDocumentProcessor()
        doc_freq: collections.Counter[str] = collections.Counter()
        n_docs = len(docs)
        if n_docs == 0:
            return []

        if isinstance(self.min_df, float):
            min_doc_count = math.ceil(self.min_df * n_docs)
        else:
            min_doc_count = int(self.min_df)

        if isinstance(self.max_df, float):
            max_doc_count = math.floor(self.max_df * n_docs)
        else:
            max_doc_count = int(self.max_df)

        for doc in docs:
            tokens = processor.process(doc)
            doc_freq.update(set(tokens))

        items = [
            token
            for token, df in doc_freq.items()
            if df >= min_doc_count and df <= max_doc_count
        ]
        items.sort(key=lambda token: (-doc_freq[token], token))
        return items[: self.vocabulary_size]

    def fit(self, docs: list[str]) -> "CorpusVectorizer":
        """
        Build vocabulary from documents.

        Parameters
        ----------
        docs : list of str
            Corpus documents

        Returns
        -------
        CorpusVectorizer
            Fitted vectorizer instance
        """
        vocabulary = None
        if self.stop_words == "saetm":
            vocabulary = self._build_saetm_vocabulary(docs)
        self.vectorizer_ = self._build_count_vectorizer(vocabulary=vocabulary)
        self.vectorizer_.fit(docs)
        self.vocab_ = self.vectorizer_.get_feature_names_out().tolist()
        return self

    def transform(self, docs: list[str]) -> "sparse.csr_matrix":
        """
        Transform documents to bag-of-words.

        Parameters
        ----------
        docs : list of str
            Documents to vectorize

        Returns
        -------
        sparse.csr_matrix
            Bag-of-words matrix (n_docs x vocab_size)
        """
        if self.vectorizer_ is None:
            raise RuntimeError("CorpusVectorizer must be fitted before transform")
        return self.vectorizer_.transform(docs).tocsr()

    def fit_transform(self, docs: list[str]) -> "sparse.csr_matrix":
        """
        Fit and transform in one step.

        Parameters
        ----------
        docs : list of str
            Corpus documents

        Returns
        -------
        sparse.csr_matrix
            Bag-of-words matrix (n_docs x vocab_size)
        """
        self.vectorizer_ = self._build_count_vectorizer()
        bow = self.vectorizer_.fit_transform(docs).tocsr()
        self.vocab_ = self.vectorizer_.get_feature_names_out().tolist()
        return bow
