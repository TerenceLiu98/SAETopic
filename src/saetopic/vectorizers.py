"""
Vectorizer for corpus vocabulary and bag-of-words construction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scipy import sparse


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
    idf_weighting : bool, default=True
        Whether to compute IDF weights (consumed by CorpusAdapter)
    stop_words : str or None, default="english"
        Stop-word list forwarded to CountVectorizer. Use "news20k" for
        20 Newsgroups-style email/forum boilerplate, "wikipedia" for
        Wikipedia-style article boilerplate, "english", None, or a custom list.
    ngram_range : tuple, default=(1, 1)
        N-gram range forwarded to CountVectorizer
    token_pattern : str or None, default=r"(?u)\\b[a-zA-Z][a-zA-Z]+\\b"
        Token pattern forwarded to CountVectorizer. The default keeps only
        alphabetic tokens of length >= 2, dropping pure-number tokens (dates,
        years) that otherwise dominate topic words on Wikipedia-style corpora.
    """

    # Drop pure-number tokens (dates/years) so they don't dominate topic words.
    DEFAULT_TOKEN_PATTERN = r"(?u)\b[a-zA-Z][a-zA-Z]+\b"

    # Extra Wikipedia boilerplate that floods topic words (date-list articles,
    # biography stubs). Unioned with English stopwords when stop_words="wikipedia".
    WIKIPEDIA_EXTRA_STOP_WORDS = frozenset(
        {
            # months / date scaffolding
            "january", "february", "march", "april", "may", "june", "july",
            "august", "september", "october", "november", "december",
            # biography / date-list boilerplate
            "born", "died", "births", "deaths", "events", "incumbents",
            "incumbent", "year", "years", "date", "unknown", "married",
            "son", "sons", "daughter", "daughters", "wife", "husband",
            "father", "mother", "aged", "century", "early", "late",
            "following", "later", "time", "new", "old", "known", "near",
            "place", "day", "today", "total",
            # result/abbreviation scaffolding surfaced by c-TF-IDF
            "did", "advance", "ha", "displaystyle",
            # high-frequency Wikipedia given names that carry no topic signal
            "john", "william", "james", "thomas", "charles", "george",
            "robert", "henry", "richard", "joseph", "edward", "samuel",
            "david", "peter", "paul", "james", "mary", "anne", "elizabeth",
            "margaret", "james", "frederick", "arthur", "albert", "walter",
        }
    )
    NEWS20K_EXTRA_STOP_WORDS = frozenset(
        {
            # contraction fragments from the default alphabetic token pattern
            "don", "doesn", "didn", "isn", "aren", "wasn", "weren", "won",
            "wouldn", "couldn", "shouldn", "haven", "hasn", "hadn", "can",
            "cant", "ll", "ve", "re", "isnt", "arent", "wasnt", "werent",
            # email / quoting / organization boilerplate
            "edu", "com", "org", "net", "gov", "writes", "article", "posting",
            "host", "nntp", "subject", "lines", "organization", "reply",
            "email", "mail", "address", "university",
            # high-frequency conversational words that dominate topic labels
            "just", "like", "know", "think", "people", "time", "does", "did",
            "say", "said", "make", "way", "want", "need", "use", "using",
            "used", "good", "new", "right", "thanks", "work", "problem",
            "problems", "question", "questions", "thing", "things", "point",
            "sure", "going", "got", "really", "probably", "look", "actually",
            "better", "best", "lot", "little", "long", "read", "help",
            # temporal/filler terms
            "year", "years", "day", "days", "week", "weeks", "today",
            "yesterday", "tomorrow", "old", "new",
        }
    )

    def __init__(
        self,
        vocabulary_size: int | None = None,
        min_df: int = 2,
        max_df: float = 0.95,
        idf_weighting: bool = True,
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

    def _build_count_vectorizer(self):
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

        kwargs: dict = dict(
            min_df=self.min_df,
            max_df=self.max_df,
            ngram_range=self.ngram_range,
            token_pattern=self.token_pattern,
            binary=False,
        )
        if self.stop_words == "wikipedia":
            kwargs["stop_words"] = list(ENGLISH_STOP_WORDS | self.WIKIPEDIA_EXTRA_STOP_WORDS)
        elif self.stop_words == "news20k":
            kwargs["stop_words"] = list(ENGLISH_STOP_WORDS | self.NEWS20K_EXTRA_STOP_WORDS)
        else:
            kwargs["stop_words"] = self.stop_words
        if self.vocabulary_size:
            kwargs["max_features"] = self.vocabulary_size
        return CountVectorizer(**kwargs)

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
        self.vectorizer_ = self._build_count_vectorizer()
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
