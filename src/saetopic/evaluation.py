"""
Evaluation metrics for topic models.

The paper-aligned metrics are:

- D: topic diversity via average pairwise Word Mover's Distance (WMD)
- CR: LLM coherence rating on topic words
- CI: LLM intruder detection accuracy on topic words

The LLM metrics accept a plain callable so callers can use vLLM, OpenAI, a
local model, or a test double without coupling this package to one provider.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

TopicWords = Mapping[int, Sequence[str | tuple[str, float]]]
LLMCallable = Callable[[str], str]


def _words_only(words: Sequence[str | tuple[str, float]], top_n: int | None = None) -> list[str]:
    selected = words if top_n is None else words[:top_n]
    result = []
    for item in selected:
        word = item[0] if isinstance(item, tuple) else item
        word = str(word).strip()
        if word:
            result.append(word)
    return result


def normalize_topic_words(
    topic_words: TopicWords,
    top_n: int | None = None,
    drop_outlier: bool = True,
) -> dict[int, list[str]]:
    """Normalize topic-word mappings to ``{topic_id: [word, ...]}``."""
    normalized = {}
    for topic_id, words in topic_words.items():
        topic_id = int(topic_id)
        if drop_outlier and topic_id == -1:
            continue
        normalized[topic_id] = _words_only(words, top_n=top_n)
    return normalized


def write_top_words_file(
    topic_words: TopicWords,
    path: str | Path,
    top_n: int = 20,
    drop_outlier: bool = True,
) -> None:
    """Write comma-separated top words, one topic per line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = normalize_topic_words(topic_words, top_n=top_n, drop_outlier=drop_outlier)
    with path.open("w", encoding="utf-8") as f:
        for topic_id in sorted(normalized):
            f.write(", ".join(normalized[topic_id]) + "\n")


def load_top_words_file(path: str | Path, top_n: int | None = None) -> dict[int, list[str]]:
    """Load a comma-separated ``top_words.txt`` file."""
    topics = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            words = [word.strip() for word in line.strip().split(",") if word.strip()]
            if top_n is not None:
                words = words[:top_n]
            if words:
                topics[idx] = words
    return topics


def compute_unique_word_diversity(topic_words: TopicWords, top_n: int = 20) -> float:
    """Compute the standard unique-word-ratio diversity metric."""
    topics = normalize_topic_words(topic_words, top_n=top_n)
    words = [word.lower() for topic in topics.values() for word in topic]
    return len(set(words)) / len(words) if words else 0.0


def _embedding_lookup_from_backend(words: list[str], embedding_model: str | Callable) -> dict[str, np.ndarray]:
    from saetopic.embeddings import EmbeddingBackend

    backend = EmbeddingBackend(model=embedding_model, normalize=True)
    embeddings = backend.embed(words)
    return {word: embeddings[idx] for idx, word in enumerate(words)}


def _resolve_word_embeddings(
    topics: dict[int, list[str]],
    word_embeddings: Mapping[str, Sequence[float]] | None,
    embedding_model: str | Callable | None,
) -> dict[str, np.ndarray]:
    vocab = sorted({word.lower() for words in topics.values() for word in words})
    if not vocab:
        return {}

    if word_embeddings is not None:
        resolved = {}
        available = {
            str(word).lower(): np.asarray(embedding, dtype=np.float32)
            for word, embedding in word_embeddings.items()
        }
        if available:
            mean_embedding = np.mean(np.stack(list(available.values())), axis=0)
        else:
            mean_embedding = np.zeros(1, dtype=np.float32)
        for word in vocab:
            resolved[word] = available.get(word, mean_embedding)
        return resolved

    if embedding_model is None:
        raise ValueError("D/WMD requires either word_embeddings or embedding_model")
    return _embedding_lookup_from_backend(vocab, embedding_model)


def _uniform_wmd(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """Compute WMD for equally weighted word lists.

    SAE-TM evaluates top-k word lists with the same k. In that common case,
    uniform EMD reduces to a minimum-cost bipartite matching.
    """
    if emb_a.size == 0 or emb_b.size == 0:
        return 0.0
    if len(emb_a) != len(emb_b):
        n = min(len(emb_a), len(emb_b))
        emb_a = emb_a[:n]
        emb_b = emb_b[:n]
    cost = np.linalg.norm(emb_a[:, None, :] - emb_b[None, :, :], axis=2)
    row_ind, col_ind = linear_sum_assignment(cost)
    return float(cost[row_ind, col_ind].mean())


def compute_wmd_diversity(
    topic_words: TopicWords,
    top_n: int = 20,
    word_embeddings: Mapping[str, Sequence[float]] | None = None,
    embedding_model: str | Callable | None = None,
) -> float:
    """Compute D: average pairwise WMD between topic top-word lists."""
    topics = normalize_topic_words(topic_words, top_n=top_n)
    if len(topics) < 2:
        return 0.0

    embedding_lookup = _resolve_word_embeddings(topics, word_embeddings, embedding_model)
    topic_embeddings = []
    for words in topics.values():
        embeddings = [embedding_lookup[word.lower()] for word in words if word.lower() in embedding_lookup]
        if embeddings:
            topic_embeddings.append(np.stack(embeddings).astype(np.float32, copy=False))

    if len(topic_embeddings) < 2:
        return 0.0

    total = 0.0
    n_pairs = 0
    for i in range(len(topic_embeddings)):
        for j in range(i + 1, len(topic_embeddings)):
            total += _uniform_wmd(topic_embeddings[i], topic_embeddings[j])
            n_pairs += 1
    return total / n_pairs if n_pairs else 0.0


def compute_diversity(
    topic_words: TopicWords,
    top_n: int = 20,
    method: str = "wmd",
    word_embeddings: Mapping[str, Sequence[float]] | None = None,
    embedding_model: str | Callable | None = None,
) -> float:
    """Compute topic diversity.

    ``method="wmd"`` is the paper-aligned D metric. ``method="unique"`` is
    provided as a lightweight diagnostic.
    """
    if method == "wmd":
        return compute_wmd_diversity(
            topic_words,
            top_n=top_n,
            word_embeddings=word_embeddings,
            embedding_model=embedding_model,
        )
    if method == "unique":
        return compute_unique_word_diversity(topic_words, top_n=top_n)
    raise ValueError(f"Unknown diversity method: {method}")


def create_coherence_prompt(word_list: Sequence[str]) -> str:
    """Create the CR prompt used for LLM topic coherence rating."""
    words_str = ", ".join(word_list)
    return (
        "You are an expert in semantics and lexical relationships. Your task is to evaluate "
        f"the coherence of the following list of words: '{words_str}'.\n\n"
        "Coherence is how well the words belong to a single, clear, and specific category.\n"
        "- A score of 100 means the words are extremely coherent.\n"
        "- A score around 50 means the words are moderately coherent.\n"
        "- A score of 0 means the words are completely unrelated.\n\n"
        'Provide your analysis as a JSON object with two keys: "rationale" and "score". '
        'The "score" must be an integer between 0 and 100. '
        "Your response MUST be only the JSON object and nothing else."
    )


def create_intruder_prompt(word_list: Sequence[str]) -> str:
    """Create the CI prompt used for LLM intruder detection."""
    words_str = ", ".join(word_list)
    return (
        "From the following list of words, identify the single word that does not belong "
        f"with the others. The words are: {words_str}. "
        "Your response must be only the single intruder word and nothing else."
    )


def parse_coherence_score(text: str) -> int | None:
    """Parse a 0-100 coherence score from a JSON LLM response."""
    cleaned = text.strip().replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\b(100|[1-9]?\d)\b", cleaned)
        if not match:
            return None
        return int(match.group(1))
    score = data.get("score")
    if isinstance(score, bool):
        return None
    if isinstance(score, (int, float)) and 0 <= score <= 100:
        return int(score)
    return None


def _normalize_answer(text: str) -> str:
    text = text.strip().lower()
    text = text.replace("```", "")
    text = re.sub(r"[^a-z0-9_\- ]+", "", text)
    return text.strip().split()[0] if text.strip() else ""


def compute_coherence_rating(
    topic_words: TopicWords,
    llm: LLMCallable,
    top_n: int = 20,
    sample_size: int = 5,
    repetitions: int = 3,
    seed: int = 42,
) -> dict[int, float]:
    """Compute CR per topic with an LLM callable."""
    rng = random.Random(seed)
    topics = normalize_topic_words(topic_words, top_n=top_n)
    scores: dict[int, list[int]] = {}

    for topic_id, words in topics.items():
        if len(words) < sample_size:
            continue
        pool = words[:top_n]
        for _ in range(repetitions):
            sample = rng.sample(pool, sample_size)
            rng.shuffle(sample)
            score = parse_coherence_score(llm(create_coherence_prompt(sample)))
            if score is not None:
                scores.setdefault(topic_id, []).append(score)

    return {topic_id: float(np.mean(values)) for topic_id, values in scores.items() if values}


def compute_intruder_detection(
    topic_words: TopicWords,
    llm: LLMCallable,
    top_n: int = 20,
    sample_size: int = 4,
    repetitions: int = 3,
    seed: int = 42,
) -> dict[int, float]:
    """Compute CI per topic with an LLM callable."""
    rng = random.Random(seed)
    topics = normalize_topic_words(topic_words, top_n=top_n)
    topic_ids = sorted(topics)
    if len(topic_ids) < 2:
        return {}

    scores: dict[int, list[float]] = {}
    for topic_id in topic_ids:
        words = topics[topic_id]
        if len(words) < sample_size:
            continue
        intruder_candidates = [other for other in topic_ids if other != topic_id and topics[other]]
        if not intruder_candidates:
            continue
        for _ in range(repetitions):
            other_id = rng.choice(intruder_candidates)
            intruder = rng.choice(topics[other_id])
            sample = rng.sample(words[:top_n], sample_size)
            test_words = sample + [intruder]
            rng.shuffle(test_words)
            predicted = _normalize_answer(llm(create_intruder_prompt(test_words)))
            scores.setdefault(topic_id, []).append(float(predicted == intruder.lower()))

    return {topic_id: float(np.mean(values)) for topic_id, values in scores.items() if values}


def summarize_metric(values: Mapping[int, float]) -> float:
    """Macro-average a per-topic metric mapping."""
    return float(np.mean(list(values.values()))) if values else 0.0


def evaluate_topic_words(
    topic_words: TopicWords,
    *,
    llm: LLMCallable | None = None,
    embedding_model: str | Callable | None = None,
    word_embeddings: Mapping[str, Sequence[float]] | None = None,
    top_n: int = 20,
    sample_size: int = 5,
    intruder_sample_size: int = 4,
    repetitions: int = 3,
    seed: int = 42,
) -> dict[str, float | dict[int, float]]:
    """Compute available paper-aligned metrics for a topic-word mapping."""
    result: dict[str, float | dict[int, float]] = {
        "D": compute_wmd_diversity(
            topic_words,
            top_n=top_n,
            word_embeddings=word_embeddings,
            embedding_model=embedding_model,
        )
    }
    if llm is not None:
        cr = compute_coherence_rating(
            topic_words,
            llm=llm,
            top_n=top_n,
            sample_size=sample_size,
            repetitions=repetitions,
            seed=seed,
        )
        ci = compute_intruder_detection(
            topic_words,
            llm=llm,
            top_n=top_n,
            sample_size=intruder_sample_size,
            repetitions=repetitions,
            seed=seed,
        )
        result["CR"] = summarize_metric(cr)
        result["CI"] = summarize_metric(ci)
        result["CR_by_topic"] = cr
        result["CI_by_topic"] = ci
    return result


def compute_coherence(
    docs: list[str],
    topic_words: TopicWords,
    metric: str = "cr",
    llm: LLMCallable | None = None,
) -> dict[int, float]:
    """Compute topic coherence.

    ``metric="cr"`` uses the paper-aligned LLM coherence rating and requires
    ``llm``. Classical corpus co-occurrence metrics are intentionally not
    implemented here because they are not the SAE-TM paper's main metric.
    """
    del docs
    if metric != "cr":
        raise NotImplementedError(f"Only paper-aligned metric='cr' is implemented, got {metric!r}")
    if llm is None:
        raise ValueError("metric='cr' requires an llm callable")
    return compute_coherence_rating(topic_words, llm=llm)


def compute_stability(
    model,
    docs: list[str],
    n_runs: int = 5,
) -> float:
    """Stability remains out of scope for the paper-aligned evaluation."""
    del model, docs, n_runs
    raise NotImplementedError("compute_stability is not implemented yet")


def iter_top_words_files(paths: Iterable[str | Path]) -> list[Path]:
    """Expand files/directories into sorted ``top_words.txt`` files."""
    files: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(path.rglob("top_words.txt"))
    return sorted(set(files))
