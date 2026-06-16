import json

import pytest

from saetopic.evaluation import (
    compute_coherence_rating,
    compute_diversity,
    compute_intruder_detection,
    compute_unique_word_diversity,
    compute_wmd_diversity,
    load_top_words_file,
    parse_coherence_score,
    summarize_metric,
    write_top_words_file,
)


def test_write_and_load_top_words_file(tmp_path):
    topic_words = {
        0: [("space", 0.9), ("orbit", 0.8)],
        1: [("car", 0.7), ("engine", 0.6)],
    }
    path = tmp_path / "top_words.txt"

    write_top_words_file(topic_words, path, top_n=2)

    assert path.read_text() == "space, orbit\ncar, engine\n"
    assert load_top_words_file(path) == {0: ["space", "orbit"], 1: ["car", "engine"]}


def test_unique_word_diversity():
    topic_words = {0: ["a", "b", "c"], 1: ["a", "d", "e"]}

    assert compute_unique_word_diversity(topic_words, top_n=3) == pytest.approx(5 / 6)
    assert compute_diversity(topic_words, top_n=3, method="unique") == pytest.approx(5 / 6)


def test_wmd_diversity_with_supplied_embeddings():
    topic_words = {
        0: ["apple", "orange"],
        1: ["car", "truck"],
    }
    embeddings = {
        "apple": [0.0, 0.0],
        "orange": [0.0, 1.0],
        "car": [10.0, 10.0],
        "truck": [10.0, 11.0],
    }

    score = compute_wmd_diversity(topic_words, top_n=2, word_embeddings=embeddings)

    assert score > 10


def test_parse_coherence_score_json_and_fallback():
    assert parse_coherence_score(json.dumps({"rationale": "ok", "score": 77})) == 77
    assert parse_coherence_score("score: 42") == 42
    assert parse_coherence_score("not a score") is None


def test_coherence_rating_with_callable_llm():
    topic_words = {0: ["space", "orbit", "nasa", "moon", "planet"]}

    def llm(prompt: str) -> str:
        assert "space" in prompt or "orbit" in prompt or "nasa" in prompt
        return '{"rationale": "coherent", "score": 90}'

    scores = compute_coherence_rating(topic_words, llm=llm, repetitions=2, seed=0)

    assert scores == {0: 90.0}
    assert summarize_metric(scores) == 90.0


def test_coherence_rating_with_batched_llm():
    topic_words = {0: ["space", "orbit", "nasa", "moon", "planet"]}
    batches = []

    def llm(prompt: str) -> str:
        raise AssertionError(f"single-call LLM should not be used: {prompt}")

    def llm_batch(prompts):
        batches.append(list(prompts))
        return ['{"rationale": "coherent", "score": 80}' for _ in prompts]

    scores = compute_coherence_rating(
        topic_words,
        llm=llm,
        llm_batch=llm_batch,
        llm_batch_size=2,
        repetitions=3,
        seed=0,
    )

    assert scores == {0: 80.0}
    assert [len(batch) for batch in batches] == [2, 1]


def test_intruder_detection_with_callable_llm():
    topic_words = {
        0: ["space", "orbit", "nasa", "moon", "planet"],
        1: ["car", "engine", "road", "wheel", "drive"],
    }

    def llm(prompt: str) -> str:
        if "car" in prompt:
            return "car"
        return "space"

    scores = compute_intruder_detection(topic_words, llm=llm, repetitions=1, seed=2)

    assert set(scores) == {0, 1}
    assert all(0.0 <= score <= 1.0 for score in scores.values())


def test_intruder_detection_with_batched_llm():
    topic_words = {
        0: ["space", "orbit", "nasa", "moon", "planet"],
        1: ["car", "engine", "road", "wheel", "drive"],
    }
    batches = []

    def llm(prompt: str) -> str:
        raise AssertionError(f"single-call LLM should not be used: {prompt}")

    def llm_batch(prompts):
        batches.append(list(prompts))
        responses = []
        for prompt in prompts:
            responses.append("car" if "car" in prompt else "space")
        return responses

    scores = compute_intruder_detection(
        topic_words,
        llm=llm,
        llm_batch=llm_batch,
        llm_batch_size=1,
        repetitions=1,
        seed=2,
    )

    assert set(scores) == {0, 1}
    assert [len(batch) for batch in batches] == [1, 1]
