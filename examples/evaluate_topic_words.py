"""
Evaluate topic-word outputs with SAE-TM paper-style metrics.

Metrics:
    D  = average pairwise WMD diversity
    CR = LLM coherence rating
    CI = LLM intruder detection accuracy

Input can be one or more top_words.txt files or directories containing them.

Example:
    PYTHONPATH=src python examples/evaluate_topic_words.py \
        results/text_topics \
        --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
        --out results/text_topic_eval.jsonl

To also compute CI/CR with vLLM:
    PYTHONPATH=src python examples/evaluate_topic_words.py \
        results/text_topics \
        --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
        --llm-backend vllm \
        --llm-model microsoft/phi-4 \
        --out results/text_topic_eval.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from saetopic.evaluation import (
    compute_coherence_rating,
    compute_intruder_detection,
    compute_wmd_diversity,
    iter_top_words_files,
    load_top_words_file,
    summarize_metric,
)


def _build_vllm_callable(model_name: str, max_model_len: int, tensor_parallel_size: int):
    try:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise ImportError(
            "vLLM evaluation requires `vllm` and `transformers`. "
            "Install them in the evaluation environment."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        dtype="auto",
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=0.9,
    )

    class VLLMCallable:
        def __call__(self, prompt: str) -> str:
            return self.batch([prompt])[0]

        def batch(self, prompts: list[str]) -> list[str]:
            formatted = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for prompt in prompts
            ]
            params = SamplingParams(max_tokens=512, temperature=0.0)
            outputs = llm.generate(formatted, params)
            return [output.outputs[0].text for output in outputs]

    return VLLMCallable()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="+",
        help="top_words.txt files or directories containing top_words.txt files.",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model used to embed topic words for WMD diversity.",
    )
    parser.add_argument(
        "--llm-backend",
        choices=["none", "vllm"],
        default="none",
        help="Set to vllm to compute CI and CR. D is always computed.",
    )
    parser.add_argument("--llm-model", default="microsoft/phi-4")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument(
        "--llm-batch-size",
        type=int,
        default=32,
        help="Number of CI/CR prompts sent to the LLM in each batch.",
    )
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--k", type=int, default=5, help="Words sampled for CR.")
    parser.add_argument("--n", type=int, default=4, help="Main words sampled for CI.")
    parser.add_argument("--r", type=int, default=3, help="Repetitions per topic for CI/CR.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="topic_eval.jsonl", help="JSONL output path.")
    args = parser.parse_args()

    top_words_files = iter_top_words_files(args.paths)
    if not top_words_files:
        raise FileNotFoundError(f"No top_words.txt files found from: {args.paths}")

    llm = None
    if args.llm_backend == "vllm":
        llm = _build_vllm_callable(
            model_name=args.llm_model,
            max_model_len=args.max_model_len,
            tensor_parallel_size=args.tensor_parallel_size,
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for path in top_words_files:
            print(f"Evaluating {path}")
            topic_words = load_top_words_file(path, top_n=args.top_n)
            short_topics = [
                topic_id
                for topic_id, words in topic_words.items()
                if len(words) < args.top_n
            ]
            if short_topics:
                print(
                    f"  warning: {len(short_topics)} topics have fewer than "
                    f"{args.top_n} words; metrics will use the available words."
                )
            result = {
                "path": str(path),
                "metrics": {
                    "D": compute_wmd_diversity(
                        topic_words,
                        top_n=args.top_n,
                        embedding_model=args.embedding_model,
                    )
                },
            }
            if llm is not None:
                cr = compute_coherence_rating(
                    topic_words,
                    llm=llm,
                    llm_batch=getattr(llm, "batch", None),
                    llm_batch_size=args.llm_batch_size,
                    top_n=args.top_n,
                    sample_size=args.k,
                    repetitions=args.r,
                    seed=args.seed,
                )
                ci = compute_intruder_detection(
                    topic_words,
                    llm=llm,
                    llm_batch=getattr(llm, "batch", None),
                    llm_batch_size=args.llm_batch_size,
                    top_n=args.top_n,
                    sample_size=args.n,
                    repetitions=args.r,
                    seed=args.seed,
                )
                result["metrics"]["CR"] = summarize_metric(cr)
                result["metrics"]["CI"] = summarize_metric(ci)
                result["CR_by_topic"] = cr
                result["CI_by_topic"] = ci

            f.write(json.dumps(result, sort_keys=True) + "\n")

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
