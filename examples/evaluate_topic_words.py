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
        --word-embeddings-dir data/gensim/w2v \
        --out results/text_topic_eval.jsonl

To also compute CI/CR with vLLM:
    PYTHONPATH=src python examples/evaluate_topic_words.py \
        results/text_topics \
        --word-embeddings-dir data/gensim/w2v \
        --llm-backend vllm \
        --llm-model microsoft/phi-4 \
        --out results/text_topic_eval.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from saetopic.evaluation import (
    compute_coherence_rating,
    compute_intruder_detection,
    compute_wmd_diversity,
    iter_top_words_files,
    load_saetm_word2vec_cache,
    load_top_words_file,
    summarize_metric,
)

console = Console()


def _build_vllm_callable(
    model_name: str,
    max_model_len: int,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_num_seqs: int | None,
    max_num_batched_tokens: int | None,
    enforce_eager: bool,
):
    try:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise ImportError(
            "vLLM evaluation requires `vllm` and `transformers`. "
            "Install them in the evaluation environment."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm_kwargs = {
        "model": model_name,
        "trust_remote_code": True,
        "dtype": "auto",
        "tensor_parallel_size": tensor_parallel_size,
        "max_model_len": max_model_len,
        "gpu_memory_utilization": gpu_memory_utilization,
        "enforce_eager": enforce_eager,
    }
    if max_num_seqs is not None:
        llm_kwargs["max_num_seqs"] = max_num_seqs
    if max_num_batched_tokens is not None:
        llm_kwargs["max_num_batched_tokens"] = max_num_batched_tokens
    llm = LLM(**llm_kwargs)

    class VLLMCallable:
        def __call__(self, prompt: str) -> str:
            return self.batch_intruder([prompt])[0]

        def _generate(
            self,
            prompts: list[str],
            *,
            max_tokens: int,
            temperature: float,
            top_p: float = 1.0,
        ) -> list[str]:
            formatted = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for prompt in prompts
            ]
            params = SamplingParams(max_tokens=max_tokens, temperature=temperature, top_p=top_p)
            try:
                outputs = llm.generate(formatted, params, use_tqdm=False)
            except TypeError:
                outputs = llm.generate(formatted, params)
            return [output.outputs[0].text for output in outputs]

        def batch_coherence(self, prompts: list[str]) -> list[str]:
            return self._generate(prompts, max_tokens=512, temperature=0.7, top_p=0.9)

        def batch_intruder(self, prompts: list[str]) -> list[str]:
            return self._generate(prompts, max_tokens=10, temperature=0.0)

    return VLLMCallable()


def _n_coherence_prompts(topic_words: dict[int, list[str]], k: int, repetitions: int) -> int:
    return sum(1 for words in topic_words.values() if len(words) >= k) * repetitions


def _n_intruder_prompts(topic_words: dict[int, list[str]], k: int, n: int, repetitions: int) -> int:
    valid_topics = [topic_id for topic_id, words in topic_words.items() if len(words) >= k]
    if len(topic_words) < 2:
        return 0
    return sum(
        1
        for topic_id in valid_topics
        if k >= n and any(other != topic_id and topic_words[other] for other in topic_words)
    ) * repetitions


def _progress_batch_callable(llm_batch, progress: Progress, task_id: TaskID):
    if llm_batch is None:
        return None

    def call(prompts: list[str]) -> list[str]:
        responses = list(llm_batch(prompts))
        progress.update(task_id, advance=len(prompts))
        return responses

    return call


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="+",
        help="top_words.txt files or directories containing top_words.txt files.",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help=(
            "Fallback embedding model used for WMD diversity if --word-embeddings-dir "
            "is unavailable. The SAE-TM reference uses a word2vec cache instead."
        ),
    )
    parser.add_argument(
        "--word-embeddings-dir",
        default=str(Path(__file__).resolve().parents[1] / "data" / "gensim" / "w2v"),
        help=(
            "Directory containing SAE-TM word2vec cache files "
            "embeddings.np.npy and vocabulary.json."
        ),
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
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="vLLM GPU memory utilization fraction.",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=None,
        help="Optional vLLM cap on concurrent sequences.",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=None,
        help="Optional vLLM cap on batched prefill/decode tokens.",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable CUDA graph capture in vLLM to reduce extra memory pressure.",
    )
    parser.add_argument(
        "--llm-batch-size",
        type=int,
        default=32,
        help="Number of CI/CR prompts sent to the LLM in each batch.",
    )
    parser.add_argument("--top-n", type=int, default=20, help="Top words used for WMD D.")
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Initial keywords per topic line for CR and CI, matching SAE-TM.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=4,
        help="Keywords subsampled from the initial k words for CI.",
    )
    parser.add_argument("--r", type=int, default=3, help="Repetitions per topic for CI/CR.")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed. Omit to match SAE-TM's unseeded evaluation script.",
    )
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
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens,
            enforce_eager=args.enforce_eager,
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    word_embeddings = None
    mean_embedding = None
    word_embeddings_dir = Path(args.word_embeddings_dir)
    if word_embeddings_dir.exists():
        word_embeddings, mean_embedding = load_saetm_word2vec_cache(word_embeddings_dir)
    elif args.embedding_model is None:
        raise FileNotFoundError(
            "SAE-TM word2vec cache not found at "
            f"{word_embeddings_dir}. Provide --word-embeddings-dir or "
            "--embedding-model for a non-reference fallback."
        )

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("elapsed •"),
        TimeRemainingColumn(),
        TextColumn("remaining"),
    )

    with out_path.open("w", encoding="utf-8") as f, progress:
        file_task = progress.add_task(
            "[cyan]Evaluating topic files",
            total=len(top_words_files),
        )
        for path in top_words_files:
            progress.update(file_task, description=f"[cyan]Evaluating {path.parent}")
            topic_words = load_top_words_file(path, top_n=args.top_n)
            short_topics = [
                topic_id
                for topic_id, words in topic_words.items()
                if len(words) < args.top_n
            ]
            if short_topics:
                console.print(
                    f"  warning: {len(short_topics)} topics have fewer than "
                    f"{args.top_n} words; metrics will use the available words."
                )
            result = {
                "path": str(path),
                "metrics": {
                    "D": compute_wmd_diversity(
                        topic_words,
                        top_n=args.top_n,
                        word_embeddings=word_embeddings,
                        mean_embedding=mean_embedding,
                        embedding_model=args.embedding_model,
                    )
                },
            }
            if llm is not None:
                cr_total = _n_coherence_prompts(topic_words, args.k, args.r)
                cr_task = progress.add_task(
                    "[magenta]CR judge prompts",
                    total=cr_total,
                )
                cr = compute_coherence_rating(
                    topic_words,
                    llm=llm,
                    llm_batch=_progress_batch_callable(
                        getattr(llm, "batch_coherence", None),
                        progress,
                        cr_task,
                    ),
                    llm_batch_size=args.llm_batch_size,
                    top_n=args.k,
                    sample_size=args.k,
                    repetitions=args.r,
                    seed=args.seed,
                )
                progress.update(cr_task, completed=cr_total)
                progress.remove_task(cr_task)

                ci_total = _n_intruder_prompts(topic_words, args.k, args.n, args.r)
                ci_task = progress.add_task(
                    "[magenta]CI judge prompts",
                    total=ci_total,
                )
                ci = compute_intruder_detection(
                    topic_words,
                    llm=llm,
                    llm_batch=_progress_batch_callable(
                        getattr(llm, "batch_intruder", None),
                        progress,
                        ci_task,
                    ),
                    llm_batch_size=args.llm_batch_size,
                    top_n=args.k,
                    sample_size=args.n,
                    repetitions=args.r,
                    seed=args.seed,
                )
                progress.update(ci_task, completed=ci_total)
                progress.remove_task(ci_task)
                ci = {topic_id: score * 100.0 for topic_id, score in ci.items()}
                result["metrics"]["CR"] = summarize_metric(cr)
                result["metrics"]["CI"] = summarize_metric(ci)
                result["CR_by_topic"] = cr
                result["CI_by_topic"] = ci

            f.write(json.dumps(result, sort_keys=True) + "\n")
            progress.update(file_task, advance=1)

    console.print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
