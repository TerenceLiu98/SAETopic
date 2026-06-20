"""Command-line interface for fitted SAETopic models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from saetopic import SAETopicModel


def _none_if_string(value: str | None) -> str | None:
    if value is None:
        return None
    if value.lower() in {"none", "null", ""}:
        return None
    return value


def _read_documents(path: str, text_column: str) -> list[str]:
    input_path = Path(path)
    suffix = input_path.suffix.lower()

    if suffix in {".txt", ".text"}:
        return [
            line.strip()
            for line in input_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    if suffix == ".jsonl":
        docs = []
        with input_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                docs.append(str(row[text_column]))
        return docs

    if suffix == ".json":
        data = json.loads(input_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            if not data:
                return []
            if isinstance(data[0], str):
                return [str(item) for item in data]
            return [str(row[text_column]) for row in data]
        raise ValueError("JSON input must be a list of strings or objects")

    if suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        frame = pd.read_csv(input_path, sep=sep)
        return frame[text_column].astype(str).tolist()

    raise ValueError(
        f"Unsupported input extension {suffix!r}. Use .txt, .jsonl, .json, .csv, or .tsv."
    )


def _load_embeddings(path: str | None) -> np.ndarray | None:
    if path is None:
        return None
    embeddings = np.load(path)
    if embeddings.ndim != 2:
        raise ValueError(f"Embeddings must be a 2D array, got shape={embeddings.shape}")
    return embeddings.astype(np.float32, copy=False)


def _write_or_print(frame: pd.DataFrame, output: str | None) -> None:
    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out_path, index=False)
    else:
        print(frame.to_string(index=False))


def _add_model_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--embedding-model", default="jinaai/jina-embeddings-v5-text-small")
    parser.add_argument("--merge-embedding-model", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--max-df", type=float, default=1.0)
    parser.add_argument("--stop-words", default="english")
    parser.add_argument("--idf-weighting", action="store_true")
    parser.add_argument("--use-ctfidf", action="store_true")
    parser.add_argument(
        "--theta-mode",
        default="dense",
        choices=["dense", "sparse_topk"],
    )


def _cmd_fit(args: argparse.Namespace) -> None:
    docs = _read_documents(args.input, args.text_column)
    embeddings = _load_embeddings(args.embeddings)

    model = SAETopicModel.from_pretrained(
        args.model,
        embedding_model=args.embedding_model,
        merge_embedding_model=_none_if_string(args.merge_embedding_model),
        n_topics=args.n_topics,
        min_df=args.min_df,
        max_df=args.max_df,
        stop_words=_none_if_string(args.stop_words),
        idf_weighting=args.idf_weighting,
        use_ctfidf=args.use_ctfidf,
        theta_mode=args.theta_mode,
        device=args.device,
    )
    model.fit_transform(docs, embeddings=embeddings)
    model.save(args.output)

    output_dir = Path(args.output)
    model.get_topic_info().to_csv(output_dir / "topic_info.csv", index=False)
    model.get_document_info().to_csv(output_dir / "document_info.csv", index=False)
    print(f"Saved SAETopic model to {output_dir}")


def _cmd_topics(args: argparse.Namespace) -> None:
    model = SAETopicModel.load(args.model)
    info = model.get_topic_info()
    if args.top_n != 10:
        topics = model.get_topics(top_n=args.top_n)
        info = info.copy()
        info["Top_Words"] = [
            ", ".join(word for word, _ in topics[int(topic_id)])
            for topic_id in info["Topic"]
        ]
    _write_or_print(info, args.output)


def _cmd_retopic(args: argparse.Namespace) -> None:
    model = SAETopicModel.load(args.model)
    model.retopic(args.n_topics)

    output = args.output or args.model
    model.save(output)
    model.get_topic_info().to_csv(Path(output) / "topic_info.csv", index=False)
    print(f"Saved retopiced model to {output}")


def _raise_not_implemented(command: str) -> None:
    raise SystemExit(f"saetopic {command!s} is not implemented yet.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SAETopic topic inference commands")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    fit_parser = subparsers.add_parser("fit", help="Fit topic model on documents")
    fit_parser.add_argument("--input", required=True, help="Input .txt/.jsonl/.json/.csv/.tsv")
    fit_parser.add_argument("--text-column", default="text", help="Text column/key name")
    fit_parser.add_argument("--model", required=True, help="SAE checkpoint path or repo id")
    fit_parser.add_argument("--embeddings", default=None, help="Optional precomputed .npy embeddings")
    fit_parser.add_argument("--n-topics", type=int, default=50, help="Number of topics")
    fit_parser.add_argument("--output", required=True, help="Output model directory")
    _add_model_options(fit_parser)
    fit_parser.set_defaults(func=_cmd_fit)

    topics_parser = subparsers.add_parser("topics", help="Export topic information")
    topics_parser.add_argument("--model", required=True, help="Saved SAETopic model directory")
    topics_parser.add_argument("--output", help="Optional output CSV path")
    topics_parser.add_argument("--top-n", type=int, default=10, help="Top words per topic")
    topics_parser.set_defaults(func=_cmd_topics)

    retopic_parser = subparsers.add_parser("retopic", help="Change topic granularity")
    retopic_parser.add_argument("--model", required=True, help="Saved SAETopic model directory")
    retopic_parser.add_argument("--n-topics", type=int, required=True, help="New topic count")
    retopic_parser.add_argument(
        "--output",
        help="Output model directory. Defaults to overwriting --model.",
    )
    retopic_parser.set_defaults(func=_cmd_retopic)

    viz_parser = subparsers.add_parser("visualize", help="Generate visualizations")
    viz_parser.add_argument("--model", required=True, help="Model directory")
    viz_parser.add_argument("--output", help="Output HTML path")
    viz_parser.set_defaults(func=lambda args: _raise_not_implemented("visualize"))

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(2)
    args.func(args)


if __name__ == "__main__":
    main()
