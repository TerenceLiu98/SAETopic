"""
Command-line interface for SAETopic.
"""

from __future__ import annotations

import argparse


def main() -> None:
    """
    Main CLI entry point.
    """
    parser = argparse.ArgumentParser(
        description="SAETopic: BERTopic-style topic modeling with SAE topic atoms"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # fit command
    fit_parser = subparsers.add_parser("fit", help="Fit topic model on documents")
    fit_parser.add_argument("--input", required=True, help="Input file path")
    fit_parser.add_argument("--text-column", default="text", help="Text column name")
    fit_parser.add_argument("--model", default="saetopic/jina-v5-sae-small", help="SAE model")
    fit_parser.add_argument("--n-topics", type=int, default=50, help="Number of topics")
    fit_parser.add_argument("--output", required=True, help="Output directory")

    # topics command
    topics_parser = subparsers.add_parser("topics", help="Get topic information")
    topics_parser.add_argument("--model", required=True, help="Model directory")
    topics_parser.add_argument("--output", help="Output CSV path")

    # retopic command
    retopic_parser = subparsers.add_parser("retopic", help="Change topic granularity")
    retopic_parser.add_argument("--model", required=True, help="Model directory")
    retopic_parser.add_argument("--n-topics", type=int, required=True, help="New number of topics")
    retopic_parser.add_argument("--output", help="Output directory")

    # visualize command
    viz_parser = subparsers.add_parser("visualize", help="Generate visualizations")
    viz_parser.add_argument("--model", required=True, help="Model directory")
    viz_parser.add_argument("--output", help="Output HTML path")

    args = parser.parse_args()

    # TODO: Implement CLI commands (Week 5)
    if args.command == "fit":
        print(f"Fit command: {args.input} -> {args.output}")
    elif args.command == "topics":
        print(f"Topics command: {args.model}")
    elif args.command == "retopic":
        print(f"Retopic command: {args.model} -> {args.n_topics} topics")
    elif args.command == "visualize":
        print(f"Visualize command: {args.model}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
