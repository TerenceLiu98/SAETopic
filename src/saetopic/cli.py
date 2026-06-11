"""
Command-line interface for SAETopic.
"""

from __future__ import annotations

import argparse
import sys


def _raise_not_implemented(command: str) -> None:
    """Exit with a clear message for planned inference CLI commands."""
    raise SystemExit(
        f"saetopic {command!s} is not implemented yet. "
        "Current supported workflows are available via `saetopic-train` "
        "or `python -m saetopic.training.cli`."
    )


def main() -> None:
    """
    Main CLI entry point.
    """
    parser = argparse.ArgumentParser(description="SAETopic topic inference commands")
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

    if args.command == "fit":
        _raise_not_implemented("fit")
    elif args.command == "topics":
        _raise_not_implemented("topics")
    elif args.command == "retopic":
        _raise_not_implemented("retopic")
    elif args.command == "visualize":
        _raise_not_implemented("visualize")
    else:
        parser.print_help()
        sys.exit(2)


if __name__ == "__main__":
    main()
