#!/usr/bin/env python3
"""Count tokens in FineWikipedia dataset using Jina v5 tokenizer."""

import argparse
from collections import defaultdict

from datasets import load_dataset
from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn
from transformers import AutoTokenizer


def count_tokens(
    subset: str = "train",
    split: str = "train",
    max_samples: int | None = None,
    text_chunk_size: int = 512,
    text_chunk_overlap: int = 32,
):
    """
    Count tokens in FineWikipedia dataset.

    Args:
        subset: Dataset subset to use ("train" or a specific FineWiki subset)
        split: HuggingFace split name
        max_samples: Maximum number of articles to process (None = all)
        text_chunk_size: Characters per chunk (matches embedding config)
        text_chunk_overlap: Overlap between chunks
    """
    console = Console()

    # Load Jina v5 tokenizer
    console.print("[bold blue]Loading Jina v5 tokenizer...[/]")
    tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v5-text-small", trust_remote_code=True)

    # Load dataset
    console.print(f"[bold blue]Loading FineWikipedia dataset ({subset}/{split})...[/]")
    dataset = load_dataset("HuggingFaceFW/finewiki", split=split)
    console.print(f"[green]Dataset loaded: {len(dataset):,} articles[/]")

    stats = {
        "n_articles": 0,
        "n_chunks": 0,
        "total_tokens": 0,
        "total_chars": 0,
        "chunk_length_distribution": defaultdict(int),
    }

    # Determine total count for progress bar
    if max_samples is not None:
        total = min(max_samples, len(dataset))
    else:
        total = len(dataset)

    console.print("[bold blue]Counting tokens...[/]")

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
    )

    with progress:
        task = progress.add_task(
            "[cyan]Processing articles...", total=total, completed=0
        )

        for idx, example in enumerate(dataset):
            if max_samples is not None and idx >= max_samples:
                break

            text = example["text"]
            if not text:
                progress.update(task, advance=1)
                continue

            stats["n_articles"] += 1
            stats["total_chars"] += len(text)

            # Chunk the text (same logic as create_streaming_dataset)
            chunks = []
            start = 0
            while start < len(text):
                end = start + text_chunk_size
                chunk = text[start:end]
                if chunk.strip():
                    chunks.append(chunk)
                start = end - text_chunk_overlap

            # Count tokens per chunk
            for chunk in chunks:
                tokens = tokenizer.encode(chunk, add_special_tokens=False)
                n_tokens = len(tokens)
                stats["n_chunks"] += 1
                stats["total_tokens"] += n_tokens

                # Bucket distribution (rounded to nearest 32)
                bucket = (n_tokens // 32) * 32
                stats["chunk_length_distribution"][bucket] += 1

            progress.update(task, advance=1)

    # Print results
    console.print("\n[bold]" + "=" * 60 + "[/]")
    console.print("[bold cyan]TOKEN COUNT RESULTS[/]")
    console.print("[bold]" + "=" * 60 + "[/]")

    console.print(f"Articles processed:     [green]{stats['n_articles']:,}[/]")
    console.print(f"Total characters:       [green]{stats['total_chars']:,}[/]")
    console.print(f"Total chunks:           [green]{stats['n_chunks']:,}[/]")
    console.print(f"Total tokens:           [bold green]{stats['total_tokens']:,}[/]")
    console.print(f"Tokens per article:     [yellow]{stats['total_tokens'] / stats['n_articles']:.1f}[/]")
    console.print(f"Tokens per chunk:       [yellow]{stats['total_tokens'] / stats['n_chunks']:.1f}[/]")
    console.print(
        f"Avg chunk length:       [yellow]{stats['total_chars'] / stats['n_chunks']:.1f}[/] chars"
    )

    console.print("\n[bold cyan]Chunk token distribution:[/]")
    for bucket in sorted(stats["chunk_length_distribution"].keys()):
        count = stats["chunk_length_distribution"][bucket]
        pct = 100 * count / stats["n_chunks"]
        bar = "█" * int(pct / 2)
        console.print(
            f"  [cyan]{bucket:4d}[/] tokens: [green]{count:6d}[/] ([yellow]{pct:5.1f}%[/]) {bar}"
        )

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count tokens in FineWikipedia")
    parser.add_argument(
        "--subset", default="train", help="Dataset subset (default: train)"
    )
    parser.add_argument(
        "--split", default="train", help="HuggingFace split (default: train)"
    )
    parser.add_argument(
        "--max-samples", type=int, default=None, help="Max articles to process"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=512, help="Text chunk size in chars"
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=32, help="Chunk overlap in chars"
    )
    args = parser.parse_args()

    count_tokens(
        subset=args.subset,
        split=args.split,
        max_samples=args.max_samples,
        text_chunk_size=args.chunk_size,
        text_chunk_overlap=args.chunk_overlap,
    )
