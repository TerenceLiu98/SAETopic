"""Minimal text topic-modeling example.

Run with a local SAE checkpoint or a Hugging Face checkpoint:

    python examples/quickstart_text.py --sae-model path/to/checkpoint

The example keeps the corpus tiny so the API surface is easy to read. For real
datasets, pass your own documents and tune n_topics, min_df, and stop_words.
"""

from __future__ import annotations

import argparse

from saetopic import SAETopicModel

DOCS = [
    "The rover collected images from the surface of Mars.",
    "The satellite entered orbit after a successful launch.",
    "Astronomers studied signals from a distant galaxy.",
    "The team won after scoring in the final minute.",
    "The coach changed tactics before the championship game.",
    "The striker signed a contract with the football club.",
    "Researchers trained a neural network for image classification.",
    "The model improved after adding more training examples.",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a minimal SAETopic text example")
    parser.add_argument(
        "--sae-model",
        required=True,
        help="Local SAE checkpoint directory or Hugging Face repo id",
    )
    parser.add_argument("--n-topics", type=int, default=3)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    model = SAETopicModel.from_pretrained(
        args.sae_model,
        n_topics=args.n_topics,
        min_df=1,
        idf_weighting=True,
        stop_words="english",
        device=args.device,
    )
    topics, probs = model.fit_transform(DOCS)

    print("topics:", topics)
    print("probabilities shape:", probs.shape)
    print(model.get_topic_info())
    print("topic 0:", model.get_topic(0))
    print(model.get_document_info())

    model.retopic(n_topics=max(2, args.n_topics - 1))
    print("after retopic:")
    print(model.get_topic_info())


if __name__ == "__main__":
    main()
