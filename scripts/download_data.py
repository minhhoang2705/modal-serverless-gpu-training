#!/usr/bin/env python3
"""Download training datasets."""

import argparse
from pathlib import Path
import urllib.request


def download_shakespeare(output_dir):
    """Download TinyShakespeare dataset."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    output_path = output_dir / "shakespeare.txt"

    print(f"Downloading TinyShakespeare from {url}...")
    urllib.request.urlretrieve(url, output_path)
    print(f"Saved to {output_path}")

    return output_path


def download_tinystories(output_dir):
    """
    Download TinyStories dataset.

    Note: This uses HuggingFace datasets, so it will be downloaded on-the-fly
    during training. This function just prints instructions.
    """
    print("\nTinyStories dataset:")
    print("This dataset will be automatically downloaded during training.")
    print("No manual download needed!")
    print("\nDataset info: https://huggingface.co/datasets/roneneldan/TinyStories")


def main():
    parser = argparse.ArgumentParser(description="Download training datasets")
    parser.add_argument(
        "--dataset",
        choices=["shakespeare", "tinystories", "all"],
        default="all",
        help="Which dataset to download"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data",
        help="Output directory"
    )

    args = parser.parse_args()

    if args.dataset in ["shakespeare", "all"]:
        download_shakespeare(args.output)

    if args.dataset in ["tinystories", "all"]:
        download_tinystories(args.output)

    print("\nDone!")


if __name__ == "__main__":
    main()
