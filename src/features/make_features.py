"""Command-line entry point for creating model-ready SBA features."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.features.build_features import build_features


def parse_args() -> argparse.Namespace:
    """Parse command-line paths."""
    parser = argparse.ArgumentParser(
        description="Create model-ready SBA 7(a) features."
    )

    parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help="Path to the cleaned SBA CSV file.",
    )

    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Destination for the feature-engineered CSV file.",
    )

    return parser.parse_args()


def main() -> None:
    """Load cleaned data, engineer features, and save the result."""
    args = parse_args()

    cleaned_df = pd.read_csv(
        args.input_path,
        dtype={"naics_sector": "string"},
        low_memory=False,
    )

    feature_df = build_features(cleaned_df)

    args.output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_df.to_csv(args.output_path, index=False)

    print(f"Saved feature dataset to: {args.output_path}")
    print(f"Shape: {feature_df.shape}")


if __name__ == "__main__":
    main()