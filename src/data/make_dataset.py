"""Command-line entry point for creating the cleaned SBA dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.clean import clean_sba_loans


def parse_args() -> argparse.Namespace:
    """Parse command-line paths."""
    parser = argparse.ArgumentParser(
        description="Clean SBA 7(a) FOIA loan data."
    )

    parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help="Path to the raw SBA CSV file.",
    )

    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Destination for the cleaned CSV file.",
    )

    return parser.parse_args()


def main() -> None:
    """Load raw data, clean it, and save the processed dataset."""
    args = parse_args()

    raw_df = pd.read_csv(
        args.input_path,
        low_memory=False,
    )

    clean_df = clean_sba_loans(raw_df)

    args.output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    clean_df.to_csv(
        args.output_path,
        index=False,
    )

    print(f"Saved cleaned dataset to: {args.output_path}")
    print(f"Shape: {clean_df.shape}")


if __name__ == "__main__":
    main()