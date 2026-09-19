"""Evaluate the frozen SBA default model once on the FY2019 test set."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.models.evaluate import (
    evaluate_model,
    validate_and_align_features,
)
from src.models.split_data import split_by_approval_year


FEATURE_DTYPES = {
    "naics_sector": "string",
    "CongressionalDistrict": "string",
    "approval_month": "string",
}


def parse_args() -> argparse.Namespace:
    """Parse feature-data, model-artifact, and report paths."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a frozen SBA default model on the FY2019 test set."
        )
    )

    parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help="Path to the feature-engineered SBA CSV file.",
    )

    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to the saved joblib model artifact.",
    )

    parser.add_argument(
        "--metrics-path",
        type=Path,
        required=True,
        help="Destination for the FY2019 metrics CSV file.",
    )

    return parser.parse_args()


def main() -> None:
    """Apply the frozen pipeline and locked threshold to FY2019."""
    args = parse_args()

    model_df = pd.read_csv(
        args.input_path,
        dtype=FEATURE_DTYPES,
        low_memory=False,
    )

    model_artifact = joblib.load(args.model_path)

    required_artifact_keys = {
        "model_name",
        "pipeline",
        "threshold",
        "feature_columns",
        "training_period",
        "validation_period",
    }

    missing_artifact_keys = (
        required_artifact_keys - set(model_artifact)
    )

    if missing_artifact_keys:
        raise ValueError(
            "Model artifact is missing required keys: "
            f"{sorted(missing_artifact_keys)}"
        )

    splits = split_by_approval_year(model_df)

    X_test = validate_and_align_features(
        splits.X_test,
        model_artifact["feature_columns"],
    )

    test_result = evaluate_model(
        model_name=model_artifact["model_name"],
        fitted_pipeline=model_artifact["pipeline"],
        X=X_test,
        y_true=splits.y_test,
        threshold=model_artifact["threshold"],
    )

    metrics_df = pd.DataFrame(
        [
            {
                "dataset": "FY2019 test set",
                "pr_auc": test_result["pr_auc"],
                "roc_auc": test_result["roc_auc"],
                "precision": test_result["precision"],
                "recall": test_result["recall"],
                "f1": test_result["f1"],
                "brier_score": test_result["brier_score"],
            }
        ]
    )

    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(args.metrics_path, index=False)

    print(f"Model: {model_artifact['model_name']}")
    print(f"Locked threshold: {model_artifact['threshold']:.4f}")
    print(f"Saved FY2019 metrics to: {args.metrics_path}")
    print(metrics_df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()