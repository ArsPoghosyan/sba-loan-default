"""Train the frozen final SBA default model and save its artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.models.evaluate import evaluate_model, select_f1_threshold
from src.models.split_data import split_by_approval_year
from src.models.train import build_tuned_xgboost_pipeline


MODEL_NAME = "Tuned XGBoost"

FEATURE_DTYPES = {
    "naics_sector": "string",
    "CongressionalDistrict": "string",
    "approval_month": "string",
}


def parse_args() -> argparse.Namespace:
    """Parse feature-data and artifact paths."""
    parser = argparse.ArgumentParser(
        description=(
            "Train the selected SBA default model using FY2010-2017 "
            "and select its threshold on FY2018."
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
        help="Destination for the saved joblib model artifact.",
    )

    return parser.parse_args()


def main() -> None:
    """Fit the final pipeline, select its threshold, and save an artifact."""
    args = parse_args()

    model_df = pd.read_csv(
        args.input_path,
        dtype=FEATURE_DTYPES,
        low_memory=False,
    )

    splits = split_by_approval_year(model_df)

    pipeline = build_tuned_xgboost_pipeline()
    pipeline.fit(splits.X_train, splits.y_train)

    validation_probabilities = pipeline.predict_proba(
        splits.X_validation
    )[:, 1]

    threshold_selection = select_f1_threshold(
        splits.y_validation,
        validation_probabilities,
    )

    validation_result = evaluate_model(
        model_name=MODEL_NAME,
        fitted_pipeline=pipeline,
        X=splits.X_validation,
        y_true=splits.y_validation,
        threshold=threshold_selection.threshold,
    )

    model_artifact = {
        "model_name": MODEL_NAME,
        "pipeline": pipeline,
        "threshold": threshold_selection.threshold,
        "validation_pr_auc": validation_result["pr_auc"],
        "validation_f1": validation_result["f1"],
        "feature_columns": splits.X_train.columns.tolist(),
        "training_period": "FY2010-FY2017",
        "validation_period": "FY2018",
    }

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_artifact, args.model_path)

    print(f"Saved model artifact to: {args.model_path}")
    print(
        f"FY2018 validation PR-AUC: "
        f"{validation_result['pr_auc']:.4f}"
    )
    print(
        f"Selected FY2018 F1 threshold: "
        f"{threshold_selection.threshold:.4f}"
    )
    print(f"FY2018 validation F1: {validation_result['f1']:.4f}")


if __name__ == "__main__":
    main()
