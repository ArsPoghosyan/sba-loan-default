"""Inference utilities for the saved SBA default model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from src.features.inference import build_inference_features
from src.models.evaluate import validate_and_align_features


REQUIRED_ARTIFACT_KEYS = {
    "model_name",
    "pipeline",
    "threshold",
    "feature_columns",
    "training_period",
    "validation_period",
}


@dataclass(frozen=True)
class PredictionResult:
    """Prediction output for one approval-time loan input."""

    model_name: str
    default_probability: float
    predicted_default: int
    threshold: float


def validate_model_artifact(model_artifact: dict[str, object]) -> None:
    """Raise an error when a model artifact lacks required metadata."""
    missing_keys = REQUIRED_ARTIFACT_KEYS - set(model_artifact)

    if missing_keys:
        raise ValueError(
            "Model artifact is missing required keys: "
            f"{sorted(missing_keys)}"
        )

    threshold = model_artifact["threshold"]

    if not isinstance(threshold, (float, int)) or not 0 <= threshold <= 1:
        raise ValueError(
            "Model artifact threshold must be between 0 and 1."
        )


def load_model_artifact(
    model_path: str | Path,
) -> dict[str, object]:
    """Load and validate a saved joblib model artifact."""
    model_artifact = joblib.load(model_path)
    validate_model_artifact(model_artifact)

    return model_artifact


def predict_from_raw_inputs(
    raw_inputs: dict[str, object],
    model_artifact: dict[str, object],
) -> PredictionResult:
    """Predict charge-off risk from one raw approval-time input record."""
    validate_model_artifact(model_artifact)

    raw_df = pd.DataFrame([raw_inputs])
    feature_df = build_inference_features(raw_df)

    aligned_features = validate_and_align_features(
        feature_df,
        model_artifact["feature_columns"],
    )

    probabilities = model_artifact["pipeline"].predict_proba(
        aligned_features
    )[:, 1]

    default_probability = float(probabilities[0])
    threshold = float(model_artifact["threshold"])

    return PredictionResult(
        model_name=str(model_artifact["model_name"]),
        default_probability=default_probability,
        predicted_default=int(default_probability >= threshold),
        threshold=threshold,
    )