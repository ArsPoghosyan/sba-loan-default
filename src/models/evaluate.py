"""Evaluation and threshold-selection utilities for binary classifiers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class ThresholdSelection:
    """An F1-maximizing validation threshold and its associated F1 score."""

    threshold: float
    f1: float


def evaluate_model(
    model_name: str,
    fitted_pipeline: object,
    X: pd.DataFrame,
    y_true: pd.Series,
    threshold: float = 0.50,
) -> dict[str, float | str]:
    """Evaluate a fitted binary classifier at one decision threshold."""
    probabilities = fitted_pipeline.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    return {
        "model": model_name,
        "threshold": threshold,
        "pr_auc": average_precision_score(y_true, probabilities),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "brier_score": brier_score_loss(y_true, probabilities),
    }


def select_f1_threshold(
    y_true: pd.Series,
    probabilities: np.ndarray,
) -> ThresholdSelection:
    """Select the validation probability threshold that maximizes F1."""
    precision, recall, thresholds = precision_recall_curve(
        y_true,
        probabilities,
    )

    if len(thresholds) == 0:
        raise ValueError(
            "At least two target classes are required for threshold selection."
        )

    f1_scores = np.divide(
        2 * precision[:-1] * recall[:-1],
        precision[:-1] + recall[:-1],
        out=np.zeros_like(precision[:-1]),
        where=(precision[:-1] + recall[:-1]) != 0,
    )

    best_index = int(np.argmax(f1_scores))

    return ThresholdSelection(
        threshold=float(thresholds[best_index]),
        f1=float(f1_scores[best_index]),
    )

def validate_and_align_features(
    X: pd.DataFrame,
    expected_feature_columns: list[str],
) -> pd.DataFrame:
    """Validate a model input schema and align columns to artifact order."""
    missing_features = (
        set(expected_feature_columns) - set(X.columns)
    )
    unexpected_features = (
        set(X.columns) - set(expected_feature_columns)
    )

    if missing_features:
        raise ValueError(
            "Input data is missing expected features: "
            f"{sorted(missing_features)}"
        )

    if unexpected_features:
        raise ValueError(
            "Input data contains unexpected features: "
            f"{sorted(unexpected_features)}"
        )

    return X.loc[:, expected_feature_columns].copy()