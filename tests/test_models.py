"""Tests for SBA model factories and evaluation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.evaluate import (
    evaluate_model,
    select_f1_threshold,
    validate_and_align_features,
)
from src.models.train import (
    build_initial_xgboost_pipeline,
    build_logistic_regression_pipeline,
    build_random_forest_pipeline,
    build_tuned_xgboost_pipeline,
    calculate_class_ratio,
)


class ProbabilityModel:
    """Small predictable model used to test evaluation logic."""

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probabilities = X["probability"].to_numpy()

        return np.column_stack(
            [1 - probabilities, probabilities]
        )


def test_calculate_class_ratio() -> None:
    """Class ratio should equal negatives divided by positives."""
    y_train = pd.Series([0, 0, 0, 1])

    assert calculate_class_ratio(y_train) == 3.0


def test_calculate_class_ratio_rejects_no_positive_class() -> None:
    """Class ratio cannot be calculated without defaults."""
    y_train = pd.Series([0, 0, 0])

    with pytest.raises(ValueError, match="no positive-class"):
        calculate_class_ratio(y_train)


def test_model_factories_create_pipeline_steps() -> None:
    """Every candidate should use the shared preprocessing pipeline."""
    pipelines = [
        build_logistic_regression_pipeline(),
        build_random_forest_pipeline(),
        build_initial_xgboost_pipeline(pd.Series([0, 0, 1])),
        build_tuned_xgboost_pipeline(),
    ]

    for pipeline in pipelines:
        assert list(pipeline.named_steps) == [
            "preprocessor",
            "classifier",
        ]


def test_tuned_xgboost_uses_selected_hyperparameters() -> None:
    """The final XGBoost factory must preserve tuning-notebook results."""
    pipeline = build_tuned_xgboost_pipeline()
    classifier = pipeline.named_steps["classifier"]

    assert classifier.n_estimators == 300
    assert classifier.learning_rate == 0.10
    assert classifier.max_depth == 7
    assert classifier.min_child_weight == 5
    assert classifier.subsample == 0.85
    assert classifier.colsample_bytree == 0.85
    assert classifier.reg_lambda == 10.0
    assert classifier.scale_pos_weight == 5.0


def test_evaluate_model_returns_expected_metrics() -> None:
    """Evaluation should calculate all reported validation/test metrics."""
    X = pd.DataFrame({"probability": [0.1, 0.8, 0.7, 0.2]})
    y_true = pd.Series([0, 1, 1, 0])

    result = evaluate_model(
        model_name="Test model",
        fitted_pipeline=ProbabilityModel(),
        X=X,
        y_true=y_true,
        threshold=0.50,
    )

    assert result["model"] == "Test model"
    assert result["threshold"] == 0.50
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0
    assert 0 <= result["pr_auc"] <= 1
    assert 0 <= result["roc_auc"] <= 1
    assert 0 <= result["brier_score"] <= 1


def test_select_f1_threshold_returns_best_validation_threshold() -> None:
    """Threshold selection should maximize F1 on validation predictions."""
    y_true = pd.Series([0, 1, 1, 0])
    probabilities = np.array([0.1, 0.8, 0.7, 0.2])

    selection = select_f1_threshold(y_true, probabilities)

    assert selection.threshold == 0.7
    assert selection.f1 == 1.0


def test_validate_and_align_features_reorders_columns() -> None:
    """Artifact feature order should be restored before prediction."""
    X = pd.DataFrame(
        {
            "second_feature": [1, 2],
            "first_feature": [3, 4],
        }
    )

    aligned_X = validate_and_align_features(
        X,
        ["first_feature", "second_feature"],
    )

    assert aligned_X.columns.tolist() == [
        "first_feature",
        "second_feature",
    ]


def test_validate_and_align_features_rejects_schema_mismatch() -> None:
    """Missing or additional prediction features should fail clearly."""
    X = pd.DataFrame(
        {
            "first_feature": [1, 2],
            "unexpected_feature": [3, 4],
        }
    )

    with pytest.raises(ValueError, match="missing expected features"):
        validate_and_align_features(
            X,
            ["first_feature", "second_feature"],
        )
