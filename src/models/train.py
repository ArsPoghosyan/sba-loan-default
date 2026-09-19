"""Model pipeline factories for SBA loan-default prediction."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.features.preprocessing import build_preprocessor


RANDOM_STATE = 42


def build_pipeline(classifier: object) -> Pipeline:
    """Combine a fresh shared preprocessor with a classifier."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier),
        ]
    )


def calculate_class_ratio(y_train: pd.Series) -> float:
    """Return the training negative-to-positive class ratio."""
    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()

    if positive_count == 0:
        raise ValueError("y_train contains no positive-class observations.")

    return negative_count / positive_count


def build_logistic_regression_pipeline() -> Pipeline:
    """Build the unweighted logistic-regression benchmark."""
    classifier = LogisticRegression(
        max_iter=5_000,
        tol=1e-4,
        solver="saga",
        random_state=RANDOM_STATE,
    )

    return build_pipeline(classifier)


def build_random_forest_pipeline() -> Pipeline:
    """Build the random-forest candidate from the tuning notebook."""
    classifier = RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    return build_pipeline(classifier)


def build_initial_xgboost_pipeline(
    y_train: pd.Series,
) -> Pipeline:
    """Build the initial class-weighted XGBoost candidate."""
    classifier = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=calculate_class_ratio(y_train),
        reg_lambda=1.0,
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    return build_pipeline(classifier)


def build_tuned_xgboost_pipeline() -> Pipeline:
    """Build the final XGBoost configuration selected by temporal CV."""
    classifier = XGBClassifier(
        n_estimators=300,
        learning_rate=0.10,
        max_depth=7,
        min_child_weight=5,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=10.0,
        scale_pos_weight=5.0,
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    return build_pipeline(classifier)