"""Time-based train, validation, and test splitting utilities."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


TARGET_COLUMN = "default"
APPROVAL_YEAR_COLUMN = "ApprovalFY"

TRAIN_YEARS = range(2010, 2018)
VALIDATION_YEAR = 2018
TEST_YEAR = 2019
EXPECTED_YEARS = set(range(2010, 2020))


@dataclass(frozen=True)
class TemporalSplit:
    """Feature and target partitions for temporal model evaluation."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_validation: pd.DataFrame
    y_validation: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series


def split_by_approval_year(
    model_df: pd.DataFrame,
) -> TemporalSplit:
    """Split model-ready data into FY2010–17, FY2018, and FY2019 sets."""
    required_columns = {TARGET_COLUMN, APPROVAL_YEAR_COLUMN}
    missing_columns = required_columns - set(model_df.columns)

    if missing_columns:
        raise ValueError(
            "Model dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    observed_years = set(model_df[APPROVAL_YEAR_COLUMN].dropna().unique())
    unexpected_years = observed_years - EXPECTED_YEARS

    if unexpected_years:
        raise ValueError(
            "Unexpected ApprovalFY values: "
            f"{sorted(unexpected_years)}"
        )

    if model_df[APPROVAL_YEAR_COLUMN].isna().any():
        raise ValueError("ApprovalFY contains missing values.")

    train_mask = model_df[APPROVAL_YEAR_COLUMN].isin(TRAIN_YEARS)
    validation_mask = (
        model_df[APPROVAL_YEAR_COLUMN] == VALIDATION_YEAR
    )
    test_mask = model_df[APPROVAL_YEAR_COLUMN] == TEST_YEAR

    if not (train_mask | validation_mask | test_mask).all():
        raise ValueError("Some rows were not assigned to a temporal split.")

    feature_columns = model_df.columns.drop(TARGET_COLUMN)

    return TemporalSplit(
        X_train=model_df.loc[train_mask, feature_columns].copy(),
        y_train=model_df.loc[train_mask, TARGET_COLUMN].copy(),
        X_validation=model_df.loc[
            validation_mask,
            feature_columns,
        ].copy(),
        y_validation=model_df.loc[
            validation_mask,
            TARGET_COLUMN,
        ].copy(),
        X_test=model_df.loc[test_mask, feature_columns].copy(),
        y_test=model_df.loc[test_mask, TARGET_COLUMN].copy(),
    )