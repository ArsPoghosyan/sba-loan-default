"""Tests for temporal SBA train, validation, and test splitting."""

from __future__ import annotations

import pandas as pd
import pytest

from src.models.split_data import split_by_approval_year


def make_model_dataframe() -> pd.DataFrame:
    """Create a small model-ready dataset covering all split periods."""
    return pd.DataFrame(
        {
            "ApprovalFY": [2010, 2017, 2018, 2019],
            "feature_value": [10, 20, 30, 40],
            "default": [0, 1, 0, 1],
        }
    )


def test_split_by_approval_year_creates_expected_partitions() -> None:
    """Rows should be assigned to the intended chronological period."""
    splits = split_by_approval_year(make_model_dataframe())

    assert len(splits.X_train) == 2
    assert len(splits.X_validation) == 1
    assert len(splits.X_test) == 1

    assert splits.y_train.tolist() == [0, 1]
    assert splits.y_validation.tolist() == [0]
    assert splits.y_test.tolist() == [1]

    assert "default" not in splits.X_train.columns
    assert "ApprovalFY" in splits.X_train.columns


def test_split_by_approval_year_rejects_unexpected_years() -> None:
    """Years outside the documented FY2010–2019 range should fail clearly."""
    model_df = make_model_dataframe()
    model_df.loc[0, "ApprovalFY"] = 2020

    with pytest.raises(ValueError, match="Unexpected ApprovalFY values"):
        split_by_approval_year(model_df)


def test_split_by_approval_year_rejects_missing_required_columns() -> None:
    """The splitter should require both target and approval year."""
    model_df = make_model_dataframe().drop(columns=["default"])

    with pytest.raises(ValueError, match="default"):
        split_by_approval_year(model_df)