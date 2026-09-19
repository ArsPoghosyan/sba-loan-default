"""Tests for SBA feature engineering functions."""

from __future__ import annotations

import pandas as pd
import pytest

from src.features.build_features import (
    DROP_AFTER_ENGINEERING,
    build_features,
    validate_required_columns,
)


def make_cleaned_dataframe() -> pd.DataFrame:
    """Create a small fixture matching the cleaned SBA schema."""
    return pd.DataFrame(
        {
            "BorrState": ["TX", "CA", "NY"],
            "BankState": ["TX", "NC", "NY"],
            "GrossApproval": [100_000.0, 200_000.0, 50_000.0],
            "SBAGuaranteedApproval": [90_000.0, 100_000.0, 25_000.0],
            "TermInMonths": [120.0, 60.0, pd.NA],
            "NaicsCode": [722110.0, 541940.0, pd.NA],
            "BusinessAge": [
                "Existing, 5 or more years",
                "New Business or 2 years or less",
                pd.NA,
            ],
            "CongressionalDistrict": [10.0, 4.0, pd.NA],
            "approval_month": [1, 12, 6],
            "naics_sector": pd.Series(["72", "54", pd.NA], dtype="string"),
            "sba_guarantee_ratio": [0.9, 0.5, 0.75],
            "default": [0, 1, 0],
        }
    )


def test_build_features_creates_expected_values() -> None:
    """Engineered features should have the expected values."""
    feature_df = build_features(make_cleaned_dataframe())

    assert feature_df["log_gross_approval"].notna().all()
    assert feature_df["log_sba_guaranteed_approval"].notna().all()
    assert feature_df["term_years"].iloc[0] == 10.0
    assert feature_df["term_years"].iloc[1] == 5.0

    assert feature_df["guarantee_ratio_bucket"].astype("string").tolist() == [
        "very_high",
        "low",
        "medium",
    ]

    assert feature_df["same_state_lender"].tolist() == [1, 0, 1]


def test_build_features_maps_business_age_and_missing_values() -> None:
    """Business age categories should map consistently."""
    feature_df = build_features(make_cleaned_dataframe())

    assert feature_df["business_age_group"].tolist() == [
        "existing_5_plus",
        "new_2_or_less",
        "unknown",
    ]


def test_build_features_converts_categorical_columns() -> None:
    """District, month, and NAICS sector should be categorical strings."""
    feature_df = build_features(make_cleaned_dataframe())

    assert feature_df["CongressionalDistrict"].tolist() == [
        "10",
        "4",
        "unknown",
    ]
    assert feature_df["approval_month"].tolist() == ["1", "12", "6"]
    assert feature_df["naics_sector"].tolist() == ["72", "54", "unknown"]


def test_build_features_removes_raw_and_redundant_columns() -> None:
    """Columns replaced by derived features should not reach modeling."""
    feature_df = build_features(make_cleaned_dataframe())

    assert not set(DROP_AFTER_ENGINEERING).intersection(feature_df.columns)

    assert "default" in feature_df.columns
    assert "log_gross_approval" in feature_df.columns
    assert "business_age_group" in feature_df.columns


def test_build_features_rejects_unknown_business_age() -> None:
    """Unexpected BusinessAge categories must not be silently mapped."""
    cleaned_df = make_cleaned_dataframe()
    cleaned_df.loc[0, "BusinessAge"] = "Unexpected category"

    with pytest.raises(ValueError, match="Unmapped BusinessAge values"):
        build_features(cleaned_df)


def test_validate_required_columns_rejects_missing_columns() -> None:
    """Missing required columns should produce a clear error."""
    cleaned_df = make_cleaned_dataframe().drop(columns=["BankState"])

    with pytest.raises(ValueError, match="BankState"):
        validate_required_columns(cleaned_df)