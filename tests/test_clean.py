"""Tests for SBA cleaning functions."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.clean import (
    DROP_COLUMNS,
    clean_sba_loans,
    validate_required_columns,
)


def make_raw_dataframe() -> pd.DataFrame:
    """Create a small raw-data fixture matching the SBA schema."""
    raw_df = pd.DataFrame(
        {
            "LoanStatus": ["P I F", "CHGOFF", "CANCLD"],
            "GrossApproval": [100_000.0, 200_000.0, 50_000.0],
            "SBAGuaranteedApproval": [90_000.0, 100_000.0, 45_000.0],
            "FranchiseCode": [pd.NA, "12345", pd.NA],
            "NaicsCode": [722110, 541940, 722211],
            "ApprovalDate": [
                "2017-01-15",
                "2018-02-20",
                "2019-03-10",
            ],
            "SoldSecMrktInd": ["Y", pd.NA, "N"],
            "InitialInterestRate": [0.0, 5.0, 6.0],
            "TermInMonths": [0.0, 120.0, 60.0],
        }
    )

    for column in DROP_COLUMNS:
        if column not in raw_df.columns:
            raw_df[column] = pd.NA

    return raw_df


def test_clean_sba_loans_keeps_only_final_outcomes() -> None:
    """Only paid-in-full and charged-off loans belong in the modeling sample."""
    cleaned_df = clean_sba_loans(make_raw_dataframe())

    assert len(cleaned_df) == 2
    assert cleaned_df["default"].tolist() == [0, 1]


def test_clean_sba_loans_creates_expected_features() -> None:
    """Deterministic approval-time features should be calculated correctly."""
    cleaned_df = clean_sba_loans(make_raw_dataframe())

    assert cleaned_df["sba_guarantee_ratio"].tolist() == [0.9, 0.5]
    assert cleaned_df["is_franchise"].tolist() == [0, 1]
    assert cleaned_df["naics_sector"].tolist() == ["72", "54"]
    assert cleaned_df["approval_month"].tolist() == [1, 2]
    assert cleaned_df["sold_secondary_market"].iloc[0] == 1
    assert pd.isna(cleaned_df["sold_secondary_market"].iloc[1])


def test_clean_sba_loans_converts_suspicious_zeros_to_missing() -> None:
    """Zero interest rates and terms should become missing values."""
    cleaned_df = clean_sba_loans(make_raw_dataframe())

    assert pd.isna(cleaned_df["InitialInterestRate"].iloc[0])
    assert pd.isna(cleaned_df["TermInMonths"].iloc[0])


def test_clean_sba_loans_removes_leakage_and_identity_columns() -> None:
    """All configured drop columns must be absent from the cleaned dataset."""
    cleaned_df = clean_sba_loans(make_raw_dataframe())

    assert not set(DROP_COLUMNS).intersection(cleaned_df.columns)


def test_clean_sba_loans_rejects_unexpected_secondary_market_codes() -> None:
    """Unknown secondary-market values must not silently become zero."""
    raw_df = make_raw_dataframe()
    raw_df.loc[0, "SoldSecMrktInd"] = "UNKNOWN"

    with pytest.raises(ValueError, match="Unexpected SoldSecMrktInd"):
        clean_sba_loans(raw_df)


def test_clean_sba_loans_rejects_invalid_guarantee_ratio() -> None:
    """Guaranteed approval cannot exceed gross approval."""
    raw_df = make_raw_dataframe()
    raw_df.loc[0, "SBAGuaranteedApproval"] = 200_000.0

    with pytest.raises(ValueError, match="guarantee ratios"):
        clean_sba_loans(raw_df)


def test_validate_required_columns_rejects_missing_columns() -> None:
    """The cleaner should fail clearly when a required field is absent."""
    raw_df = make_raw_dataframe().drop(columns=["NaicsCode"])

    with pytest.raises(ValueError, match="NaicsCode"):
        validate_required_columns(raw_df)