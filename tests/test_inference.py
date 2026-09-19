"""Tests for raw approval-time inference feature creation."""

from __future__ import annotations

import pandas as pd
import pytest

from src.features.inference import (
    EXPECTED_MODEL_FEATURES,
    build_inference_features,
    validate_raw_approval_fields,
)


def make_raw_approval_dataframe() -> pd.DataFrame:
    """Create a raw approval-time fixture for one loan."""
    return pd.DataFrame(
        {
            "ApprovalDate": ["2018-02-20"],
            "ApprovalFY": [2018],
            "BankState": ["TX"],
            "BorrState": ["TX"],
            "BusinessAge": ["Existing, 5 or more years"],
            "BusinessType": ["Corporation"],
            "CollateralInd": ["Y"],
            "CongressionalDistrict": [10.0],
            "FixedorVariableInterestInd": ["V"],
            "FranchiseCode": [pd.NA],
            "GrossApproval": [100_000.0],
            "InitialInterestRate": [5.5],
            "JobsSupported": [12.0],
            "NaicsCode": [722110.0],
            "ProcessingMethod": ["Preferred Lenders Program"],
            "ProjectState": ["TX"],
            "RevolverStatus": ["N"],
            "SBADistrictOffice": ["DALLAS"],
            "SBAGuaranteedApproval": [90_000.0],
            "SoldSecMrktInd": ["Y"],
            "TermInMonths": [120.0],
        }
    )


def test_build_inference_features_creates_model_schema() -> None:
    """Raw inputs should produce exactly the training-model features."""
    feature_df = build_inference_features(
        make_raw_approval_dataframe()
    )

    assert set(feature_df.columns) == EXPECTED_MODEL_FEATURES
    assert feature_df.shape == (1, 23)


def test_build_inference_features_creates_expected_values() -> None:
    """Deterministic inference features should match training logic."""
    feature_df = build_inference_features(
        make_raw_approval_dataframe()
    )

    assert feature_df["sba_guarantee_ratio"].iloc[0] == 0.9
    assert feature_df["is_franchise"].iloc[0] == 0
    assert feature_df["naics_sector"].iloc[0] == "72"
    assert feature_df["approval_month"].iloc[0] == "2"
    assert feature_df["sold_secondary_market"].iloc[0] == 1
    assert feature_df["term_years"].iloc[0] == 10.0
    assert feature_df["same_state_lender"].iloc[0] == 1
    assert (
        feature_df["business_age_group"].iloc[0]
        == "existing_5_plus"
    )


def test_build_inference_features_converts_suspicious_zeros() -> None:
    """Zero interest rate and term should become missing values."""
    raw_df = make_raw_approval_dataframe()
    raw_df["InitialInterestRate"] = 0.0
    raw_df["TermInMonths"] = 0.0

    feature_df = build_inference_features(raw_df)

    assert pd.isna(feature_df["InitialInterestRate"].iloc[0])
    assert pd.isna(feature_df["term_years"].iloc[0])


def test_build_inference_features_rejects_invalid_secondary_market_code() -> None:
    """Unknown secondary-market codes must not be silently mapped."""
    raw_df = make_raw_approval_dataframe()
    raw_df["SoldSecMrktInd"] = "UNKNOWN"

    with pytest.raises(ValueError, match="Unexpected SoldSecMrktInd"):
        build_inference_features(raw_df)


def test_validate_raw_approval_fields_rejects_missing_input() -> None:
    """The inference converter should require all raw approval inputs."""
    raw_df = make_raw_approval_dataframe().drop(
        columns=["GrossApproval"]
    )

    with pytest.raises(ValueError, match="GrossApproval"):
        validate_raw_approval_fields(raw_df)