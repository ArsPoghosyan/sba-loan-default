"""Convert raw approval-time inputs into model-ready inference features."""

from __future__ import annotations

import pandas as pd

from src.features.build_features import build_features
from src.features.preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
)


RAW_APPROVAL_FIELDS = {
    "ApprovalDate",
    "ApprovalFY",
    "BankState",
    "BorrState",
    "BusinessAge",
    "BusinessType",
    "CollateralInd",
    "CongressionalDistrict",
    "FixedorVariableInterestInd",
    "FranchiseCode",
    "GrossApproval",
    "InitialInterestRate",
    "JobsSupported",
    "NaicsCode",
    "ProcessingMethod",
    "ProjectState",
    "RevolverStatus",
    "SBADistrictOffice",
    "SBAGuaranteedApproval",
    "SoldSecMrktInd",
    "TermInMonths",
}

RAW_COLUMNS_TO_DROP = [
    "ApprovalDate",
    "FranchiseCode",
    "SoldSecMrktInd",
]

EXPECTED_MODEL_FEATURES = set(
    NUMERIC_FEATURES + CATEGORICAL_FEATURES
)


def validate_raw_approval_fields(raw_df: pd.DataFrame) -> None:
    """Raise an error when required approval-time inputs are absent."""
    missing_fields = RAW_APPROVAL_FIELDS - set(raw_df.columns)

    if missing_fields:
        raise ValueError(
            "Raw approval input is missing required fields: "
            f"{sorted(missing_fields)}"
        )


def build_inference_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Create model-ready features from raw approval-time loan inputs.

    This reproduces the deterministic transformations used for the
    training dataset without requiring an outcome field such as LoanStatus.
    """
    validate_raw_approval_fields(raw_df)

    inference_df = raw_df.copy()

    if not inference_df["GrossApproval"].gt(0).all():
        raise ValueError(
            "GrossApproval must be positive before calculating "
            "guarantee ratios."
        )

    inference_df["sba_guarantee_ratio"] = (
        inference_df["SBAGuaranteedApproval"]
        / inference_df["GrossApproval"]
    )

    if not inference_df["sba_guarantee_ratio"].between(0, 1).all():
        raise ValueError(
            "SBA guarantee ratios must fall between 0 and 1."
        )

    inference_df["is_franchise"] = (
        inference_df["FranchiseCode"].notna().astype(int)
    )

    inference_df["naics_sector"] = (
        inference_df["NaicsCode"]
        .astype("Int64")
        .astype("string")
        .str[:2]
    )

    approval_dates = pd.to_datetime(
        inference_df["ApprovalDate"],
        errors="coerce",
    )

    if approval_dates.isna().any():
        raise ValueError(
            "ApprovalDate must contain valid dates."
        )

    inference_df["approval_month"] = (
        approval_dates.dt.month.astype("Int64")
    )

    observed_secondary_market_values = set(
        inference_df["SoldSecMrktInd"].dropna().unique()
    )

    unexpected_values = (
        observed_secondary_market_values - {"Y", "N"}
    )

    if unexpected_values:
        raise ValueError(
            "Unexpected SoldSecMrktInd values: "
            f"{sorted(unexpected_values)}"
        )

    inference_df["sold_secondary_market"] = (
        inference_df["SoldSecMrktInd"].map({"Y": 1, "N": 0})
    )

    inference_df.loc[
        inference_df["InitialInterestRate"] == 0,
        "InitialInterestRate",
    ] = pd.NA

    inference_df.loc[
        inference_df["TermInMonths"] == 0,
        "TermInMonths",
    ] = pd.NA

    inference_df = inference_df.drop(
        columns=RAW_COLUMNS_TO_DROP,
    )

    feature_df = build_features(inference_df)

    missing_features = EXPECTED_MODEL_FEATURES - set(
        feature_df.columns
    )
    unexpected_features = set(feature_df.columns) - (
        EXPECTED_MODEL_FEATURES
    )

    if missing_features or unexpected_features:
        raise ValueError(
            "Inference feature schema does not match the model schema. "
            f"Missing: {sorted(missing_features)}. "
            f"Unexpected: {sorted(unexpected_features)}."
        )

    return feature_df