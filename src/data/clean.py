"""Cleaning utilities for SBA 7(a) loan data."""

from __future__ import annotations

import pandas as pd


FINAL_OUTCOME_MAP = {
    "P I F": 0,
    "CHGOFF": 1,
}

REQUIRED_COLUMNS = {
    "LoanStatus",
    "GrossApproval",
    "SBAGuaranteedApproval",
    "FranchiseCode",
    "NaicsCode",
    "ApprovalDate",
    "SoldSecMrktInd",
    "InitialInterestRate",
    "TermInMonths",
}

DROP_COLUMNS = [
    "LoanStatus",
    "PaidInFullDate",
    "ChargeOffDate",
    "GrossChargeOffAmount",
    "AsOfDate",
    "Program",
    "LocationID",
    "BorrName",
    "BorrStreet",
    "BorrCity",
    "BorrZip",
    "BankName",
    "BankFDICNumber",
    "BankNCUANumber",
    "BankStreet",
    "BankCity",
    "BankZip",
    "FranchiseCode",
    "FranchiseName",
    "NaicsDescription",
    "SoldSecMrktInd",
    "ProjectCounty",
    "ApprovalDate",
    "FirstDisbursementDate",
]


def validate_required_columns(df: pd.DataFrame) -> None:
    """Raise an error if required raw-data columns are absent."""
    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Raw dataset is missing required columns: {sorted(missing_columns)}"
        )


def clean_sba_loans(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Create a leakage-safe SBA 7(a) modeling dataset.

    Keeps only loans with clear final outcomes, creates deterministic
    approval-time features, removes leakage/identity columns, and preserves
    remaining missing values for train-only pipeline imputation.
    """
    validate_required_columns(raw_df)

    clean_df = raw_df.loc[
        raw_df["LoanStatus"].isin(FINAL_OUTCOME_MAP)
    ].copy()

    if clean_df.empty:
        raise ValueError("No rows with clear final outcomes were found.")

    clean_df["default"] = clean_df["LoanStatus"].map(FINAL_OUTCOME_MAP)

    if not clean_df["GrossApproval"].gt(0).all():
        raise ValueError(
            "GrossApproval must be positive before calculating guarantee ratios."
        )

    clean_df["sba_guarantee_ratio"] = (
        clean_df["SBAGuaranteedApproval"] / clean_df["GrossApproval"]
    )

    if not clean_df["sba_guarantee_ratio"].between(0, 1).all():
        raise ValueError("SBA guarantee ratios must fall between 0 and 1.")

    clean_df["is_franchise"] = clean_df["FranchiseCode"].notna().astype(int)

    clean_df["naics_sector"] = (
        clean_df["NaicsCode"]
        .astype("Int64")
        .astype("string")
        .str[:2]
    )

    clean_df["ApprovalDate"] = pd.to_datetime(
        clean_df["ApprovalDate"],
        errors="coerce",
    )

    clean_df["approval_month"] = (
        clean_df["ApprovalDate"]
        .dt.month
        .astype("Int64")
    )

    observed_secondary_market_values = set(
        clean_df["SoldSecMrktInd"].dropna().unique()
    )
    expected_secondary_market_values = {"Y", "N"}

    unexpected_values = (
        observed_secondary_market_values
        - expected_secondary_market_values
    )

    if unexpected_values:
        raise ValueError(
            "Unexpected SoldSecMrktInd values: "
            f"{sorted(unexpected_values)}"
        )

    clean_df["sold_secondary_market"] = clean_df[
        "SoldSecMrktInd"
    ].map({"Y": 1, "N": 0})

    missing_drop_columns = set(DROP_COLUMNS) - set(clean_df.columns)

    if missing_drop_columns:
        raise ValueError(
            "Raw dataset is missing expected removable columns: "
            f"{sorted(missing_drop_columns)}"
        )

    clean_df = clean_df.drop(columns=DROP_COLUMNS)

    clean_df.loc[
        clean_df["InitialInterestRate"] == 0,
        "InitialInterestRate",
    ] = pd.NA

    clean_df.loc[
        clean_df["TermInMonths"] == 0,
        "TermInMonths",
    ] = pd.NA

    return clean_df