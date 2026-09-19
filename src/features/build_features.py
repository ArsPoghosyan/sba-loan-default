"""Feature engineering utilities for SBA 7(a) loan data."""

from __future__ import annotations

import numpy as np
import pandas as pd


BUSINESS_AGE_MAP = {
    "Existing, 5 or more years": "existing_5_plus",
    "Existing or more than 2 years old": "existing_2_plus",
    "New, Less than 1 Year old": "new_under_1",
    "New Business or 2 years or less": "new_2_or_less",
    "Startup, Loan Funds will Open Business": "startup",
    "Change of Ownership": "change_ownership",
    "Unanswered": "unknown",
    "Less than 2 years old but at least 1": "young_1_to_2",
    "Less than 3 years old but at least 2": "young_2_to_3",
    "Less than 4 years old but at least 3": "young_3_to_4",
    "Less than 5 years old but at least 4": "young_4_to_5",
    "Loan Funds will Open Business": "startup",
}

REQUIRED_COLUMNS = {
    "BorrState",
    "BankState",
    "GrossApproval",
    "SBAGuaranteedApproval",
    "TermInMonths",
    "NaicsCode",
    "BusinessAge",
    "CongressionalDistrict",
    "approval_month",
    "naics_sector",
    "sba_guarantee_ratio",
}

DROP_AFTER_ENGINEERING = [
    "GrossApproval",
    "SBAGuaranteedApproval",
    "TermInMonths",
    "NaicsCode",
    "BusinessAge",
    "BorrState",
]


def validate_required_columns(df: pd.DataFrame) -> None:
    """Raise an error if required cleaned-data columns are absent."""
    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            "Cleaned dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )


def build_features(cleaned_df: pd.DataFrame) -> pd.DataFrame:
    """Create deterministic model features from the cleaned SBA dataset."""
    validate_required_columns(cleaned_df)

    feature_df = cleaned_df.copy()

    feature_df["log_gross_approval"] = np.log1p(
        feature_df["GrossApproval"]
    )
    feature_df["log_sba_guaranteed_approval"] = np.log1p(
        feature_df["SBAGuaranteedApproval"]
    )

    feature_df["term_years"] = feature_df["TermInMonths"] / 12

    feature_df["guarantee_ratio_bucket"] = pd.cut(
        feature_df["sba_guarantee_ratio"],
        bins=[0, 0.5, 0.75, 0.85, 1.0],
        labels=["low", "medium", "high", "very_high"],
        include_lowest=True,
    )

    feature_df["same_state_lender"] = (
        feature_df["BorrState"] == feature_df["BankState"]
    ).astype(int)

    unmapped_business_age = (
        set(feature_df["BusinessAge"].dropna().unique())
        - set(BUSINESS_AGE_MAP)
    )

    if unmapped_business_age:
        raise ValueError(
            "Unmapped BusinessAge values: "
            f"{sorted(unmapped_business_age)}"
        )

    feature_df["business_age_group"] = (
        feature_df["BusinessAge"]
        .map(BUSINESS_AGE_MAP)
        .fillna("unknown")
    )

    feature_df["naics_sector"] = (
        feature_df["naics_sector"]
        .astype("string")
        .fillna("unknown")
    )

    feature_df["CongressionalDistrict"] = (
        feature_df["CongressionalDistrict"]
        .astype("Int64")
        .astype("string")
        .fillna("unknown")
    )

    feature_df["approval_month"] = (
        feature_df["approval_month"]
        .astype("string")
    )

    return feature_df.drop(columns=DROP_AFTER_ENGINEERING)