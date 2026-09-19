"""Raw approval-time input schema for model inference."""

from __future__ import annotations

import pandas as pd


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


def validate_raw_approval_fields(raw_df: pd.DataFrame) -> None:
    """Raise an error when required approval-time inputs are absent."""
    missing_fields = RAW_APPROVAL_FIELDS - set(raw_df.columns)

    if missing_fields:
        raise ValueError(
            "Raw approval input is missing required fields: "
            f"{sorted(missing_fields)}"
        )
