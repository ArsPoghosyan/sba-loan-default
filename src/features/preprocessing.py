"""Preprocessing utilities for SBA default-model features."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = [
    "ApprovalFY",
    "InitialInterestRate",
    "JobsSupported",
    "sba_guarantee_ratio",
    "is_franchise",
    "sold_secondary_market",
    "log_gross_approval",
    "log_sba_guaranteed_approval",
    "term_years",
    "same_state_lender",
]

CATEGORICAL_FEATURES = [
    "BankState",
    "ProcessingMethod",
    "FixedorVariableInterestInd",
    "ProjectState",
    "SBADistrictOffice",
    "CongressionalDistrict",
    "BusinessType",
    "RevolverStatus",
    "CollateralInd",
    "naics_sector",
    "approval_month",
    "guarantee_ratio_bucket",
    "business_age_group",
]


def build_preprocessor() -> ColumnTransformer:
    """Create the preprocessing pipeline used by every candidate model.

    The returned transformer is intentionally unfitted. Fit it only on the
    FY2010–2017 training data, as part of a scikit-learn Pipeline.
    """
    numeric_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                ),
            ),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )