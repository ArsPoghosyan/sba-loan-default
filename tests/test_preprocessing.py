"""Tests for SBA model preprocessing."""

from __future__ import annotations

import pandas as pd
import numpy as np

from src.features.preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_preprocessor,
)


def make_feature_dataframe() -> pd.DataFrame:
    """Create a small fixture containing all model input columns."""
    data = {
        "ApprovalFY": [2016, 2017, 2018],
        "InitialInterestRate": [5.0, np.nan, 6.5],
        "JobsSupported": [3.0, 10.0, 1.0],
        "sba_guarantee_ratio": [0.5, 0.75, 0.9],
        "is_franchise": [0, 1, 0],
        "sold_secondary_market": [1.0, np.nan, 0.0],
        "log_gross_approval": [10.0, 11.0, 9.0],
        "log_sba_guaranteed_approval": [9.5, 10.5, 8.5],
        "term_years": [5.0, 10.0, 7.0],
        "same_state_lender": [1, 0, 1],
        "BankState": ["TX", "CA", "TX"],
        "ProcessingMethod": ["Standard", "Express", "Standard"],
        "FixedorVariableInterestInd": ["V", "F", "V"],
        "ProjectState": ["TX", "CA", "NY"],
        "SBADistrictOffice": ["DALLAS", "LOS ANGELES", "NEW YORK"],
        "CongressionalDistrict": ["10", "4", "unknown"],
        "BusinessType": ["Corporation", "Partnership", "Corporation"],
        "RevolverStatus": ["N", "Y", "N"],
        "CollateralInd": ["Y", "N", "Y"],
        "naics_sector": ["72", "54", "unknown"],
        "approval_month": ["1", "12", "6"],
        "guarantee_ratio_bucket": ["low", "medium", "very_high"],
        "business_age_group": [
            "existing_5_plus",
            "startup",
            "unknown",
        ],
    }

    return pd.DataFrame(data)


def test_build_preprocessor_uses_expected_feature_groups() -> None:
    """The preprocessor should retain the locked model feature schema."""
    preprocessor = build_preprocessor()

    transformer_names = [
        name
        for name, _, _ in preprocessor.transformers
    ]

    assert transformer_names == ["num", "cat"]
    assert preprocessor.transformers[0][2] == NUMERIC_FEATURES
    assert preprocessor.transformers[1][2] == CATEGORICAL_FEATURES


def test_preprocessor_fits_and_transforms_training_data() -> None:
    """Imputation, scaling, and encoding should work with missing values."""
    feature_df = make_feature_dataframe()
    preprocessor = build_preprocessor()

    transformed_features = preprocessor.fit_transform(feature_df)

    assert transformed_features.shape[0] == len(feature_df)
    assert transformed_features.shape[1] > (
        len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)
    )


def test_preprocessor_handles_unseen_categories() -> None:
    """Later-year categories absent during fitting should not cause errors."""
    train_df = make_feature_dataframe().iloc[:2].copy()
    validation_df = make_feature_dataframe().iloc[[2]].copy()

    preprocessor = build_preprocessor()
    preprocessor.fit(train_df)

    transformed_validation = preprocessor.transform(validation_df)

    assert transformed_validation.shape[0] == 1