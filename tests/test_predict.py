"""Tests for saved-model inference utilities."""

from __future__ import annotations

import joblib
import numpy as np
import pytest

from src.features.preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
)
from src.models.predict import (
    load_model_artifact,
    predict_from_raw_inputs,
    validate_model_artifact,
)


class ProbabilityPipeline:
    """Predictable pipeline substitute for inference tests."""

    def predict_proba(self, X: object) -> np.ndarray:
        return np.array([[0.2, 0.8]])


def make_raw_inputs() -> dict[str, object]:
    """Create one valid raw approval-time input record."""
    return {
        "ApprovalDate": "2018-02-20",
        "ApprovalFY": 2018,
        "BankState": "TX",
        "BorrState": "TX",
        "BusinessAge": "Existing, 5 or more years",
        "BusinessType": "Corporation",
        "CollateralInd": "Y",
        "CongressionalDistrict": 10.0,
        "FixedorVariableInterestInd": "V",
        "FranchiseCode": None,
        "GrossApproval": 100_000.0,
        "InitialInterestRate": 5.5,
        "JobsSupported": 12.0,
        "NaicsCode": 722110.0,
        "ProcessingMethod": "Preferred Lenders Program",
        "ProjectState": "TX",
        "RevolverStatus": "N",
        "SBADistrictOffice": "DALLAS",
        "SBAGuaranteedApproval": 90_000.0,
        "SoldSecMrktInd": "Y",
        "TermInMonths": 120.0,
    }


def make_model_artifact() -> dict[str, object]:
    """Create a lightweight valid artifact for inference tests."""
    return {
        "model_name": "Tuned XGBoost",
        "pipeline": ProbabilityPipeline(),
        "threshold": 0.7482,
        "feature_columns": (
            NUMERIC_FEATURES + CATEGORICAL_FEATURES
        ),
        "training_period": "FY2010-FY2017",
        "validation_period": "FY2018",
    }


def test_predict_from_raw_inputs_returns_expected_result() -> None:
    """Raw input should produce probability and locked-threshold class."""
    result = predict_from_raw_inputs(
        make_raw_inputs(),
        make_model_artifact(),
    )

    assert result.model_name == "Tuned XGBoost"
    assert result.default_probability == 0.8
    assert result.threshold == 0.7482
    assert result.predicted_default == 1


def test_validate_model_artifact_rejects_missing_metadata() -> None:
    """An incomplete artifact should fail before prediction."""
    artifact = make_model_artifact()
    del artifact["threshold"]

    with pytest.raises(ValueError, match="threshold"):
        validate_model_artifact(artifact)


def test_validate_model_artifact_rejects_invalid_threshold() -> None:
    """A threshold must represent a valid probability cutoff."""
    artifact = make_model_artifact()
    artifact["threshold"] = 1.5

    with pytest.raises(ValueError, match="between 0 and 1"):
        validate_model_artifact(artifact)


def test_load_model_artifact_loads_valid_joblib_file(
    tmp_path: object,
) -> None:
    """A valid serialized artifact should load successfully."""
    model_path = tmp_path / "model.joblib"
    artifact = make_model_artifact()

    joblib.dump(artifact, model_path)

    loaded_artifact = load_model_artifact(model_path)

    assert loaded_artifact["model_name"] == "Tuned XGBoost"
    assert loaded_artifact["threshold"] == 0.7482