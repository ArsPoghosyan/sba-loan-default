"""Streamlit interface for SBA 7(a) loan charge-off prediction."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import streamlit as st

from src.models.predict import (
    load_model_artifact,
    predict_from_raw_inputs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "tuned_xgboost_pipeline.joblib"

STATE_OPTIONS = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL",
    "GA", "HI", "IA", "ID", "IL", "IN", "KS", "KY", "LA", "MA",
    "MD", "ME", "MI", "MN", "MO", "MS", "MT", "NC", "ND", "NE",
    "NH", "NJ", "NM", "NV", "NY", "OH", "OK", "OR", "PA", "RI",
    "SC", "SD", "TN", "TX", "UT", "VA", "VT", "WA", "WI", "WV",
    "WY",
]

BUSINESS_AGE_OPTIONS = [
    "Existing, 5 or more years",
    "Existing or more than 2 years old",
    "New, Less than 1 Year old",
    "New Business or 2 years or less",
    "Startup, Loan Funds will Open Business",
    "Change of Ownership",
    "Unanswered",
    "Less than 2 years old but at least 1",
    "Less than 3 years old but at least 2",
    "Less than 4 years old but at least 3",
    "Less than 5 years old but at least 4",
    "Loan Funds will Open Business",
]

BUSINESS_TYPE_OPTIONS = [
    "CORPORATION",
    "INDIVIDUAL",
    "PARTNERSHIP",
]

PROCESSING_METHOD_OPTIONS = [
    "7a General",
    "7a with EWCP",
    "Builders CAPLine",
    "Certified Lenders Program",
    "Certified Lenders Program with EWCP",
    "Community Advantage Initiative",
    "Community Advantage International Trade",
    "Community Express",
    "Contract CAPLine",
    "Dealer Floor Plan Loans",
    "Export Express",
    "Export Import Harmonization  (EXIM)",
    "Gulf Opportunity Pilot Loan Program",
    "International Trade Loans",
    "Patriot Express Loans",
    "Preferred Lenders Program",
    "Preferred Lenders with EWCP",
    "Rural Loan Initiative",
    "SBA Express Bridge Loan",
    "SBA Express Program",
    "Seasonal CAPLine",
    "Small Asset Base Line of Credit (CAPLine)",
    "Small Loan Advantage Initiative",
    "Working Capital CAPLine",
]


def calculate_fiscal_year(approval_date: date) -> int:
    """Return the US federal fiscal year for an approval date."""
    return approval_date.year + int(approval_date.month >= 10)


@st.cache_resource
def get_model_artifact() -> dict[str, object]:
    """Load the local frozen model artifact once per Streamlit session."""
    return load_model_artifact(MODEL_PATH)


def optional_yes_no_to_indicator(value: str) -> str | None:
    """Convert a form choice to the SBA Y/N convention or missing value."""
    if value == "Yes":
        return "Y"

    if value == "No":
        return "N"

    return None


st.set_page_config(
    page_title="SBA Loan Charge-Off Predictor",
    page_icon="",
    layout="wide",
)

st.title("SBA 7(a) Loan Charge-Off Predictor")
st.caption(
    "Educational portfolio demo using historical SBA 7(a) loans "
    "approved from FY2010-FY2019."
)

try:
    model_artifact = get_model_artifact()
except FileNotFoundError:
    st.error(
        "Model artifact not found. Create it first by running "
        "`src.models.train_final` from the project root."
    )
    st.stop()

st.info(
    "Enter approval-time information below. The app engineers the same "
    "features used during training, then applies the frozen tuned "
    "XGBoost model."
)

with st.form("loan_input_form"):
    st.subheader("Loan and applicant information")

    left_column, right_column = st.columns(2)

    with left_column:
        approval_date = st.date_input(
            "Approval date",
            value=date(2018, 6, 1),
            min_value=date(2010, 1, 1),
            max_value=date(2019, 12, 31),
        )

        gross_approval = st.number_input(
            "Gross approval amount ($)",
            min_value=1.0,
            value=100_000.0,
            step=1_000.0,
        )

        guaranteed_approval = st.number_input(
            "SBA-guaranteed approval amount ($)",
            min_value=0.0,
            max_value=gross_approval,
            value=min(90_000.0, gross_approval),
            step=1_000.0,
        )

        term_in_months = st.number_input(
            "Loan term (months)",
            min_value=0,
            value=120,
            step=1,
            help="A value of 0 is treated as missing, matching training.",
        )

        initial_interest_rate = st.number_input(
            "Initial interest rate (%)",
            min_value=0.0,
            value=5.5,
            step=0.25,
            help="A value of 0 is treated as missing, matching training.",
        )

        jobs_supported = st.number_input(
            "Jobs supported",
            min_value=0,
            value=10,
            step=1,
        )

        naics_code = st.number_input(
            "NAICS code",
            min_value=0,
            value=722110,
            step=1,
            help="Use the six-digit industry code when available.",
        )

        business_age = st.selectbox(
            "Business age",
            BUSINESS_AGE_OPTIONS,
        )

        business_type = st.selectbox(
            "Business type",
            BUSINESS_TYPE_OPTIONS,
        )

    with right_column:
        borrower_state = st.selectbox(
            "Borrower state",
            STATE_OPTIONS,
            index=STATE_OPTIONS.index("TX"),
        )

        lender_state = st.selectbox(
            "Lender state",
            STATE_OPTIONS,
            index=STATE_OPTIONS.index("TX"),
        )

        project_state = st.selectbox(
            "Project state",
            STATE_OPTIONS,
            index=STATE_OPTIONS.index("TX"),
        )

        processing_method = st.selectbox(
            "Processing method",
            PROCESSING_METHOD_OPTIONS,
            index=PROCESSING_METHOD_OPTIONS.index(
                "Preferred Lenders Program"
            ),
        )

        interest_type = st.selectbox(
            "Interest type",
            ["V", "F"],
            format_func=lambda value: (
                "Variable" if value == "V" else "Fixed"
            ),
        )

        congressional_district = st.number_input(
            "Congressional district",
            min_value=0,
            max_value=99,
            value=10,
            step=1,
        )

        district_office = st.text_input(
            "SBA district office",
            value="DALLAS",
            help="For example: DALLAS, NEW YORK, or LOS ANGELES.",
        )

        collateral_indicator = st.selectbox(
            "Collateral required",
            ["Yes", "No"],
        )

        revolver_indicator = st.selectbox(
            "Revolving loan",
            ["No", "Yes"],
        )

        franchise_indicator = st.selectbox(
            "Franchise business",
            ["No", "Yes"],
        )

        secondary_market_indicator = st.selectbox(
            "Sold in secondary market",
            ["Not reported", "Yes", "No"],
        )

    submitted = st.form_submit_button(
        "Estimate charge-off risk",
        type="primary",
    )

if submitted:
    raw_inputs = {
        "ApprovalDate": approval_date.isoformat(),
        "ApprovalFY": calculate_fiscal_year(approval_date),
        "BankState": lender_state,
        "BorrState": borrower_state,
        "BusinessAge": business_age,
        "BusinessType": business_type,
        "CollateralInd": optional_yes_no_to_indicator(
            collateral_indicator
        ),
        "CongressionalDistrict": float(congressional_district),
        "FixedorVariableInterestInd": interest_type,
        "FranchiseCode": (
            "APP_REPORTED_FRANCHISE"
            if franchise_indicator == "Yes"
            else None
        ),
        "GrossApproval": float(gross_approval),
        "InitialInterestRate": float(initial_interest_rate),
        "JobsSupported": float(jobs_supported),
        "NaicsCode": float(naics_code),
        "ProcessingMethod": processing_method,
        "ProjectState": project_state,
        "RevolverStatus": optional_yes_no_to_indicator(
            revolver_indicator
        ),
        "SBADistrictOffice": district_office.strip().upper(),
        "SBAGuaranteedApproval": float(guaranteed_approval),
        "SoldSecMrktInd": optional_yes_no_to_indicator(
            secondary_market_indicator
        ),
        "TermInMonths": float(term_in_months),
    }

    try:
        prediction = predict_from_raw_inputs(
            raw_inputs,
            model_artifact,
        )
    except ValueError as error:
        st.error(f"Prediction could not be generated: {error}")
        st.stop()

    probability_percent = prediction.default_probability * 100
    threshold_percent = prediction.threshold * 100

    st.divider()
    st.subheader("Prediction result")

    metric_column, threshold_column = st.columns(2)

    with metric_column:
        st.metric(
            "Predicted charge-off probability",
            f"{probability_percent:.1f}%",
        )

    with threshold_column:
        st.metric(
            "Operating threshold",
            f"{threshold_percent:.1f}%",
        )

    st.progress(
        min(max(prediction.default_probability, 0.0), 1.0)
    )

    if prediction.predicted_default == 1:
        st.warning(
            "Elevated charge-off risk at the selected operating threshold."
        )
    else:
        st.success(
            "Below the selected operating threshold."
        )

    st.caption(
        "The classification uses the frozen FY2018 F1-maximizing "
        f"threshold of {prediction.threshold:.4f}. A probability below "
        "this threshold does not mean that a loan is risk-free."
    )

st.divider()
st.subheader("Important limitation")
st.caption(
    "This is an educational machine-learning portfolio project based on "
    "historical SBA data. It is not a production credit-decision tool and "
    "must not be used to approve, deny, price, or otherwise make lending "
    "decisions."
)
