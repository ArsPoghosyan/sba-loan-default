# SBA 7(a) Loan Charge-Off Prediction

A binary-classification project that predicts whether an SBA 7(a) loan will be charged off, using SBA FOIA loan data for fiscal years 2010–2019.

The project implements an end-to-end machine-learning workflow: exploratory analysis, leakage-aware cleaning, feature engineering, chronological validation, model comparison, final evaluation, and interpretation.

## Project status

- [x] Exploratory data analysis
- [x] Data cleaning and processed dataset creation
- [x] Feature engineering and chronological train/validation/test split
- [x] Dummy-classifier and logistic-regression baselines
- [x] Random forest and XGBoost comparison
- [x] Model tuning and validation-threshold selection
- [x] Final FY2019 evaluation, calibration, and interpretation
- [ ] Reusable `src/` pipeline and automated tests

## Dataset

- Source: SBA FOIA 7(a) loan-data release
- Period: FY2010–FY2019
- Raw records: 545,751
- Modeling sample: 428,874 loans with clear final outcomes
- Target:
  - `0`: paid in full (`P I F`)
  - `1`: charged off (`CHGOFF`)
- Overall charge-off rate: 7.96%

The raw data is not committed to this repository. Place the SBA release at:

```text
data/raw/FOIA_7a_FY2010_FY2019_asof_260630.csv
```

The accompanying [SBA data dictionary](docs/7a_504_foia_data_dictionary.xlsx) is included for reference.

## Methodology

### Target and leakage controls

Only loans with clear final outcomes are retained: `P I F` and `CHGOFF`. Ambiguous statuses such as `CANCLD`, `EXEMPT`, and `COMMIT` are excluded.

Fields that reveal information after the loan outcome are removed, including `PaidInFullDate`, `ChargeOffDate`, `GrossChargeOffAmount`, and `AsOfDate`. Identity, address, and other high-cardinality fields are also excluded from the initial models.

### Feature engineering

The current feature set uses only information available at approval time. It includes:

- Log-transformed gross and SBA-guaranteed approval amounts
- Loan term in years
- SBA guarantee ratio and a ratio bucket
- Industry sector derived from NAICS
- Business-age groups
- Approval month, state, district, collateral, business, and loan-program characteristics
- Franchise, secondary-market, and same-state lender indicators

Zero interest rates and zero loan terms are treated as missing values. Remaining missing values are handled inside model pipelines, using training data only.

### Temporal evaluation design

The project uses a chronological split to better approximate deployment on future loans:

| Split | Approval years | Purpose |
| --- | --- | --- |
| Training | FY2010–FY2017 | Fit models and preprocessing |
| Validation | FY2018 | Compare models, tune, and select the threshold |
| Test | FY2019 | Final one-time evaluation |

The validation default rate is 10.13%, compared with 7.55% in training, illustrating why a random split would be less realistic.

## Model-selection results

Models below were trained on FY2010–FY2017 and evaluated on FY2018. PR-AUC is the primary model-selection metric because charge-offs are the minority class.

| Model | PR-AUC | ROC-AUC | F1 at 0.50 | Brier score |
| --- | ---: | ---: | ---: | ---: |
| Dummy classifier | 0.1013 | 0.5000 | 0.0000 | 0.1013 |
| Logistic regression | 0.5864 | 0.8777 | 0.5354 | 0.0616 |
| Logistic regression, balanced classes | 0.4900 | 0.8538 | 0.3187 | 0.2405 |
| Random forest | 0.7819 | 0.9406 | 0.7416 | 0.0483 |
| Initial XGBoost | 0.8227 | 0.9671 | 0.7072 | 0.0615 |
| Tuned XGBoost | **0.8257** | **0.9671** | **0.7547** | **0.0445** |

Tuned XGBoost was selected using expanding temporal cross-validation within FY2010–FY2017 and FY2018 validation PR-AUC. Its FY2018 F1-maximizing threshold was 0.7482.

## Final FY2019 test results

The frozen tuned XGBoost pipeline was evaluated once on the untouched FY2019 test set using the locked threshold selected on FY2018.

| PR-AUC | ROC-AUC | Precision | Recall | F1 | Brier score |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.7750 | 0.9460 | 0.7389 | 0.7082 | 0.7232 | 0.0518 |

The decline from FY2018 validation performance is expected for a later loan cohort and reinforces the value of chronological evaluation. See the [test metrics](reports/fy2019_test_metrics.csv) and [grouped feature importance](reports/fy2019_grouped_feature_importance.csv) tables.

## Repository structure

```text
├── data/
│   ├── raw/                 # Local SBA source data; excluded from Git
│   ├── interim/             # Optional intermediate data; excluded from Git
│   └── processed/           # Cleaned and feature datasets; excluded from Git
├── docs/                    # Data dictionary and supporting material
├── models/                  # Saved model artifacts; excluded from Git
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_baseline_models.ipynb
│   ├── 05_model_tuning.ipynb
│   └── 06_interpretation.ipynb
├── reports/                 # Final metrics, feature-importance tables, and figures
├── src/                     # Reusable pipeline code (in progress)
└── tests/                   # Automated tests (in progress)
```

## Getting started

```bash
git clone https://github.com/ArsPoghosyan/sba-loan-default.git
cd sba-loan-default

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
jupyter lab
```

After adding the raw dataset to `data/raw/`, run the notebooks in numerical order.

## Limitations

- The data contains only loans with final observed statuses. More recent loans may have had less time to resolve, which can introduce outcome-observation bias.
- The threshold was selected by maximizing FY2018 validation-set F1 because no lender-specific cost ratio or review-capacity constraint is available. In a production lending setting, the operating threshold should reflect business costs and risk appetite.
- Aggregated split importance can favor categorical feature families with many one-hot levels, so feature importance should be interpreted alongside SHAP results rather than as a causal ranking.
- Results describe this historical SBA dataset and should not be interpreted as a production credit-decision system.

## License

This project is licensed under the [MIT License](LICENSE).
