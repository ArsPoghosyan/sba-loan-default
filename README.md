# SBA 7(a) Loan Charge-Off Prediction

A binary-classification project that predicts whether an SBA 7(a) loan will be charged off, using SBA FOIA loan data for fiscal years 2010–2019.

The project is being built as an end-to-end, reproducible machine-learning workflow: exploratory analysis, leakage-aware cleaning, feature engineering, chronological validation, model comparison, and final interpretation.

## Project status

- [x] Exploratory data analysis
- [x] Data cleaning and processed dataset creation
- [x] Feature engineering and chronological train/validation/test split
- [x] Dummy-classifier and logistic-regression baselines
- [ ] Random forest and XGBoost comparison
- [ ] Model tuning and validation-threshold selection
- [ ] Final FY2019 evaluation, calibration, and interpretation
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

## Current baseline results

Models below were trained on FY2010–FY2017 and evaluated on FY2018. PR-AUC is the primary metric because charge-offs are the minority class.

| Model | PR-AUC | ROC-AUC | F1 at 0.50 | Brier score |
| --- | ---: | ---: | ---: | ---: |
| Dummy classifier | 0.1013 | 0.5000 | 0.0000 | 0.1013 |
| Logistic regression | **0.5864** | **0.8777** | **0.5354** | **0.0616** |
| Logistic regression, balanced classes | 0.4900 | 0.8538 | 0.3187 | 0.2405 |

Unweighted logistic regression is the current benchmark. The class-weighted variant achieves higher recall at the default threshold, but has lower PR-AUC and substantially poorer probability calibration. These are validation results only; FY2019 remains untouched.

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
├── reports/figures/         # Exported figures; excluded from Git
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
- The selected threshold will maximize validation-set F1 because no lender-specific cost ratio or review-capacity constraint is available. In a production lending setting, the operating threshold should reflect business costs and risk appetite.
- Results describe this historical SBA dataset and should not be interpreted as a production credit-decision system.

## License

This project is licensed under the [MIT License](LICENSE).
