# ESADE DSF 26 — Personal Loan Credit Decision

Kaggle competition: [esade-dsf-26-personal-loan-credit-decision](https://www.kaggle.com/competitions/esade-dsf-26-personal-loan-credit-decision)

**Task.** Binary classification: predict `credit_decision` (1 = approved, 0 = rejected) for personal loan applications.

## Description

The dataset has been assembled from multiple legacy source systems. You should expect some level of inconsistency of formats, units and categorical encodings, as well as missing values and sparse fields. Try to fix as much of the data as possible, and when possible.

Do not assume that every column is equally reliable or equally appropriate for modelling.

## Files

- `train.csv` — Training data with the target variable (25,000 rows)
- `test.csv` — Test data without the target variable (5,000 rows)
- `sample_submission.csv` — A sample submission file in the correct format. Please make sure that the solution you submit follows this structure.
- `data_dictionary.csv` — Description of variables

## Column Dictionary

| Column | Type | Student description | Notes |
|---|---|---|---|
| `id` | identifier | Unique synthetic identifier for each loan application. | Do not use as a meaningful applicant characteristic. |
| `date` | date / string | Date associated with the credit decision or application record. | Imported from more than one legacy source; formats may not be fully standardized. |
| `prev_default` | categorical / string | Historical indication of whether the applicant had a previous default event. | Legacy encodings are not fully standardized. |
| `age` | numeric | Applicant age at the time of the credit decision, when available. | Some records are missing. |
| `birth_year` | numeric | Applicant year of birth, when available. | |
| `religion` | categorical | Synthetic demographic group code. | Synthetic group code is used because demographic variables may be subject to legal and regulatory limitations for fairness and discrimination-related issues. |
| `race` | categorical | Synthetic demographic group code. | Synthetic group code is used because demographic variables may be subject to legal and regulatory limitations for fairness and discrimination-related issues. |
| `external_pd_score` | numeric | External provider's estimated default-risk score, when available. | |
| `cr_scores_fico` | numeric | FICO-like credit score when available. | |
| `cr_scores_vantage` | numeric | Vantage-like credit score when available. | |
| `cr_scores_schufa` | numeric | SCHUFA-like credit score when available. | |
| `highest_ed` | ordinal categorical / numeric | Highest reported education level, coded from 1 to 5. | Higher values represent higher reported education levels. |
| `job_category` | categorical | Structured employment category derived from the applicant's job profile. | Values describe broad employment stability and job profile groups. |
| `ann_income` | numeric / string-like numeric | Reported annual income. | Imported from more than one legacy source. |
| `other_income` | numeric / string-like numeric | Reported additional income beyond main annual income. | Imported from more than one legacy source. |
| `status` | categorical | Applicant civil or household status. | |
| `kids` | numeric | Number of reported dependent children. | |
| `analyst_opinion` | text | Short qualitative note written during the credit review process. | Free-text field. May contain useful underwriting information. |
| `amount` | numeric | Loan amount requested by the applicant. | |
| `vip` | boolean / categorical | Indicator of whether the applicant belongs to a special relationship segment. | It may affect the credit decision process in some cases. |
| `risk_indicator_1` | numeric | Risk indicator from an aggregator source. | Unfortunately, we don't have more information on the type of underlying data. Please check consistency between variables. |
| `risk_indicator_2` | numeric | Risk indicator from an aggregator source. | Unfortunately, we don't have more information on the type of underlying data. Please check consistency between variables. |
| `risk_indicator_3` | numeric | Risk indicator from an aggregator source. | Unfortunately, we don't have more information on the type of underlying data. Please check consistency between variables. |
| `internal_code` | numeric | Internal processing code. | |
| `credit_decision` | binary target | Historical credit decision. 1 means approved; 0 means rejected. | Available only in the training file. This is the target variable for modeling. |

## Project Layout

```
data/             # train.csv, test.csv, sample_submission.csv, data_dictionary.csv
notebooks/        # EDA, baseline, modeling
src/              # reusable code (preprocessing, features, models)
submissions/      # generated submission files
```

## Environment

Python 3.11 + the `MLbp2-env` conda env from the course (`../MLbp2-env.yml`).

```bash
conda env create -f ../MLbp2-env.yml
conda activate MLbp2-env
```

## Working Notes

- `date` arrives in multiple formats (`YYYY-MM-DD`, `Mon-YYYY`) → needs normalization.
- `prev_default` has legacy encodings (e.g. `Yes`, `0`, missing).
- `ann_income` / `other_income` may be string-like — parse carefully.
- `religion` and `race` are synthetic fairness-sensitive codes — **exclude from the model**.
- Three credit scores (`fico`, `vantage`, `schufa`) and `external_pd_score` are mostly missing → impute and/or add missingness indicators.
- `risk_indicator_*` columns — the data dictionary hints at consistency checks across the three.
- `analyst_opinion` is free text — potential for simple NLP features (length, keyword flags, sentiment).
- `age` and `birth_year` should be cross-checked against `date`.
