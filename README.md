# ESADE DSF 26 — Personal Loan Credit Decision

Kaggle competition: [esade-dsf-26-personal-loan-credit-decision](https://www.kaggle.com/competitions/esade-dsf-26-personal-loan-credit-decision)

**Task.** Binary classification: predict `credit_decision` (1 = approved, 0 = rejected) for personal loan applications.

## Data

- `train.csv` — 25,000 rows, target included
- `test.csv` — 5,000 rows
- `sample_submission.csv` — `id, credit_decision`
- `data_dictionary.csv` — column descriptions (kept in repo, no PII)

> Train/test/submission CSVs are **gitignored** to respect Kaggle's redistribution rules. Download them via the Kaggle API or from the competition page and place them under `data/`.

### Download

```bash
pip install kagglehub
python -c "import kagglehub; print(kagglehub.competition_download('esade-dsf-26-personal-loan-credit-decision'))"
```

Or manually download the zip from Kaggle and extract into `data/`.

## Layout

```
data/             # raw CSVs (gitignored except data_dictionary.csv)
notebooks/        # EDA, baseline, modeling
src/              # reusable code (preprocessing, features, models)
submissions/      # generated submission files (gitignored)
```

## Environment

Python 3.11 + the `MLbp2-env` conda env from the course (`../MLbp2-env.yml`).

## Notes on the data

- `date` arrives in multiple formats (`YYYY-MM-DD`, `Mon-YYYY`) → needs normalization.
- `prev_default` has legacy encodings (e.g. `Yes`, `0`, missing).
- `ann_income` / `other_income` are string-like numerics — parse carefully.
- `religion` and `race` are synthetic codes; per the data dictionary they're flagged as fairness-sensitive. **Exclude from the model.**
- `analyst_opinion` is free text — potential for simple NLP features (length, sentiment, keyword flags).
- Three credit scores (`fico`, `vantage`, `schufa`) and `external_pd_score` are mostly missing → impute / use missingness indicators.
- Three `risk_indicator_*` columns — data dictionary hints at consistency checks worth exploring.
