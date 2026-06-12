# Slide 6 — Feature Engineering

## New features we built

| Feature | Definition | Rationale |
|---|---|---|
| `total_income` | `ann_income + other_income` | Captures real disposable income |
| `debt_to_income` | `amount / total_income` | Standard credit metric |
| `age_check_ok` | Consistency flag from `age` vs `birth_year` vs `date` | Data quality signal |
| `risk_indicator_avg` | Mean of the three indicators | Smooths noise |
| `risk_indicator_disagreement` | Std of the three | Captures source disagreement |
| `analyst_text_length` | Number of words in `analyst_opinion` | Longer notes often signal concerns |
| `analyst_negative_kw` | Count of keywords like "warning", "concern", "weak" | Cheap NLP feature |
| `missing_score_count` | How many credit scores are missing | Missingness as signal |
| _TBD_ | | |

## Encoding choices

- Categoricals: _TBD (one-hot / target / ordinal)_
- Booleans (`vip`): cast to 0/1
- `highest_ed`: keep as ordinal (1–5)

---

### Speaker notes
Highlight the **NLP feature from `analyst_opinion`** — usually a crowd-pleaser, shows you went beyond tabular features.

### Assets to add
- `assets/figures/06_feature_correlations.png`
