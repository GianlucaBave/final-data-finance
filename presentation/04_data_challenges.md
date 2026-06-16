# Slide 4 — Data Challenges (Bottlenecks)

> The data dictionary warns us upfront: *"You should expect some level of inconsistency of formats, units and categorical encodings, as well as missing values and sparse fields."* — and *"Do not assume that every column is equally reliable."*

We treated this as a forensics problem. The most important tool we built was the **opinion-probe integrity check**: `analyst_opinion` is 60 fixed template sentences whose approval rates we know, and the *same* 60 appear in test. So for any suspect column we compare its correlation with the opinion tier in **train vs test** — a genuine feature must behave the same in both; a sabotaged one won't.

## Issues found (verified, not assumed)

| # | Column(s) | Issue | Evidence | Action |
|---|---|---|---|---|
| **C1** | `internal_code` | **Planted leak.** +0.92 correlation with target in train; a single rule `internal_code > 50` gets **99.1%** train accuracy. | Opinion-probe: corr with opinion tier = **+0.46 in train, ≈ −0.02 in test**. Marginal distribution *identical* train↔test (so a naïve distribution check misses it) — it is **scrambled** in test. | **DROP** |
| **C2** | `external_pd_score` | Present in train, **gone in test** | 5.2% missing in train vs **100% missing in test** | **DROP** |
| **C3** | `ann_income`, `other_income` | **Mixed units** — half the rows in thousands | Log-histogram is bimodal with an **empty gap between ~700 and ~2,000**; 49.2% of rows sit below 700 | Rescale ×1000 below 700 |
| **C4** | `date` | Two formats | ISO `2022-06-08` (12,707 rows) vs `Apr-2026` (12,293 rows), ~50/50 | Parse both |
| **C5** | `prev_default` | Inconsistent encodings | `{0, No, 1, Yes, blank}` — and 47.8% missing | Map → {0, 1, NaN}; 54% approval w/o default vs ~7% with |
| **C6** | `risk_indicator_1` / `2` | **Duplicate columns** | Correlation = **1.0** on the 1,007 rows where both exist; otherwise mutually exclusive | Coalesce into one |
| **C7** | `risk_indicator_1/2/3` | **Inverted-U**, not linear | Mid-range (~50) ≈ 0.63–0.65 approval; top decile (>78) ≈ 0.26; linear corr only −0.08 | Add shape features (Slide 6) |
| **C8** | `cr_scores_fico` / `vantage` / `schufa` | Mostly missing, **3 different scales** | 90.1% / 95.1% / 97.0% missing; largely mutually exclusive (one bureau per applicant) | z-score per bureau, coalesce |
| **C9** | `analyst_opinion` | Free text + **possible prompt injection** | 60 unique templates, same set in train & test; scanned all 60 → **no injection found** | Use as categorical (strongest legit feature) |
| **C10** | `religion`, `race` | Protected attributes | Synthetic codes; **zero correlation with real risk** but shift approval rates | **EXCLUDE** (Slide 7) |

## Headline figures

- **Target balance:** 51.98% approved / 48.02% rejected
- **Missingness leaders:** `cr_scores_schufa` 97%, `vantage` 95%, `fico` 90%, `prev_default` 48%, `risk_indicator_1/2` ~49% each, `age`/`birth_year` ~10%
- **The two columns that would have wrecked us if trusted:** `internal_code` (leak) and `external_pd_score` (absent in test) — together they have the two highest single-column correlations with the target (+0.92, −0.57)

---

### Speaker notes
This is the slide that wins the project. Lead with: *"The teacher told us he'd try to fool us. He did — at least four times. Here's how we caught each one."* The star is **C1**: explain the opinion-probe in one sentence — *"a real risk signal has to relate to the analysts' own opinions the same way in train and test; `internal_code` did in train but was random in test, so it had been scrambled."* Note that anyone who used `internal_code` would see 99% in cross-validation and then collapse to ~55% on the leaderboard.

### Assets to add
- `assets/figures/04_internal_code_leak.png` — corr-with-opinion-tier, train vs test (the smoking gun)
- `assets/figures/04_income_bimodal_hist.png` — the empty unit gap
- `assets/figures/04_missing_values_bar.png` — missingness train vs test (highlight `external_pd_score`)
- `assets/figures/04_risk_inverted_u.png` — approval rate by risk decile
- `assets/figures/04_target_balance.png`

### Source
`src/audit.py` … `audit6.py` (forensic trail) and notebook §2–§3.
