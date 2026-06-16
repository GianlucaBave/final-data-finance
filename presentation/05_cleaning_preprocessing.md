# Slide 5 — Defusing the Three Traps

The professor planted three deliberate traps. Using each one as-is would either inflate cross-validation or fail in production. We found, measured, and defused all three.

## TRAP #1 — `internal_code` is a planted leak

- Correlation with target in train: **+0.92**
- Simple rule `internal_code > 50` → **99% accuracy in train**
- Looks too good to be true. It is.

**Detection — the opinion-probe.** We computed the per-`analyst_opinion` approval rate from train as a proxy for the *true* underlying signal, then correlated `internal_code` with that signal **separately** in train and test:

| Feature | corr in train | corr in test |
|---|---|---|
| `internal_code` | **+0.46** | **≈ 0.00** |
| (everything else) | small | similar |

Train says `internal_code` is highly informative; test says it is pure noise. **The values were scrambled in the test set.** A model that leans on it would score 99% on the public-cheat and ~52% on the private 60%.

**Action:** dropped.

## TRAP #2 — `external_pd_score` disappears in test

- Train missingness: 5% → looks usable
- **Test missingness: 100%** → useless at inference time

**Detection:** `train.isna().mean()` vs `test.isna().mean()` → wide gap, immediately visible.

**Action:** dropped.

## TRAP #3 — `race`, `religion` are an ethical trap

- Synthetic codes, included in the file
- Empirically worth **+0.7 pt** accuracy
- Forbidden in real credit scoring (GDPR, ECOA)

**Action:** dropped on principle. See Slide 7.

## Plus: messy legacy data

| Column | Mess | Fix |
|---|---|---|
| `date` | `Apr-2026` and `2022-06-08` mixed | Parse both, mid-month for `Mon-YYYY` |
| `ann_income` | Bimodal log10 with empty gap 700–2000 | `* 1000` if value < 700 |
| `other_income` | Same, plus zero/missing | Same rule, plus `_missing` flag |
| `prev_default` | `{Yes, 1, '1'}`, `{No, 0, '0'}`, blank | Map to {0, 1, NaN}, add `_missing` flag |
| `risk_indicator_1/2` | Mutually exclusive | Coalesce into `risk12` |
| Credit scores | 3 bureaus, different scales | Z-score per bureau **on train only**, then coalesce |

---

### Speaker notes
TRAP #1 is the climax — spend the most time on it. The opinion-probe trick is what separates this project from a tutorial-grade submission.

### Assets to add
- `assets/figures/05_internal_code_corr_train_vs_test.png`
- `assets/figures/05_income_bimodal_histogram.png`
- `assets/figures/05_missing_train_vs_test.png`
