# Presentation — Working Folder

This folder collects the content for the final presentation **as the model develops**. Each markdown file maps to one slide (or slide section). When content is stable we build the actual slide deck.

## Slide Map

| # | File | Slide | Status |
|---|---|---|---|
| 1 | [`01_cover.md`](01_cover.md) | Cover & Team | 🟢 Final (needs partner name) |
| 2 | [`02_problem_context.md`](02_problem_context.md) | Problem & Business Context | 🟢 Final |
| 3 | [`03_data_overview.md`](03_data_overview.md) | The Data | 🟢 Final |
| 4 | [`04_data_challenges.md`](04_data_challenges.md) | Reading the Data Dictionary as a Riddle | 🟢 Final |
| 5 | [`05_cleaning_preprocessing.md`](05_cleaning_preprocessing.md) | Defusing the Three Traps | 🟢 Final |
| 6 | [`06_feature_engineering.md`](06_feature_engineering.md) | Feature Engineering | 🟢 Final |
| 7 | [`07_ethical_considerations.md`](07_ethical_considerations.md) | Ethical Considerations | 🟢 Final |
| 8 | [`08_modeling_approach.md`](08_modeling_approach.md) | Modeling Approach | 🟢 Final |
| 9 | [`09_results.md`](09_results.md) | Results | 🟢 Final |
| 10 | [`10_bottlenecks_lessons.md`](10_bottlenecks_lessons.md) | Bottlenecks & Lessons Learned | 🟢 Final |
| 11 | [`11_future_improvements.md`](11_future_improvements.md) | Future Improvements | 🟢 Final |
| 12 | [`12_qa.md`](12_qa.md) | Q&A / Thank You | 🟢 Final |

Legend: ⚪ Empty · 🟡 Draft · 🟢 Final

## Narrative arc — "The Professor's Traps"

The deck follows a detective-story arc rather than the usual EDA → cleaning → model → results template:

1. **Setup** (1–3): the bank problem; data + base rate; out-of-time test
2. **The riddle** (4): six hints in the data dictionary, treated as adversarial input
3. **Defusing the traps** (5): `internal_code` leak, `external_pd_score` disappearance, `race`/`religion` ethics
4. **Cleaning + features** (5–6): mess parsed; domain-aware features
5. **Ethics** (7): +0.7 pt refused, quantified
6. **Model + validation** (8): progression NB → LR → RF → LGBM → ensemble; rolling next-month
7. **Results** (9): OOF 0.8590, AUC 0.9377; confusion matrix; importances
8. **Reflections** (10–11): public-LB noise floor, what failed, what's next
9. **Q&A** (12)

## Asset Folders

- `assets/figures/` — plots and charts exported from notebooks
- `assets/tables/` — clean tables (CSV / markdown)

## Workflow

1. As results come in, drop them into the relevant slide file.
2. Export plots from notebooks into `assets/figures/` with a descriptive name (e.g. `05_internal_code_corr_train_vs_test.png`).
3. Once content is stable, convert markdown into the final deck (PowerPoint / Google Slides / Keynote).
