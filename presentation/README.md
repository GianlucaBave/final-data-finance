# Presentation — Working Folder

This folder collects the content for the final presentation **as we develop the model**. Each markdown file maps to one slide (or slide section). We fill them in incrementally; once content is stable, we build the actual slide deck.

## Slide Map

| # | File | Slide | Status |
|---|---|---|---|
| 1 | [`01_cover.md`](01_cover.md) | Cover & Team | 🟢 Content ready |
| 2 | [`02_problem_context.md`](02_problem_context.md) | Problem & Business Context | 🟢 Content ready |
| 3 | [`03_data_overview.md`](03_data_overview.md) | The Data | 🟢 Content ready |
| 4 | [`04_data_challenges.md`](04_data_challenges.md) | Data Challenges (bottlenecks) | 🟢 Content ready |
| 5 | [`05_cleaning_preprocessing.md`](05_cleaning_preprocessing.md) | Data Cleaning & Preprocessing | 🟢 Content ready |
| 6 | [`06_feature_engineering.md`](06_feature_engineering.md) | Feature Engineering | 🟢 Content ready |
| 7 | [`07_ethical_considerations.md`](07_ethical_considerations.md) | Ethical Considerations | 🟢 Content ready |
| 8 | [`08_modeling_approach.md`](08_modeling_approach.md) | Modeling Approach | 🟢 Content ready |
| 9 | [`09_results.md`](09_results.md) | Results | 🟢 Content ready |
| 10 | [`10_bottlenecks_lessons.md`](10_bottlenecks_lessons.md) | Bottlenecks & Lessons Learned | 🟢 Content ready |
| 11 | [`11_future_improvements.md`](11_future_improvements.md) | Future Improvements | 🟢 Content ready |
| 12 | [`12_qa.md`](12_qa.md) | Q&A / Thank You | 🟢 Content ready |

Legend: ⚪ Empty · 🟡 Draft · 🟢 Content ready (text done; figures pending) · ✅ Final deck built

**Status (June 2026):** all slide *content* populated from the actual code & results
(`src/prep.py`, the audit scripts, `notebooks/credit_decision_pipeline.ipynb`, and the
v1→v8 submission results). Remaining work = export the figures listed under each slide's
"Assets to add" and build the deck.

## Asset Folders

- `assets/figures/` — plots and charts exported from notebooks
- `assets/tables/` — clean tables (CSV / markdown)

## Workflow

1. As we explore data and build the model, drop notes into the relevant slide file.
2. Export key plots from notebooks into `assets/figures/` with a descriptive name (e.g. `04_missing_values_heatmap.png`).
3. Once all sections are ✅, convert markdown into the final deck (PowerPoint / Google Slides / Keynote).
