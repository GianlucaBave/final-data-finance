# Presentation — Working Folder

This folder collects the content for the final presentation **as we develop the model**. Each markdown file maps to one slide (or slide section). We fill them in incrementally; once content is stable, we build the actual slide deck.

## Slide Map

| # | File | Slide | Status |
|---|---|---|---|
| 1 | [`01_cover.md`](01_cover.md) | Cover & Team | 🟡 Draft |
| 2 | [`02_problem_context.md`](02_problem_context.md) | Problem & Business Context | 🟡 Draft |
| 3 | [`03_data_overview.md`](03_data_overview.md) | The Data | 🟡 Draft |
| 4 | [`04_data_challenges.md`](04_data_challenges.md) | Data Challenges (bottlenecks) | ⚪ Empty |
| 5 | [`05_cleaning_preprocessing.md`](05_cleaning_preprocessing.md) | Data Cleaning & Preprocessing | ⚪ Empty |
| 6 | [`06_feature_engineering.md`](06_feature_engineering.md) | Feature Engineering | ⚪ Empty |
| 7 | [`07_ethical_considerations.md`](07_ethical_considerations.md) | Ethical Considerations | 🟡 Draft |
| 8 | [`08_modeling_approach.md`](08_modeling_approach.md) | Modeling Approach | ⚪ Empty |
| 9 | [`09_results.md`](09_results.md) | Results | ⚪ Empty |
| 10 | [`10_bottlenecks_lessons.md`](10_bottlenecks_lessons.md) | Bottlenecks & Lessons Learned | ⚪ Empty |
| 11 | [`11_future_improvements.md`](11_future_improvements.md) | Future Improvements | ⚪ Empty |
| 12 | [`12_qa.md`](12_qa.md) | Q&A / Thank You | ⚪ Empty |

Legend: ⚪ Empty · 🟡 Draft · 🟢 Final

## Asset Folders

- `assets/figures/` — plots and charts exported from notebooks
- `assets/tables/` — clean tables (CSV / markdown)

## Workflow

1. As we explore data and build the model, drop notes into the relevant slide file.
2. Export key plots from notebooks into `assets/figures/` with a descriptive name (e.g. `04_missing_values_heatmap.png`).
3. Once all sections are ✅, convert markdown into the final deck (PowerPoint / Google Slides / Keynote).
