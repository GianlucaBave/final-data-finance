# Slide 4 — Reading the Data Dictionary as a Riddle

The professor gave us **six explicit hints** in the brief and the dictionary. We treated them as adversarial input — not as background colour.

| # | What the professor wrote | What we read between the lines | Where we used it |
|---|---|---|---|
| H1 | *"Multiple legacy source systems"* | Expect mixed formats (dates, encodings, scales) | Slide 5 — cleaning |
| H2 | *"Inconsistencies of formats, units, encodings"* | Currency / units may differ row-to-row | Slide 5 — `ann_income` is bimodal |
| H3 | *"Religion/race — legal and regulatory limitations"* | Ethical trap. They are predictive on paper. | Slide 7 — refused (+0.7pt cost) |
| H4 | **"Do not assume every column is equally reliable"** | Some column **must not be trusted** — start hunting | Slide 5 — `internal_code` is sabotaged |
| H5 | *"Risk indicators — please check consistency"* | They are not three independent signals | Slide 5 — r1/r2 mutually exclusive → coalesce |
| H6 | *"analyst_opinion — may contain useful information"* | Free text is signal, not noise | Slide 6 — top feature in the model |

## Headline data findings

- **Target balance:** 52% approved / 48% rejected → naive majority baseline = 0.52
- **Missingness:** wide range — from 0% (loan amount) to **100% in test** (`external_pd_score`)
- **Date formats:** `2024-11-29` (ISO) and `Nov-2025` (Mon-YYYY), roughly 50/50

---

### Speaker notes
Frame this slide as the *contract* with the rest of the deck — every hint H1–H6 will be addressed.

### Assets to add
- `assets/figures/04_missing_values_heatmap.png`
- `assets/figures/04_target_balance.png`
