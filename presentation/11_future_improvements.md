# Slide 11 — Future Improvements

## Already done (so don't list as "future")

- ✅ Model blending across LightGBM + XGBoost + CatBoost
- ✅ Repeated cross-validation + pseudo-labeling
- ✅ A fairness analysis (we *quantified* the bias cost, Slide 7)

## With more time / data we would…

- **SHAP explanations per decision** — turn the model into an auditable tool the credit team can defend application-by-application.
- **Probability calibration** (isotonic / Platt) — our threshold is tuned for accuracy, but a real lender needs well-calibrated default probabilities for pricing and capital.
- **Cost-sensitive thresholds** — accuracy treats a wrong approval and a wrong rejection equally; in reality a default costs far more than a lost customer. We'd optimize expected € loss, not accuracy.
- **Formal disparate-impact monitoring** — automated approval-rate parity checks across the demographic codes as an ongoing production guardrail.
- **Concept-drift monitoring** — the rolling harness showed May 2026 is harder than the average month; a deployed model needs scheduled retraining and drift alerts.
- **Richer ensembling** — a neural-net member or a proper stacking meta-learner with more base models (our quick LR-stack lost to a blend, but a deeper stack might win).

---

### Speaker notes
Keep it short and honest — separate what we *did* from what we'd *do next*. The strongest "next step" for a finance audience is **cost-sensitive decisions + calibration**: it reframes the whole thing from "Kaggle accuracy" to "real lending economics," which is what the course is actually about.
