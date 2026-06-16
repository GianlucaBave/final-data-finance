# Slide 11 — Future Improvements

## With more time, in priority order

1. **Probability calibration** (Platt scaling / isotonic) — current threshold is tuned on raw OOF probabilities; calibrated probabilities would also enable cost-sensitive decisions in production.
2. **SHAP per-decision explanations** — required for any deployed credit decisioning model (regulator-facing).
3. **Formal fairness audit** — disparate-impact ratio of model approvals across the protected attributes we excluded, to confirm no proxy leakage through `job_category`, ZIP-like codes, etc.
4. **Cost-sensitive training** — weight FP / FN by their real business cost (loan default amount vs lost revenue), rather than optimising symmetric accuracy.
5. **NLP on `analyst_opinion`** — currently treated as a categorical with 60 unique strings. TF-IDF or embeddings would generalise to unseen wordings.
6. **Stacking with a logistic meta-learner** — tested, slightly under the simple blend in OOF; worth revisiting with calibrated inputs.
7. **Production monitoring** — drift detection on each feature (Population Stability Index), alerting when the next legacy import looks like a new source system.

---

### Speaker notes
Keep this short and ambitious. These are not "things we did badly" — they are "the next 4 weeks of work if the bank hired us".
