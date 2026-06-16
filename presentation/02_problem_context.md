# Slide 2 — Problem & Business Context

## The problem

Banks receive thousands of personal loan applications. For each one, an analyst (or model) decides: **approve or reject**. Wrong decisions cost money in two directions:

- **Approving bad loans** → defaults, write-offs
- **Rejecting good loans** → lost revenue, customer churn

## Our task

Predict the historical `credit_decision` for 5,000 unseen applications:
- `1` → approved
- `0` → rejected

**Type:** binary classification. **Metric:** accuracy on the hard 0/1 label (not AUC) — so the probability→label **threshold** is part of the model, not an afterthought.

## What we are really predicting

The target is **the analyst's historical decision**, not whether the loan actually defaulted. We are learning to reproduce the bank's existing approval policy. Two consequences shaped the whole project:

- The free-text `analyst_opinion` is almost a written rationale for the label — the single strongest *legitimate* feature.
- Any bias the human analysts had (e.g. by demographic group) is baked into the target. Reproducing it would raise accuracy but would be illegal and unethical in production (see Slide 7).

## Why it matters

- Automating screening lets analysts focus on borderline / high-value cases (the brief: analysts will only re-check a random monthly subsample).
- The model also exposes which features actually drive the decision — useful and auditable for a credit team.

## The time dimension (matters for validation)

Training data spans ~4 years (**2022-01 → 2026-04**); the test set is **entirely May 2026**. So the real task is *forecasting the next month*, which is why we validate against time, not just random folds (Slide 8).

---

### Speaker notes
Stress that this mirrors a real underwriting workflow — the "historical decision" is what a human analyst made. We are learning the bank's existing risk policy from data, warts (bias) and all. The "predict next month" framing is the justification for our time-based validation and is worth saying out loud.
