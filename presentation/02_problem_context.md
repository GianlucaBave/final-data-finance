# Slide 2 — Problem & Business Context

## The problem

Banks receive thousands of personal loan applications. For each one, an analyst (or model) decides: **approve or reject**. Wrong decisions cost money in two directions:

- **Approving bad loans** → defaults, write-offs
- **Rejecting good loans** → lost revenue, customer churn

## Our task

Predict the historical `credit_decision` for 5,000 unseen applications:
- `1` → approved
- `0` → rejected

**Type:** binary classification.

## Why it matters

- Automating screening lets analysts focus on borderline / high-value cases
- A well-calibrated model also flags which features actually drive the decision (interpretability for credit teams)

---

### Speaker notes
Stress that this mirrors a real underwriting workflow — the "historical decision" is what a human analyst made. We are essentially learning the bank's existing risk policy from data.
