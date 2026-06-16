# Slide 2 — Problem & Business Context

## The task

A bank receives personal loan applications. For each one, an analyst decides: **approve** (1) or **reject** (0). We learn that historical decision from **25,000 past cases** and predict it for **5,000 new ones**.

**Type:** binary classification.

## Why it matters

| Wrong decision | Business cost |
|---|---|
| Approving a bad loan (FP) | Default, charge-off, write-down |
| Rejecting a good loan (FN) | Lost revenue, customer churn, brand cost |

The cost is **asymmetric in practice**, but Kaggle evaluates on **accuracy** — a deliberate simplification. We optimise for accuracy while keeping the cost trade-off visible (confusion matrix on Slide 9).

## Why this dataset is special

The professor explicitly tells us in the brief:

> *"You should expect some level of inconsistency of formats... Do not assume that every column is equally reliable or equally appropriate for modelling."*

This is not an ordinary cleaning exercise — **the dataset hides traps planted on purpose**. The next slides walk through the traps we found, defused, or refused.

---

### Speaker notes
End on the hook: *"This isn't just a credit-scoring problem. It's a test of how carefully you read the data."*
