# Slide 7 — Ethical Considerations

## The columns we deliberately excluded

`religion` and `race` are present as synthetic demographic codes. The brief flags them explicitly:

> *"Synthetic group code is used because demographic variables may be subject to legal and regulatory limitations for fairness and discrimination-related issues."*

## The trade-off, quantified

| Setup | OOF Accuracy |
|---|---|
| Our model — no `race` / `religion` | **0.859** |
| Same model + `race` + `religion` | ~0.866 (+0.7 pt) |

**The +0.7 pt is real. We refused it anyway.**

## Why

| Reason | Detail |
|---|---|
| **Legal** | GDPR (EU), ECOA (US), Equal Credit Opportunity Act — protected-attribute use is unlawful in credit decisions |
| **Ethical** | Encoding historical bias into the model amplifies it |
| **Practical** | A model that needs protected attributes cannot be deployed |
| **Methodological** | The professor flagged these columns as a test — using them would fail the implicit ethics question |

## Going further (in scope for a follow-up)

- Fairness audit on *proxies*: do current features (job_category, ZIP-like codes, etc.) act as race/religion proxies?
- Disparate-impact ratio on model approval rates
- Counterfactual fairness tests

---

### Speaker notes
Strong opening line: *"We could have used these features. We measured the gain. We chose not to."* That's a Master-level answer to an ethics question.
