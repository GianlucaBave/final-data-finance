# Slide 7 — Ethical Considerations

## The columns we deliberately excluded

`religion` and `race` are present in the dataset as **synthetic demographic codes**. The data dictionary explicitly flags them:

> *"Synthetic group code is used because demographic variables may be subject to legal and regulatory limitations for fairness and discrimination-related issues."*

## Our decision

We **drop** both columns from the modelling pipeline.

## Why

- **Legal:** EU and US frameworks (GDPR, ECOA, Equal Credit Opportunity Act) prohibit credit decisions based on protected attributes.
- **Ethical:** Even if predictive, using these features would encode and amplify historical bias.
- **Practical:** A model that depends on protected attributes is unusable in production.

## Going further (mentioned, not implemented)

- Fairness audits on proxies (e.g. ZIP-code-like features) — none of those here
- Disparate-impact testing on model outputs

---

### Speaker notes
Lead with a strong line: *"We could have used these features. We chose not to — and we'd argue you shouldn't either."* Makes the project look mature.
