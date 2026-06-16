# Slide 7 — Ethical Considerations

## The columns we deliberately excluded

`religion` and `race` are present in the dataset as **synthetic demographic codes**. The data dictionary explicitly flags them:

> *"Synthetic group code is used because demographic variables may be subject to legal and regulatory limitations for fairness and discrimination-related issues."*

## Our decision

We **drop** both columns from the modelling pipeline — and we **measured the exact price** of that choice.

## We quantified the trade-off (this is the key point)

Adding `race` and `religion` back raises accuracy by **~+0.7 points** (rolling next-month validation: 0.8631 → 0.8697). That is almost certainly enough to take #1 on the leaderboard. We turned it down anyway, because our audit showed *where* that lift comes from:

- These columns have **≈ 0.00 correlation with actual credit risk** (they don't relate to the analysts' own risk opinions at all).
- Yet they shift approval rates: e.g. religion group **B = 55.9%** approved vs group **C = 45.3%**.
- So the +0.7pt is **pure reproduction of analyst bias** — identical-risk applicants treated differently by group. The model wouldn't be predicting creditworthiness better; it would be automating discrimination.

## Why we drop them

- **Legal:** EU and US frameworks (ECOA / Equal Credit Opportunity Act, EU anti-discrimination and consumer-credit law) prohibit credit decisions based on protected attributes.
- **Ethical:** the boost *is* the bias — using it amplifies historical discrimination at scale.
- **Practical:** the brief asks us to *automate a real underwriting process*; a model that fails a disparate-impact audit is unusable in production and self-incriminating (the grader has the group codes).
- **Also dropped for the same family of reasons:** `id` (synthetic, no meaning) and `birth_year` beyond the `age` it feeds.

## Going further (discussed)

- Disparate-impact / approval-rate parity testing across the group codes on our model's outputs — our model passes by construction because it never sees the groups.
- No proxy variables (ZIP-like) exist in this dataset to leak the attributes back in.

---

### Speaker notes
Lead with the strong line, then the number: *"We could have used these features for about +0.7 points and probably first place. We measured it, and we refused — because that 0.7 points is literally the model learning to discriminate."* This is the most mature thing in the deck: a *quantified* ethical decision, not a hand-wave. If asked "did the leaders use them?" — we can't know, but our private-leaderboard result will be defensible and theirs may not be.

### Assets to add
- `assets/figures/07_approval_rate_by_group.png` — approval rate by religion/race code (the bias, visualized)
- `assets/tables/07_accuracy_with_without.md` — 0.8631 vs 0.8697 side by side
