# Slide 10 — Bottlenecks & Lessons Learned

## What was hardest

1. **Detecting the `internal_code` leak.** A 99%-accuracy rule is a red flag, not a result. The "opinion-probe" trick (correlate the suspect feature with the per-`analyst_opinion` approval rate in train **and** in test) is what made the leak visible.
2. **The +0.7 pt ethics decision.** Real accuracy left on the table because the model has to be deployable.
3. **Date normalisation.** Two formats (`Apr-2026`, `2022-06-08`) at ~50/50 → no library catches both, had to chain both parsers.
4. **Resisting public-LB chasing.** Our last three submissions land at 0.8535 / 0.8540 / 0.8565 — within ±5 rows on 2,000. The public board can't distinguish them; we stopped submitting.

## What we tried that didn't work

| Experiment | Effect | Verdict |
|---|---|---|
| Recency-weighted training (half-life 1.5y, 3y) | −0.3 pt on rolling | Hurts |
| Dropping time features | −0.3 pt | Hurts |
| Provenance flags (which legacy system) | ~0 pt | Wash |
| 7-config CatBoost hyper-search | all within 0.2 pt | Diminishing returns |
| Stacked logistic regression on OOF | < blend | Blend wins |

## Lessons learned

- **Read the data dictionary as adversarial input.** Every "minor" note is a clue.
- **Always check `train.isna()` vs `test.isna()`** — `external_pd_score` was caught in one line.
- **A single suspiciously strong feature is a leak hypothesis**, not a win.
- **Clean methodology > +0.005 on public LB.** The private 60% rewards generalisation, not p-hacking.

---

### Speaker notes
This slide proves we *understood* the problem rather than just produced numbers. Failures are part of the story.
