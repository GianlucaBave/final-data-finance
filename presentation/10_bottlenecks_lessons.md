# Slide 10 — Bottlenecks & Lessons Learned

## What was hardest

1. **Forensics, not modeling.** The bulk of the value came *before* any model — catching the `internal_code` leak. It looks like the best feature in the dataset (+0.92 corr, 99% train accuracy) and would have silently capped us at ~0.55 on the leaderboard. Catching it required inventing the **opinion-probe** (compare a column's behavior in train vs test).
2. **Telling fake signal from real signal.** `internal_code` (leak) and `external_pd_score` (absent in test) are the two highest-correlation columns — both useless. Meanwhile the risk indicators looked useless (linear corr −0.08) but hid an inverted-U.
3. **Reading a leaderboard that can't be read.** Public = 2,000 rows; one row = 0.0005. Our last three submissions differ by ~20 visible rows — pure noise (±2–3 rows). We had to stop trusting it.

## What we tried that didn't work

- **Recency-weighted training** (down-weight old rows) — *hurt* (0.8600 vs 0.8631). The 4-year history is informative; recency bias threw it away.
- **Dropping time features** — hurt.
- **Provenance flags** (which legacy system a row came from) — a wash (0.8600 vs 0.8604). The format mess was a cleaning test, not a feature.
- **Logistic-regression stacking** of the three models — lost to a simple weighted blend (0.8606 vs 0.8615).
- **Looser decision thresholds** (v3, v7) — measurably worse on the public board.

## Lessons learned

- **Read the data dictionary first.** Every trap was foreshadowed ("not equally reliable", "check consistency", "legal limitations").
- **Validate against the real task.** A random split would have rewarded the leak and the loose threshold; the rolling next-month harness exposed both.
- **The public leaderboard is a trap too.** Chasing it = overfitting the 40% that doesn't count. We selected finals by validation.
- **Sometimes the best feature engineering is deletion.** We removed the 4 most "predictive" columns on purpose.
- **Quantify ethical choices.** "We dropped race/religion" is weak; "dropping them costs exactly +0.7pt of bias-driven accuracy, and here's the proof" is strong.

---

### Speaker notes
This slide proves we understood the problem, not just ran sklearn. Don't hide the failures — recency weighting and stacking *not* working shows we actually tested, and the provenance idea shows we thought about the data's origin. The one-liner: *"The hardest part of this competition wasn't building the model — it was figuring out which 4 columns were lying to us."*
