# Model card: PocketSmart merchant-string classifier (and companions)

All figures are in [`results/metrics.json`](results/metrics.json), and the README tables are rendered from it. This card describes the models; it doesn't restate numbers that could drift.

**Models.** (1) A merchant-string classifier: TF-IDF character 2–4-grams → LogisticRegression (class-balanced). A RandomForest was compared and not shipped (McNemar). (2) An anomaly detector: rolling 28-day per-category z-score (raw amount). (3) A weekly spend forecaster: per category, the lowest-holdout-MAE model out of naive, seasonal-naive, linear on lags, and category mean. (4) Gemini, which **only narrates** the JSON from 1–3 and never computes or predicts.

## Intended use
- Categorising a single user's Indian bank-statement lines into 10 spending categories, to support personal budgeting summaries.
- Flagging unusually large spends *for review* by the account holder.
- A rough next-week spend expectation per category, with an 80% interval and the model's historical error shown alongside it.
- An academic demonstration of honest ML evaluation (baselines, time splits, calibration, real-data validation).

## Out of scope. Do NOT use it for:
- Credit scoring, lending, insurance or any decision about a person. The models were never evaluated for that, and category errors are common (see below).
- Fraud detection. The detector favours precision on large-amount outliers, and it doesn't look at merchant novelty, location or device.
- Tax, accounting or legal categorisation.
- Investment or financial advice. The narration layer is explicitly instructed not to give it.
- Other countries' statement formats, or non-INR currencies.

## Training data
**Synthetic.** 24 months of transactions from `data/generate.py` (seed 42), with messy UPI/POS/NEFT/IMPS/ACH/ATM strings, shared platforms across categories, payments to individuals from a shared name pool, and injected amount outliers (`is_anomaly`, used only for scoring). No real person's data was used in training. The generator encodes the author's assumptions about Indian statements, and those assumptions are the main risk.

## Evaluation data
1. **Synthetic test set:** the last 20% of rows by date (time split, never random). Metrics come with 1,000-resample bootstrap CIs. There's a leakage gate (accuracy must be in 0.78–0.92), and one generator-tuning round was logged in DECISIONS.md.
2. **Real validation set:** about 300 real statement strings (digits masked) labelled by three people, plus 30 anchors labelled by all three, for Fleiss' κ and the human-agreement ceiling. The expected result was pre-registered in DECISIONS.md before ingest, and no change is permitted in response to the result. *Status: see `real_validation.status` in metrics.json.*

## Known failure modes (from `results/error_analysis.md` and the per-class report)
- **Shared-platform ambiguity.** Swiggy/Zomato strings that are really quick-commerce grocery orders are predicted Food & Dining with high confidence. Flipkart/Amazon grocery orders are predicted Shopping. The bank string doesn't contain the sub-service, so text alone can't resolve it.
- **Payments to individuals.** Rent, tuition, kirana and auto payments to named people look alike and default toward Transfers. Rent paid via IMPS is confidently predicted Transfers.
- **Underconfident probabilities.** Actual accuracy is well above the stated confidence (see ECE in metrics.json). Treat `confidence` as a ranking signal. The `uncertain` flag uses a threshold calibrated on observed accuracy.
- **Anomaly detector.** False positives on monthly bills: the 28-day window is shorter than a billing cycle, so each bill is compared against a month without one. Low precision in heavy-tailed categories (Transfers). It detects large amounts only, not unusual merchants.
- **Forecaster.** For most categories a constant (the category mean) is as good as or better than the regression. Intervals are empirical, from a 16-week window, and can under-cover when the holdout is more volatile than the calibration period.

## Human-agreement ceiling
Mean pairwise agreement among the three labellers on the 30 anchors is the realistic ceiling for accuracy, and Fleiss' κ is reported alongside it. Both are in `metrics.json → real_validation.human_agreement` once the labels are ingested. A model near the ceiling has run out of *text* signal. Further gains need non-text features (amount, recurrence, payee history).

## Ethical and privacy notes
Real strings are digit-masked before being written anywhere, and they're gitignored and excluded from the container. There's a single user, with no authentication and no persistence of user input beyond the in-memory latency counters and the LLM response cache. The cache stores narration text keyed by a sha256 of the payload, and lives on ephemeral disk in Cloud Run. Note: if a user-typed merchant string is included in an /advice request, the narration may repeat it.
