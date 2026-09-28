# DECISIONS.md

A record of every judgement call made while building PocketSmart AI, and the reason for each one.
Entries are appended phase by phase, in the order the decisions were made.

## Phase 0: environment and repo

- **`.gitignore` was the first commit, on its own.** No other file existed in git history before it, so no secret or artefact could slip in earlier.
- **I used the existing `.venv` (Python 3.11.9) instead of conda.** conda isn't installed on the build machine, and a venv already existed on the exact pinned Python version. Deployment is unaffected, since the container is `python:3.11-slim` either way.
- **I used `google-genai` instead of `google-generativeai`.** Google deprecated `google-generativeai` in favour of `google-genai`. Starting a new project on a deprecated SDK days before submission is an avoidable risk. `requirements.txt` already listed `google-genai`.
- **The Gemini model default is `gemini-3.5-flash`, and it's overridable with `GEMINI_MODEL`.** I didn't guess the string from memory. It's the newest GA (non-preview) Flash name registered in the installed SDK's own model table (`google/genai/_local_tokenizer_loader.py`, google-genai 2.25.0), and the SDK's own tests use it too. `models.list()` couldn't be called because there's no API key on the build machine. Run `python -m src.gemini --list-models` with a key to confirm. If the model name is ever wrong, `/advice` falls back to the template, so the endpoint can't fail.
- **Every dependency is pinned with `==`.** The Docker build retrains the models, and `src/verify_build.py` requires that they reproduce the committed metrics. That only holds if the library versions are identical.
- **`statsmodels` was removed, and `scipy` is the only addition.** SARIMAX is out of scope. McNemar, the bootstrap and Fleiss' kappa are implemented in `src/stats.py` with numpy + scipy.
- **Build-machine note:** Windows Smart App Control blocked some pandas DLLs on their first import (the reputation check timed out). A retry succeeded. This is a local Windows issue only. The Linux container is unaffected.
- **Repo-local git identity** is `Mukilkani23`. There was no global git identity on this machine.

## Phase 1: synthetic data generator

- **Date span: 2024-09-01 → 2026-08-31, which gives 24 months.** "Today" for the app is the last date in the data, not the wall clock. This keeps the demo stable, and a month summary never shows up empty.
- **I added a `direction` column (debit/credit).** Salary is a *credit* labelled Transfers. Spend totals, forecasts and anomaly detection use debits only; counting income as spending would be nonsense. The classifier trains on all rows, because it only sees the string.
- **The label must not be recoverable from the string format.** Every category is paid mostly through the same UPI formats. Format-specific cues exist only where they're realistic (ATM withdrawals, ACH autopay for bills).
- **There are three sources of real ambiguity, and all three are tuning knobs in `data/generate.py`:**
  1. *Shared platforms.* Amazon Pay, Paytm, PhonePe, GPay, Flipkart, Swiggy, Zomato, Tata and Reliance each appear under several categories. A Swiggy row can be a meal (Food) or an Instamart order (Groceries).
  2. *Payments to people.* `PERSON_SHARE` sends a fraction of each category to an individual, drawn from **one shared name pool**: an auto driver (Transport), a kirana owner (Groceries), a tuition teacher (Education), the landlord (Bills). From the text alone these are indistinguishable, so they cap the achievable accuracy. That's also true of real bank data.
  3. *Surface noise.* Legal entity names (Swiggy = BUNDL TECHNOLOGIES, Ola = ANI TECHNOLOGIES), misspellings, truncation to 18–32 characters (`TRUNCATE_P`), random casing, and trailing junk.
- **Rent goes to a person (`SELVARAJ M`) via UPI/IMPS, but is labelled Bills & Utilities.** That's how rent looks on a real statement, and it creates a genuine Bills-vs-Transfers confusion.
- **Anomalies: 2% of debit rows have their amount multiplied by U(4, 10).** They're flagged in `is_anomaly`, which is used only to score the detector and is never a model input.
- **Result: 9,622 rows, which is inside the 8,000–12,000 brief.**

## Phase 1.5: real-validation scaffold

- **Three sheets of 130 rows each (100 unique + the same 30 anchors), not one file.** The anchors have to be labelled by all three people, so each labeller needs their own sheet. `to_label.csv` is the master list. Anchors are shuffled in among the other rows so labellers can't treat them differently. Anchor IDs are `ANC01..30`, and the other IDs are `LA/LB/LC###`.
- **PRIVACY: runs of ≥4 digits are masked to `#` before a real string is written anywhere.** Real strings contain phone numbers (`UPI/98xxxxxxxx/...`), UPI refs and account numbers. `features.normalise()` applies the *same* mask to every string, synthetic included, so the model never depends on digits. That costs nothing, because reference numbers carry no category signal. Real strings, sheets and labels are gitignored and excluded from the Docker/gcloud upload.
- **The `##` header lines hold the 10 category definitions, so labellers don't invent categories.** Ingest strips lines starting with `##` by hand instead of using pandas `comment='#'`, because masked strings contain `#`.
- **Anchor 3-way ties are excluded, not guessed.** When there's no majority there is no ground truth. Each tie is logged here automatically.
- **Raw percent agreement is reported alongside Fleiss' κ.** κ is chance-corrected, so it's on a different scale from accuracy. The model-vs-human comparison uses mean pairwise agreement, which *is* on the accuracy scale.
- **If there's no real-strings file, the sheet is built from held-out synthetic strings, with a loud warning.** That lets the ingest → validate pipeline be tested end to end before the real labels exist.

## Phase 2: classifier

- **scikit-learn is pinned to 1.7.2, not 1.9.1.** Windows Smart App Control permanently blocked one compiled file in 1.9.1 (`_argkmin_classmode`). I did not change the security setting. I switched to the widely deployed 1.7.2 release, whose files passed the check. The pin covers both the local venv and the Docker image, so they stay identical.
- **Split: time-based, with the last 20% of rows by date as test** (cutoff 2026-04-05). Whole days never straddle the cutoff. A random split would let the model train on the future.
- **The TF-IDF vectorizer lives inside a sklearn `Pipeline`,** so `fit` only ever sees training strings. The test set's vocabulary and IDF values can't leak into training.
- **Generator tuning, round 1 (the gate fired BELOW the band).**
  - Round 0: LR accuracy 0.729 (below 0.78), so the generator was too noisy.
  - The change: `PERSON_SHARE` halved for every category except Transfers (e.g. Groceries 0.20 → 0.10), and `TRUNCATE_P` lowered from 0.25 to 0.20.
  - Round 1: LR accuracy **0.793** [0.774, 0.811], inside the band. **I stopped after one round.** I did not keep tuning toward a "nicer" number. The CI still crosses 0.78, and that is reported as is.
  - The model's hyperparameters were never touched.
- **McNemar ship rule, correction.** The plan said "p < 0.05 → ship the higher macro-F1". The first run exposed a contradiction: McNemar was significant *in favour of LR* (more discordant wins), yet RF had a macro-F1 higher by 0.001. McNemar tests *error rates*, so a significant result can only be read in the direction the test points. The corrected rule is: p ≥ 0.05 → LR, and p < 0.05 → the model the test favours. I made the change after seeing that run's output, which I'm stating openly. The change only makes the rule consistent with what the test measures; it would have shipped LR on that run as well.
- **Result: LR ships.** McNemar p = 0.008, with 71 vs 42 discordant wins for LR. RF has the higher macro-F1 (0.760 vs 0.746), because it does better on the small classes, but it makes more errors overall. The two models trade off: RF favours small-class recall, LR favours overall accuracy.
- **RandomForest uses `class_weight="balanced_subsample"`,** the forest's equivalent of LR's balanced weighting. Without it the comparison would be unfair.
- **Abstain threshold 0.75:** the lowest threshold whose accuracy on covered rows is ≥ 0.95. It's stored in the model bundle, not written into source code.

## Phase 3: anomaly detection

- **The window is the previous 28 days (`closed="left"`), so the current day is excluded.** Otherwise a large spike raises its own mean and standard deviation and partly hides itself.
- **`MIN_HISTORY = 5`.** A standard deviation from fewer than 5 points is meaningless, so those rows get no z-score and aren't evaluated. The first 28 days are warm-up and are also excluded.
- **My hypothesis was wrong, and I'm reporting it as is.** I expected log-z to beat raw-z, because amounts are lognormal. The data says raw-z does better: AUC-PR 0.386 vs 0.335. The reason: the injected anomalies multiply the amount by 4–10×. In log space that's a shift of only 1.4–2.3, which is just 1.5–3.8 σ in high-variance categories, while on the raw scale the jump is enormous. **The served detector follows the pre-coded rule (highest AUC-PR): raw-z.**
- **The served threshold is best-F1 from the sweep, |z| > 4.0.** It's chosen on the same data it's scored on, because there's no separate tuning split for a single scalar, so that F1 is optimistic. This is stated in `metrics.json`. F1 is still rising at 4.0, the edge of the sweep. I did *not* extend the sweep to chase a higher score.
- **Per-category precision reveals the failure modes.** Bills & Utilities (0.19) is multimodal (rent ₹14k and a mobile recharge ₹599 share one mean and SD). Transfers (0.13) is extremely heavy-tailed. A single z-score per category assumes a unimodal distribution, and that assumption breaks here.

## Phase 4: forecasting

- **Weeks run Monday to Sunday, and only complete weeks are used** (104). The partial first and last weeks would look like fake drops in spend.
- **Week-of-month is taken from the week's Thursday.** A week belongs to the month most of its days fall in (the ISO convention). It's one-hot encoded, because it's a category, not a quantity.
- **Evaluation is one-step-ahead walk-forward, with no refit on holdout data.** Every model predicts week t from actual weeks < t. That's the fair setup for the naive baselines, which also use the last actual value.
- **The 80% interval is calibrated on a separate 16-week window immediately before the holdout.** Calibrating on the holdout itself would make 80% coverage true by construction. The result was mean coverage 0.800, which is genuine here. Note the granularity: with 8 holdout weeks, per-category coverage can only move in steps of 0.125.
- **MAPE is computed only on weeks with non-zero actuals,** because MAPE is undefined when actual = 0. The count of weeks used is stored per category.
- **Serving: each category serves its holdout-MAE winner.** Where the winner is linear, it's refit on all 104 weeks for the live prediction, and the interval quantiles are reused from calibration.
- **Result.** Linear regression has the lowest *mean* MAE, but **category_mean is the per-category winner in 5 of 10 categories**. For most categories, weekly spend is close to i.i.d. noise around a mean, so a constant beats models that chase last week's noise. Miscellaneous is flagged NOT ROBUST: its winner beat the reference in only 4 of 8 weeks.
