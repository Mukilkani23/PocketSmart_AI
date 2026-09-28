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
