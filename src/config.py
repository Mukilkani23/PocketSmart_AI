"""Single place for paths, seeds and tunable constants.

Everything that affects a reported number lives here so a viva question like
"where does the 3.0 come from?" has one answer: this file.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
CACHE_DIR = Path(os.getenv("POCKETSMART_CACHE_DIR", ROOT / "cache"))

TRANSACTIONS_CSV = DATA_DIR / "transactions.csv"
REAL_STRINGS_RAW = DATA_DIR / "real_strings_raw.txt"
REAL_VALIDATION_CSV = DATA_DIR / "real_validation.csv"
CLASSIFIER_PATH = MODELS_DIR / "classifier.joblib"
FORECAST_PATH = MODELS_DIR / "forecast.joblib"
ANOMALY_PATH = MODELS_DIR / "anomaly.joblib"
LABEL_AGREEMENT_PATH = RESULTS_DIR / "label_agreement.json"
BUILD_VERIFICATION_PATH = MODELS_DIR / "build_verification.json"
METRICS_PATH = RESULTS_DIR / "metrics.json"

SEED = 42

# The 10 categories, with the one-line definition that labellers see.
CATEGORIES: dict[str, str] = {
    "Food & Dining": "Restaurants, cafes, food delivery (Swiggy/Zomato meals), tea shops, street food.",
    "Groceries": "Supermarkets, kirana stores, vegetables, milk, quick-commerce grocery (Blinkit/Zepto/Instamart).",
    "Transport": "Cabs, autos, buses, trains (IRCTC), metro, fuel, parking, tolls/FASTag.",
    "Shopping": "Clothes, electronics, e-commerce orders (Amazon/Flipkart/Myntra), household items.",
    "Bills & Utilities": "Rent, electricity, water, mobile/broadband/DTH recharges, insurance, credit-card bills.",
    "Entertainment": "Movies, OTT subscriptions (Netflix/Hotstar/Spotify), events, gaming.",
    "Health": "Pharmacy, hospital, clinic, diagnostics/lab tests, doctor consultations.",
    "Education": "School/college fees, courses, books, coaching, exam fees.",
    "Transfers": "Money sent to/received from people or own accounts, salary credit, NEFT/IMPS to individuals.",
    "Miscellaneous": "ATM cash withdrawals, bank charges, donations, anything that fits nowhere else.",
}
CATEGORY_NAMES = list(CATEGORIES)

# ---- Data -----------------------------------------------------------------
DATA_START = "2024-09-01"
DATA_END = "2026-08-31"  # "today" for the app is the last date in the data

# ---- Classifier -----------------------------------------------------------
TEST_FRACTION = 0.20              # last 20% of rows by date
ACCURACY_BAND = (0.78, 0.92)      # >0.92 on synthetic data = the generator is leaking labels
BOOTSTRAP_RESAMPLES = 1000
MCNEMAR_ALPHA = 0.05
CALIBRATION_BINS = 10
ABSTAIN_SWEEP = [round(0.30 + 0.05 * i, 2) for i in range(13)]  # 0.30 .. 0.90
ABSTAIN_TARGET_ACC = 0.95
ABSTAIN_FALLBACK = 0.70

# ---- Anomaly --------------------------------------------------------------
ANOMALY_WINDOW_DAYS = 28
ANOMALY_Z_THRESHOLD = 3.0         # the brief's threshold; the sweep reports the others
ANOMALY_Z_SWEEP = [2.0, 2.5, 3.0, 3.5, 4.0]

# ---- Forecast -------------------------------------------------------------
FORECAST_HOLDOUT_WEEKS = 8
FORECAST_CALIBRATION_WEEKS = 16   # residual window for the 80% interval, strictly before the holdout
FORECAST_LAGS = [1, 2, 3, 4]
INTERVAL_QUANTILES = (0.10, 0.90)

# ---- Gemini ---------------------------------------------------------------
# Default model: newest GA Flash name registered in the installed google-genai 2.25.0
# (google/genai/_local_tokenizer_loader.py). Override with GEMINI_MODEL in .env.
GEMINI_MODEL_DEFAULT = "gemini-3.5-flash-lite"
GEMINI_TIMEOUT_S = 5.0            # user-facing budget, enforced with a thread timeout
GEMINI_HTTP_TIMEOUT_MS = 10_000   # SDK socket timeout; the Gemini API rejects deadlines < 10 s
PROMPT_VERSION = "v2"  # bump on any prompt change: invalidates cached narrations
# Cost constants — VERIFY BEFORE VIVA against https://ai.google.dev/pricing.
# They only feed an "estimated cost" display, never a model decision.
FLASH_USD_PER_1M_IN = 0.30
FLASH_USD_PER_1M_OUT = 2.50
USD_TO_INR = 88.0


def abstain_threshold() -> float:
    """The shipped abstain threshold, read from the trained bundle (not hard-coded)."""
    try:
        import joblib

        return float(joblib.load(CLASSIFIER_PATH)["abstain_threshold"])
    except Exception:
        return ABSTAIN_FALLBACK
