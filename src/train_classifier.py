"""Merchant-string -> category classifier.

Run:  python -m src.train_classifier      (also called by src.evaluate)

- TIME-BASED split: the last 20% of rows by date are the test set. A random split
  would let the model see e.g. March 2026 strings while being "tested" on February
  2026, which is future leakage. It also hides drift (new merchants, new formats).
- TF-IDF is fit INSIDE a Pipeline on train only. The test set's vocabulary and IDF
  weights never influence training.
- LogisticRegression(class_weight='balanced') is primary; RandomForest is the
  comparison. Selection rule, fixed in advance: exact McNemar test on the test rows;
  if p >= 0.05 the models are statistically indistinguishable and LR ships (simpler,
  calibrated-ish probabilities, readable coefficients). Only if p < 0.05 does the
  higher macro-F1 win.
"""
from __future__ import annotations

import hashlib
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

from src import config
from src.features import make_vectorizer, normalise
from src.stats import bootstrap_ci, mcnemar_exact, reliability

LEAKAGE_EXIT = 2
BELOW_BAND_EXIT = 3


def time_split(df: pd.DataFrame):
    """Last TEST_FRACTION of rows by date = test. Whole days never straddle the split."""
    df = df.sort_values(["date", "txn_id"], kind="mergesort").reset_index(drop=True)
    cutoff = df["date"].iloc[int(len(df) * (1 - config.TEST_FRACTION))]
    return df[df["date"] < cutoff], df[df["date"] >= cutoff], cutoff


def build_models() -> dict[str, Pipeline]:
    return {
        "LogisticRegression": Pipeline([
            ("tfidf", make_vectorizer()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=3000, random_state=config.SEED)),
        ]),
        "RandomForest": Pipeline([
            ("tfidf", make_vectorizer()),
            ("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample",
                                           random_state=config.SEED, n_jobs=-1)),
        ]),
    }


def _macro_f1(y, p):
    return f1_score(y, p, average="macro", zero_division=0)


def train_and_evaluate(df: pd.DataFrame, n_boot: int = config.BOOTSTRAP_RESAMPLES) -> dict:
    train, test, cutoff = time_split(df)
    X_tr, y_tr, X_te, y_te = train["merchant_raw"], train["category"], test["merchant_raw"], test["category"].to_numpy()
    labels = config.CATEGORY_NAMES

    fitted, preds, results = {}, {}, {}
    for name, pipe in build_models().items():
        pipe.fit(X_tr, y_tr)
        p = pipe.predict(X_te)
        fitted[name], preds[name] = pipe, p
        acc, mf1 = accuracy_score(y_te, p), _macro_f1(y_te, p)
        results[name] = {
            "accuracy": round(acc, 4),
            "macro_f1": round(mf1, 4),
            "accuracy_ci95": [round(x, 4) for x in bootstrap_ci(y_te, p, accuracy_score, n_boot, config.SEED)],
            "macro_f1_ci95": [round(x, 4) for x in bootstrap_ci(y_te, p, _macro_f1, n_boot, config.SEED)],
        }

    # --- model selection: McNemar, rule fixed in advance ---------------------
    mc = mcnemar_exact(y_te, preds["LogisticRegression"], preds["RandomForest"])
    better_f1 = max(results, key=lambda k: results[k]["macro_f1"])
    if mc["p_value"] >= config.MCNEMAR_ALPHA:
        shipped = "LogisticRegression"
        reason = (f"McNemar p={mc['p_value']:.4f} >= {config.MCNEMAR_ALPHA}: no significant difference, "
                  f"so ship the simpler LogisticRegression (higher macro-F1 was {better_f1}).")
    else:
        # McNemar tests ERROR RATES, so when it is significant the winner is the model it
        # favours (more discordant wins), even if macro-F1 differs by noise in the other direction.
        t = mc["table"]
        shipped = "LogisticRegression" if t["a_correct_b_wrong"] > t["a_wrong_b_correct"] else "RandomForest"
        reason = (f"McNemar p={mc['p_value']:.4g} < {config.MCNEMAR_ALPHA}: significant difference in error rate "
                  f"in favour of {shipped} ({max(t['a_correct_b_wrong'], t['a_wrong_b_correct'])} vs "
                  f"{min(t['a_correct_b_wrong'], t['a_wrong_b_correct'])} discordant wins); higher macro-F1 was {better_f1}.")
    mc.update({"model_a": "LogisticRegression", "model_b": "RandomForest", "shipped": shipped, "reason": reason})

    model, p_ship = fitted[shipped], preds[shipped]
    report = classification_report(y_te, p_ship, labels=labels, output_dict=True, zero_division=0)
    per_class = {c: {k: round(report[c][k], 4) for k in ("precision", "recall", "f1-score", "support")} for c in labels}
    cm = confusion_matrix(y_te, p_ship, labels=labels)

    # --- calibration of the shipped model -----------------------------------
    proba = model.predict_proba(X_te)
    classes = list(model.classes_)
    conf = proba.max(axis=1)
    top = np.array(classes)[proba.argmax(axis=1)]
    correct = top == y_te
    calib = reliability(conf, correct, config.CALIBRATION_BINS)

    # --- abstain threshold sweep ---------------------------------------------
    sweep = []
    for t in config.ABSTAIN_SWEEP:
        covered = conf >= t
        sweep.append({"threshold": t, "coverage": round(float(covered.mean()), 4),
                      "accuracy_on_covered": round(float(correct[covered].mean()), 4) if covered.any() else None})
    ok = [s for s in sweep if s["accuracy_on_covered"] is not None and s["accuracy_on_covered"] >= config.ABSTAIN_TARGET_ACC]
    abstain = ok[0]["threshold"] if ok else config.ABSTAIN_FALLBACK
    abstain_rule = (f"lowest threshold with accuracy_on_covered >= {config.ABSTAIN_TARGET_ACC}" if ok
                    else f"no threshold reached {config.ABSTAIN_TARGET_ACC}; fallback {config.ABSTAIN_FALLBACK}")
    abstain_row = next(s for s in sweep if s["threshold"] == abstain) if abstain in config.ABSTAIN_SWEEP else None

    # --- error analysis: most confident mistakes -----------------------------
    wrong = np.where(~correct)[0]
    worst = wrong[np.argsort(-conf[wrong], kind="mergesort")][:15]
    errors = [{"raw": X_te.iloc[i], "normalised": normalise(X_te.iloc[i]), "true": y_te[i],
               "pred": top[i], "confidence": round(float(conf[i]), 4)} for i in worst]

    acc = results[shipped]["accuracy"]
    lo, hi = config.ACCURACY_BAND
    gate = "LEAKAGE_SUSPECTED" if acc > hi else ("BELOW_BAND" if acc < lo else "OK")

    bundle = {
        "pipeline": model,
        "model_name": shipped,
        "classes": classes,
        "abstain_threshold": abstain,
        "n_train_rows": int(len(train)),
        "train_range": [train["date"].min(), train["date"].max()],
        "test_range": [test["date"].min(), test["date"].max()],
    }
    metrics = {
        "split": {"type": "time-based", "cutoff_date": cutoff, "n_train": int(len(train)), "n_test": int(len(test)),
                  "train_range": bundle["train_range"], "test_range": bundle["test_range"]},
        "models": results,
        "mcnemar": mc,
        "shipped_model": shipped,
        "accuracy": acc,
        "macro_f1": results[shipped]["macro_f1"],
        "gate": gate,
        "per_class": per_class,
        "confusion_matrix": {"labels": labels, "matrix": cm.tolist()},
        "calibration": calib,
        "abstain_sweep": sweep,
        "abstain_threshold": abstain,
        "abstain_rule": abstain_rule,
        "abstain_coverage": abstain_row["coverage"] if abstain_row else None,
        "abstain_accuracy_on_covered": abstain_row["accuracy_on_covered"] if abstain_row else None,
        "bootstrap_resamples": n_boot,
    }
    return {"bundle": bundle, "metrics": metrics, "errors": errors}


def save_bundle(bundle: dict) -> str:
    config.MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(bundle, config.CLASSIFIER_PATH)
    return hashlib.sha256(config.CLASSIFIER_PATH.read_bytes()).hexdigest()[:12]


def write_error_analysis(errors: list[dict], path=config.RESULTS_DIR / "error_analysis.md"):
    path.parent.mkdir(exist_ok=True)
    lines = ["# Error analysis: the 15 most confident WRONG predictions (synthetic test set)", "",
             "Generated by `python -m src.evaluate --report`. These are the model's worst failures: "
             "high confidence *and* wrong. Read each row as a failure mode, not a typo.", "",
             "| # | raw merchant string | true | predicted | confidence |", "|---|---|---|---|---|"]
    for i, e in enumerate(errors, 1):
        raw = e["raw"].replace("|", "\\|")
        lines.append(f"| {i} | `{raw}` | {e['true']} | {e['pred']} | {e['confidence']:.3f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_report(m: dict):
    print("\n=== CLASSIFIER (time-based split) ===")
    s = m["split"]
    print(f"train {s['train_range'][0]}..{s['train_range'][1]} n={s['n_train']:,} | "
          f"test {s['test_range'][0]}..{s['test_range'][1]} n={s['n_test']:,}")
    print(f"\n{'model':<20}{'accuracy':>28}{'macro-F1':>28}")
    for name, r in m["models"].items():
        a, f = r["accuracy_ci95"], r["macro_f1_ci95"]
        print(f"{name:<20}{r['accuracy']:>9.3f} [{a[0]:.3f}, {a[1]:.3f}]{'':>2}{r['macro_f1']:>9.3f} [{f[0]:.3f}, {f[1]:.3f}]")
    mc = m["mcnemar"]
    print(f"\nMcNemar table: {mc['table']}  discordant={mc['discordant']}  p={mc['p_value']:.4g}")
    print(f"SHIP: {mc['shipped']}. {mc['reason']}")
    print(f"\nPer-class report ({m['shipped_model']}):")
    print(f"{'category':<20}{'prec':>7}{'recall':>8}{'f1':>7}{'support':>9}")
    for c, r in m["per_class"].items():
        print(f"{c:<20}{r['precision']:>7.3f}{r['recall']:>8.3f}{r['f1-score']:>7.3f}{int(r['support']):>9}")
    print(f"{'accuracy':<20}{m['accuracy']:>22.3f}   macro-F1 {m['macro_f1']:.3f}")
    cal = m["calibration"]
    print(f"\nCalibration: ECE={cal['ece']:.4f}  Brier={cal['brier']:.4f}")
    print("Abstain sweep (threshold: coverage / acc-on-covered):")
    print("  " + "  ".join(f"{r['threshold']:.2f}:{r['coverage']:.2f}/{r['accuracy_on_covered']}" for r in m["abstain_sweep"]))
    print(f"Abstain threshold = {m['abstain_threshold']} ({m['abstain_rule']}); "
          f"coverage={m['abstain_coverage']} acc-on-covered={m['abstain_accuracy_on_covered']}")
    if m["gate"] == "LEAKAGE_SUSPECTED":
        print(f"\n!!! LEAKAGE SUSPECTED: accuracy {m['accuracy']:.3f} > {config.ACCURACY_BAND[1]}. The generator is "
              "leaking labels. Add category vocabulary overlap in data/generate.py. Do NOT accept this score.")
    elif m["gate"] == "BELOW_BAND":
        print(f"\n!!! BELOW BAND: accuracy {m['accuracy']:.3f} < {config.ACCURACY_BAND[0]}. The generator is too noisy.")
    else:
        print(f"\nGATE OK: accuracy {m['accuracy']:.3f} is inside {config.ACCURACY_BAND}.")


def gate_exit_code(m: dict) -> int:
    return {"OK": 0, "LEAKAGE_SUSPECTED": LEAKAGE_EXIT, "BELOW_BAND": BELOW_BAND_EXIT}[m["gate"]]


def main():
    df = pd.read_csv(config.TRANSACTIONS_CSV)
    out = train_and_evaluate(df)
    print_report(out["metrics"])
    sha = save_bundle(out["bundle"])
    print(f"Saved {config.CLASSIFIER_PATH.name} (sha256 {sha})")
    return gate_exit_code(out["metrics"])


if __name__ == "__main__":
    sys.exit(main())
