"""Assert the freshly trained model reproduces the COMMITTED metrics.

Run:  python -m src.verify_build     (a Dockerfile RUN step, after `python -m src.evaluate`)

The image retrains from source (seeded, pinned versions). This recomputes
synthetic test accuracy and macro-F1 from the new bundle and compares them with
results/metrics.json, which is the file the README and /metrics publish.
Any difference > TOL fails the Docker build, so the deployed model is verified
against the published numbers.

TOL = 0.005 absorbs harmless float differences between Windows (where metrics.json
was produced) and Linux (the container), e.g. in the lbfgs solver. A real drift,
such as a changed generator, library version or code, is far larger than that.
"""
from __future__ import annotations

import hashlib
import json
import sys

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src import config
from src.train_classifier import time_split

TOL = 0.005


def main() -> int:
    if not config.METRICS_PATH.exists():
        print("BUILD VERIFICATION FAILED: results/metrics.json missing (run `python -m src.evaluate --report` and commit it)")
        return 1
    committed = json.loads(config.METRICS_PATH.read_text())["classifier"]
    bundle = joblib.load(config.CLASSIFIER_PATH)
    _, test, _ = time_split(pd.read_csv(config.TRANSACTIONS_CSV))
    pred = bundle["pipeline"].predict(test["merchant_raw"])
    acc = accuracy_score(test["category"], pred)
    mf1 = f1_score(test["category"], pred, average="macro", zero_division=0)
    d_acc, d_f1 = abs(acc - committed["accuracy"]), abs(mf1 - committed["macro_f1"])
    sha = hashlib.sha256(config.CLASSIFIER_PATH.read_bytes()).hexdigest()[:12]
    ok = d_acc <= TOL and d_f1 <= TOL and bundle["model_name"] == committed["shipped_model"]

    config.BUILD_VERIFICATION_PATH.write_text(json.dumps({
        "metrics_verified": ok, "model_sha256": sha, "model_name": bundle["model_name"],
        "rebuilt_accuracy": round(acc, 4), "committed_accuracy": committed["accuracy"],
        "rebuilt_macro_f1": round(mf1, 4), "committed_macro_f1": committed["macro_f1"],
        "acc_delta": round(d_acc, 4), "f1_delta": round(d_f1, 4), "tolerance": TOL,
    }, indent=2))
    if ok:
        print(f"BUILD MODEL MATCHES COMMITTED METRICS (acc delta={d_acc:.4f}, macro-F1 delta={d_f1:.4f}, "
              f"model={bundle['model_name']}, sha256={sha})")
        return 0
    print("BUILD VERIFICATION FAILED: rebuilt model does not match results/metrics.json")
    print(f"  model     rebuilt={bundle['model_name']}  committed={committed['shipped_model']}")
    print(f"  accuracy  rebuilt={acc:.4f}  committed={committed['accuracy']:.4f}  delta={d_acc:.4f}")
    print(f"  macro-F1  rebuilt={mf1:.4f}  committed={committed['macro_f1']:.4f}  delta={d_f1:.4f}  (tol {TOL})")
    return 1


if __name__ == "__main__":
    sys.exit(main())
