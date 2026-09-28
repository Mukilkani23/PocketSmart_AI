"""Per-category rolling z-score anomaly detector.

Run:  python -m src.anomaly

For each debit, compare its amount with the SAME category's transactions over the
PREVIOUS 28 days (closed='left': the current day is excluded, so a spike can't
inflate its own baseline and hide itself).

Two variants, both evaluated:
- log-z: z computed on log(amount). Amounts are lognormal (right-skewed), so on
  the raw scale a normal big purchase already sits several SDs out; the log turns
  the distribution roughly normal, where "|z| > 3" means what it says.
- raw-z: z on the raw rupee amount, kept as the comparison.

Ground truth `is_anomaly` is used ONLY here, to score the detector. Nothing is
trained on it. The detector has no fitted parameters except the threshold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

from src import config

MIN_HISTORY = 5  # need at least this many past txns in the window to compute a z-score


def score(df: pd.DataFrame) -> pd.DataFrame:
    """Return debits with z_log, z_raw and the 28-day typical amount (window excludes today)."""
    d = df[df["direction"] == "debit"].copy()
    d["date"] = pd.to_datetime(d["date"])
    d["log_amt"] = np.log(d["amount"])
    d = d.sort_values(["category", "date", "txn_id"], kind="mergesort")
    win = f"{config.ANOMALY_WINDOW_DAYS}D"
    parts = []
    for _, g in d.groupby("category", sort=True):
        g = g.set_index("date")
        roll_log = g["log_amt"].rolling(win, closed="left")
        roll_raw = g["amount"].rolling(win, closed="left")
        n = roll_log.count()
        g = g.assign(
            n_hist=n.to_numpy(),
            mu_log=roll_log.mean().to_numpy(), sd_log=roll_log.std().to_numpy(),
            mu_raw=roll_raw.mean().to_numpy(), sd_raw=roll_raw.std().to_numpy(),
        )
        parts.append(g.reset_index())
    d = pd.concat(parts, ignore_index=True)
    ok = (d["n_hist"] >= MIN_HISTORY) & (d["sd_log"] > 0)
    d["z_log"] = np.where(ok, (d["log_amt"] - d["mu_log"]) / d["sd_log"], np.nan)
    d["z_raw"] = np.where(ok, (d["amount"] - d["mu_raw"]) / d["sd_raw"], np.nan)
    d["typical_amount"] = np.exp(d["mu_log"])  # geometric mean = "typical" spend in the window
    warmup_end = d["date"].min() + pd.Timedelta(days=config.ANOMALY_WINDOW_DAYS)
    d["evaluable"] = ok & (d["date"] >= warmup_end)
    d["date"] = d["date"].dt.strftime("%Y-%m-%d")
    return d.sort_values(["date", "txn_id"], kind="mergesort").reset_index(drop=True)


def _prf(y, flag):
    p, r, f, _ = precision_recall_fscore_support(y, flag, average="binary", zero_division=0)
    return round(float(p), 4), round(float(r), 4), round(float(f), 4)


def evaluate(scored: pd.DataFrame) -> dict:
    e = scored[scored["evaluable"]]
    y = e["is_anomaly"].to_numpy()
    out = {"n_evaluated": int(len(e)), "n_true_anomalies": int(y.sum()),
           "window_days": config.ANOMALY_WINDOW_DAYS, "min_history": MIN_HISTORY, "variants": {}}
    for var in ("log", "raw"):
        z = np.abs(e[f"z_{var}"].to_numpy())
        sweep = []
        for t in config.ANOMALY_Z_SWEEP:
            flag = (z > t).astype(int)
            p, r, f = _prf(y, flag)
            tp = int(((flag == 1) & (y == 1)).sum())
            fp = int(((flag == 1) & (y == 0)).sum())
            fn = int(((flag == 0) & (y == 1)).sum())
            tn = int(((flag == 0) & (y == 0)).sum())
            sweep.append({"z": t, "precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn, "tn": tn})
        best = max(sweep, key=lambda s: (s["f1"], -s["z"]))
        out["variants"][var] = {"auc_pr": round(float(average_precision_score(y, z)), 4),
                                "sweep": sweep, "best_f1_threshold": best["z"]}
    out["base_rate"] = round(float(y.mean()), 4)

    # Served detector = variant with higher AUC-PR, at its best-F1 threshold.
    served = max(out["variants"], key=lambda v: out["variants"][v]["auc_pr"])
    thr = out["variants"][served]["best_f1_threshold"]
    out["served"] = {"variant": served, "threshold": thr,
                     "note": "threshold chosen on the same data it is scored on (no separate tuning split); "
                             "treat its F1 as optimistic"}
    at3 = next(s for s in out["variants"][served]["sweep"] if s["z"] == config.ANOMALY_Z_THRESHOLD)
    out["at_brief_threshold_3"] = at3

    z = np.abs(e[f"z_{served}"].to_numpy())
    per_cat = {}
    for cat in config.CATEGORY_NAMES:
        m = (e["category"] == cat).to_numpy()
        if not m.any():
            continue
        flag = (z[m] > thr).astype(int)
        yt = y[m]
        tp, fp = int(((flag == 1) & (yt == 1)).sum()), int(((flag == 1) & (yt == 0)).sum())
        per_cat[cat] = {"flagged": int(flag.sum()), "tp": tp, "fp": fp, "true_anomalies": int(yt.sum()),
                        "precision": round(tp / (tp + fp), 4) if tp + fp else None,
                        "recall": round(tp / yt.sum(), 4) if yt.sum() else None}
    out["per_category"] = per_cat
    return out


def print_report(m: dict):
    print("\n=== ANOMALY DETECTION (rolling 28-day per-category z-score) ===")
    print(f"evaluated debits={m['n_evaluated']:,}  true anomalies={m['n_true_anomalies']}  base rate={m['base_rate']:.2%}")
    for var, v in m["variants"].items():
        print(f"\n{var}-z   AUC-PR={v['auc_pr']:.3f}   (random detector AUC-PR = base rate {m['base_rate']:.3f})")
        print(f"  {'|z|>':>5}{'prec':>8}{'recall':>8}{'F1':>7}{'TP':>6}{'FP':>6}{'FN':>6}{'TN':>7}")
        for s in v["sweep"]:
            print(f"  {s['z']:>5}{s['precision']:>8.3f}{s['recall']:>8.3f}{s['f1']:>7.3f}{s['tp']:>6}{s['fp']:>6}{s['fn']:>6}{s['tn']:>7}")
    s = m["served"]
    print(f"\nServed: {s['variant']}-z at |z|>{s['threshold']} (best F1). Brief's |z|>3: "
          f"P={m['at_brief_threshold_3']['precision']:.3f} R={m['at_brief_threshold_3']['recall']:.3f}")
    print("Per-category precision at the served threshold:")
    for c, r in m["per_category"].items():
        print(f"  {c:<20} flagged={r['flagged']:>4}  TP={r['tp']:>3}  FP={r['fp']:>4}  precision={r['precision']}  recall={r['recall']}")


def main():
    df = pd.read_csv(config.TRANSACTIONS_CSV)
    print_report(evaluate(score(df)))


if __name__ == "__main__":
    main()
