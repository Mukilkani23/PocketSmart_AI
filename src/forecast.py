"""Weekly spend forecasting per category, with honest baselines.

Run:  python -m src.forecast

Four models, all reported, none dropped:
  naive            next week = last week
  seasonal_naive   next week = the week 4 weeks ago (~same point in the month)
  linear           LinearRegression on lags 1-4 + one-hot week-of-month, fit per category
  category_mean    constant = mean of the training weeks (the "do nothing clever" floor)

Evaluation: the last 8 complete weeks are held out by date. One-step-ahead
walk-forward: each holdout week is predicted from the actual weeks before it, and
nothing is refit on holdout data. The winner per category is the lowest holdout MAE,
and if that's a baseline, the baseline is what gets served. That is the result.

80% prediction interval: empirical 10th/90th percentile of one-step residuals from
a SEPARATE 16-week calibration window just before the holdout (models fit only on
data before that window). Coverage is then measured on the holdout. Calibrating on
the holdout itself would give ~80% coverage by construction, which proves nothing.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src import config

MODELS = ["naive", "seasonal_naive", "linear", "category_mean"]
LAGS = config.FORECAST_LAGS


def weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Debit spend per (week starting Monday, category). Complete weeks only, zeros filled."""
    d = df[df["direction"] == "debit"].copy()
    d["date"] = pd.to_datetime(d["date"])
    d["week"] = d["date"] - pd.to_timedelta(d["date"].dt.dayofweek, unit="D")
    first_full = d["date"].min() + pd.to_timedelta((7 - d["date"].min().dayofweek) % 7, unit="D")
    last_full = d["date"].max() - pd.to_timedelta((d["date"].max().dayofweek + 1) % 7, unit="D")  # last Sunday
    d = d[(d["week"] >= first_full) & (d["week"] + pd.Timedelta(days=6) <= last_full)]
    w = d.pivot_table(index="week", columns="category", values="amount", aggfunc="sum", fill_value=0.0)
    weeks = pd.date_range(w.index.min(), w.index.max(), freq="7D")
    return w.reindex(index=weeks, columns=config.CATEGORY_NAMES, fill_value=0.0)


def week_of_month(week_start: pd.Timestamp) -> int:
    """Week-of-month (1-5) of the week's Thursday: a week belongs to the month most of it lies in."""
    return ((week_start + pd.Timedelta(days=3)).day - 1) // 7 + 1


def features(y: np.ndarray, weeks: pd.DatetimeIndex, t: int) -> np.ndarray:
    """Row for predicting week t: lags 1..4 + one-hot week-of-month (5 slots)."""
    wom = np.zeros(5)
    wom[week_of_month(weeks[t]) - 1] = 1
    return np.concatenate([[y[t - k] for k in LAGS], wom])


def fit_linear(y: np.ndarray, weeks, end: int) -> LinearRegression:
    """Fit on target weeks max(LAGS)..end-1, i.e. strictly before `end`."""
    X = np.array([features(y, weeks, t) for t in range(max(LAGS), end)])
    return LinearRegression().fit(X, y[max(LAGS):end])


def predict_one(model: str, y, weeks, t, lin: LinearRegression | None, train_mean: float) -> float:
    if model == "naive":
        return float(y[t - 1])
    if model == "seasonal_naive":
        return float(y[t - 4])
    if model == "category_mean":
        return float(train_mean)
    return max(0.0, float(lin.predict(features(y, weeks, t)[None, :])[0]))  # spend can't be negative


def walk_forward(y, weeks, start: int, end: int) -> dict[str, np.ndarray]:
    """One-step-ahead predictions for weeks start..end-1; models fit only on weeks < start."""
    lin = fit_linear(y, weeks, start)
    mean = float(y[:start].mean())
    return {m: np.array([predict_one(m, y, weeks, t, lin, mean) for t in range(start, end)]) for m in MODELS}


def _mae(a, p):
    return float(np.mean(np.abs(a - p)))


def _mape(a, p):
    m = a > 0
    return float(np.mean(np.abs(a[m] - p[m]) / a[m]) * 100) if m.any() else None


def run(df: pd.DataFrame) -> dict:
    W = weekly(df)
    weeks = W.index
    n = len(weeks)
    h, c = config.FORECAST_HOLDOUT_WEEKS, config.FORECAST_CALIBRATION_WEEKS
    hold_start, cal_start = n - h, n - h - c
    lo_q, hi_q = config.INTERVAL_QUANTILES

    per_cat, served, history = {}, {}, {}
    for cat in config.CATEGORY_NAMES:
        y = W[cat].to_numpy(float)
        actual = y[hold_start:]
        preds = walk_forward(y, weeks, hold_start, n)
        rows = {}
        for m, p in preds.items():
            mape = _mape(actual, p)
            rows[m] = {"mae": round(_mae(actual, p), 2), "mape": None if mape is None else round(mape, 2)}
        winner = min(MODELS, key=lambda m: (rows[m]["mae"], MODELS.index(m)))

        # k/8: in how many holdout weeks did the winner beat the reference baseline?
        ref = "naive" if winner == "seasonal_naive" else "seasonal_naive"
        k = int(np.sum(np.abs(actual - preds[winner]) < np.abs(actual - preds[ref])))

        # 80% interval from a calibration window strictly before the holdout
        cal_preds = walk_forward(y, weeks, cal_start, hold_start)[winner]
        resid = y[cal_start:hold_start] - cal_preds
        q_lo, q_hi = np.quantile(resid, [lo_q, hi_q])
        lower = np.maximum(0, preds[winner] + q_lo)
        upper = preds[winner] + q_hi
        covered = (actual >= lower) & (actual <= upper)

        # live next-week forecast: winner refit on ALL complete weeks, same calibrated residual quantiles
        y_ext = np.append(y, np.nan)
        weeks_ext = weeks.append(pd.DatetimeIndex([weeks[-1] + pd.Timedelta(days=7)]))
        lin_all = fit_linear(y, weeks, n) if winner == "linear" else None
        point = predict_one(winner, y_ext, weeks_ext, n, lin_all, float(y.mean()))

        per_cat[cat] = {
            "models": rows, "winner": winner, "n_mape_weeks": int((actual > 0).sum()),
            "linear_beats_mean": rows["linear"]["mae"] < rows["category_mean"]["mae"],
            "winner_vs_reference": {"reference": ref, "weeks_won": k, "of": h, "robust": k > h // 2},
            "interval": {"q10_residual": round(float(q_lo), 2), "q90_residual": round(float(q_hi), 2),
                         "holdout_coverage": round(float(covered.mean()), 4)},
        }
        served[cat] = {
            "category": cat, "week_start": str(weeks_ext[-1].date()),
            "prediction": round(point, 2), "lower_80": round(max(0.0, point + q_lo), 2),
            "upper_80": round(point + q_hi, 2), "model_used": winner,
            "holdout_mae": rows[winner]["mae"], "holdout_mape": rows[winner]["mape"],
            "interval_coverage_holdout": round(float(covered.mean()), 4),
            "weeks_beat_reference": f"{k}/{h} vs {ref}",
        }
        history[cat] = {"weeks": [str(w.date()) for w in weeks], "actual": [round(v, 2) for v in y],
                        "holdout_start": str(weeks[hold_start].date()),
                        "holdout_pred": [round(v, 2) for v in preds[winner]],
                        "lower": [round(v, 2) for v in lower], "upper": [round(v, 2) for v in upper]}

    summary = {}
    for m in MODELS:
        maes = [per_cat[c]["models"][m]["mae"] for c in per_cat]
        mapes = [per_cat[c]["models"][m]["mape"] for c in per_cat if per_cat[c]["models"][m]["mape"] is not None]
        summary[m] = {"mean_mae": round(float(np.mean(maes)), 2), "mean_mape": round(float(np.mean(mapes)), 2),
                      "categories_won": sum(per_cat[c]["winner"] == m for c in per_cat)}
    all_cov = np.mean([per_cat[c]["interval"]["holdout_coverage"] for c in per_cat])
    metrics = {
        "n_weeks": n, "holdout_weeks": h, "calibration_weeks": c,
        "holdout_range": [str(weeks[hold_start].date()), str(weeks[-1].date())],
        "calibration_range": [str(weeks[cal_start].date()), str(weeks[hold_start - 1].date())],
        "summary": summary, "per_category": per_cat,
        "overall_best_model": min(summary, key=lambda m: summary[m]["mean_mae"]),
        "linear_beats_category_mean_overall": summary["linear"]["mean_mae"] < summary["category_mean"]["mean_mae"],
        "interval_target": 0.80, "interval_coverage_mean": round(float(all_cov), 4),
        "not_robust_winners": [c for c in per_cat if not per_cat[c]["winner_vs_reference"]["robust"]],
    }
    return {"metrics": metrics, "served": served, "history": history}


def save(out: dict):
    config.MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump({"served": out["served"], "history": out["history"]}, config.FORECAST_PATH)


def print_report(m: dict):
    print("\n=== FORECAST (weekly spend per category, one-step walk-forward) ===")
    print(f"{m['n_weeks']} complete weeks; holdout {m['holdout_range'][0]}..{m['holdout_range'][1]} "
          f"({m['holdout_weeks']} wks); interval calibration {m['calibration_range'][0]}..{m['calibration_range'][1]}")
    print(f"\n{'Model':<16}{'mean MAE (INR)':>16}{'mean MAPE %':>13}{'cats won':>10}")
    for k, s in m["summary"].items():
        print(f"{k:<16}{s['mean_mae']:>16,.2f}{s['mean_mape']:>13.2f}{s['categories_won']:>10}")
    print(f"\nOverall best on mean MAE: {m['overall_best_model']}. "
          f"Linear beats category-mean overall: {m['linear_beats_category_mean_overall']}")
    print(f"\n{'category':<20}{'naive':>9}{'seas':>9}{'linear':>9}{'mean':>9}  {'winner':<15}{'k/8':>6}{'PI cov':>8}")
    for c, r in m["per_category"].items():
        mm, k = r["models"], r["winner_vs_reference"]
        flag = "" if k["robust"] else "  NOT ROBUST"
        print(f"{c:<20}{mm['naive']['mae']:>9.0f}{mm['seasonal_naive']['mae']:>9.0f}{mm['linear']['mae']:>9.0f}"
              f"{mm['category_mean']['mae']:>9.0f}  {r['winner']:<15}{k['weeks_won']:>4}/{k['of']}"
              f"{r['interval']['holdout_coverage']:>8.2f}{flag}")
    print(f"\n80% interval: mean holdout coverage = {m['interval_coverage_mean']:.3f} (target 0.80)")


def main():
    df = pd.read_csv(config.TRANSACTIONS_CSV)
    out = run(df)
    print_report(out["metrics"])
    save(out)


if __name__ == "__main__":
    main()
