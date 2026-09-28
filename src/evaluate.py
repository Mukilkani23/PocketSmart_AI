"""Single entry point that (re)builds every model and every reported number.

  python -m src.evaluate            build artifacts only (models/*.joblib). Used by the Dockerfile.
  python -m src.evaluate --report   ALSO writes results/metrics.json, the PNGs,
                                    results/error_analysis.md, results/real_errors.md, and rewrites
                                    the README tables between <!-- METRICS:<NAME>:START/END --> markers.

results/metrics.json is the single source of truth: no number appears in the
README unless it is rendered from that file by this script. Everything is seeded,
so two runs produce a byte-identical metrics.json (timings are printed, never stored).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time

import numpy as np
import pandas as pd

from src import anomaly, config, forecast, train_classifier, validate_real

README = config.ROOT / "README.md"


# --------------------------------------------------------------------------- build
def build(report: bool) -> dict:
    df = pd.read_csv(config.TRANSACTIONS_CSV)
    t0 = time.perf_counter()

    clf = train_classifier.train_and_evaluate(df)
    train_classifier.print_report(clf["metrics"])
    sha = train_classifier.save_bundle(clf["bundle"])
    print(f"Saved classifier.joblib sha256={sha}  ({time.perf_counter() - t0:.1f}s)")

    scored = anomaly.score(df)
    an = anomaly.evaluate(scored)
    anomaly.print_report(an)
    anomaly.save(scored, an)

    fc = forecast.run(df)
    forecast.print_report(fc["metrics"])
    forecast.save(fc)

    data = {
        "synthetic": True, "generator": "data/generate.py", "seed": config.SEED,
        "n_rows": int(len(df)), "date_range": [df["date"].min(), df["date"].max()],
        "n_months": int(pd.to_datetime(df["date"]).dt.to_period("M").nunique()),
        "n_categories": len(config.CATEGORY_NAMES),
        "category_counts": {c: int(n) for c, n in df["category"].value_counts().reindex(config.CATEGORY_NAMES).items()},
        "direction_counts": {k: int(v) for k, v in df["direction"].value_counts().items()},
        "n_anomalies": int(df["is_anomaly"].sum()), "anomaly_rate": round(float(df["is_anomaly"].mean()), 4),
    }
    cfg = {
        "test_fraction": config.TEST_FRACTION, "accuracy_band": list(config.ACCURACY_BAND),
        "tfidf_char_ngram_range": [2, 4], "anomaly_window_days": config.ANOMALY_WINDOW_DAYS,
        "anomaly_z_sweep": config.ANOMALY_Z_SWEEP, "forecast_holdout_weeks": config.FORECAST_HOLDOUT_WEEKS,
        "forecast_calibration_weeks": config.FORECAST_CALIBRATION_WEEKS, "forecast_lags": config.FORECAST_LAGS,
        "seasonal_lag_weeks": 4, "interval_quantiles": list(config.INTERVAL_QUANTILES),
        "mcnemar_alpha": config.MCNEMAR_ALPHA, "abstain_target_accuracy": config.ABSTAIN_TARGET_ACC,
        "library_versions": _versions(),
    }
    metrics = {"config": cfg, "data": data, "classifier": clf["metrics"], "anomaly": an, "forecast": fc["metrics"]}

    if report:
        real = validate_real.run(metrics["classifier"]["accuracy"])
        validate_real.print_report(real)
        metrics["real_validation"] = real
        train_classifier.write_error_analysis(clf["errors"])
        plots(metrics, scored, fc)
        config.RESULTS_DIR.mkdir(exist_ok=True)
        config.METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n")
        inject_readme(metrics)
        print(f"\nWrote {config.METRICS_PATH.relative_to(config.ROOT)} and results/*.png, *.md")
    print(f"Total runtime {time.perf_counter() - t0:.1f}s")
    return metrics


def _versions() -> dict:
    import scipy
    import sklearn

    return {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__, "scipy": scipy.__version__}


# --------------------------------------------------------------------------- plots
def plots(m: dict, scored: pd.DataFrame, fc: dict):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve

    R = config.RESULTS_DIR
    R.mkdir(exist_ok=True)
    labels = m["classifier"]["confusion_matrix"]["labels"]
    cm = np.array(m["classifier"]["confusion_matrix"]["matrix"], float)
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(9, 7.5))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j]:
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true (row = all test txns of that category)")
    ax.set_title(f"Confusion matrix: {m['classifier']['shipped_model']}, time-split test set\n"
                 "colour = row-normalised recall, numbers = counts")
    fig.tight_layout()
    fig.savefig(R / "confusion_matrix.png", dpi=110)
    plt.close(fig)

    cal = m["classifier"]["calibration"]
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect calibration")
    ax.plot([b["mean_confidence"] for b in cal["curve"]], [b["accuracy"] for b in cal["curve"]], "o-",
            label=f"{m['classifier']['shipped_model']} (ECE {cal['ece']:.3f})")
    ax.axvline(m["classifier"]["abstain_threshold"], color="tab:red", lw=1,
               label=f"abstain threshold {m['classifier']['abstain_threshold']}")
    ax.set_xlabel("mean predicted confidence (bin)")
    ax.set_ylabel("actual accuracy (bin)")
    ax.set_title("Reliability curve (10 bins, top-1 confidence)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(R / "calibration.png", dpi=110)
    plt.close(fig)

    e = scored[scored["evaluable"]]
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for var in ("raw", "log"):
        p, r, _ = precision_recall_curve(e["is_anomaly"], np.abs(e[f"z_{var}"]))
        ax.plot(r, p, label=f"{var}-z (AUC-PR {m['anomaly']['variants'][var]['auc_pr']:.3f})")
    ax.axhline(m["anomaly"]["base_rate"], color="grey", ls="--", label=f"random ({m['anomaly']['base_rate']:.3f})")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("Anomaly detector: precision-recall")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(R / "anomaly_pr.png", dpi=110)
    plt.close(fig)

    fig, axes = plt.subplots(5, 2, figsize=(12, 14), sharex=True)
    for ax, cat in zip(axes.flat, config.CATEGORY_NAMES):
        h = fc["history"][cat]
        weeks = pd.to_datetime(h["weeks"])
        ax.plot(weeks, h["actual"], color="black", lw=1, label="actual")
        hw = weeks[-len(h["holdout_pred"]):]
        ax.fill_between(hw, h["lower"], h["upper"], alpha=0.25, label="80% interval")
        ax.plot(hw, h["holdout_pred"], "o-", ms=3, label=fc["served"][cat]["model_used"])
        ax.set_title(f"{cat}: {fc['served'][cat]['model_used']} (holdout MAE {fc['served'][cat]['holdout_mae']:,.0f})",
                     fontsize=9)
        ax.legend(fontsize=7, loc="upper left")
        loc = mdates.AutoDateLocator(maxticks=6)
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    fig.suptitle("Weekly spend (INR): history, 8-week holdout prediction and 80% interval")
    fig.tight_layout()
    fig.savefig(R / "forecast_plot.png", dpi=100)
    plt.close(fig)


# --------------------------------------------------------------------------- README
def _f(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def render_blocks(m: dict) -> dict[str, str]:
    c, a, f, d = m["classifier"], m["anomaly"], m["forecast"], m["data"]
    B = {}
    B["DATA"] = "\n".join([
        f"- **Synthetic**, generated by `{d['generator']}` (seed {d['seed']}): **{d['n_rows']:,} transactions**, "
        f"{d['date_range'][0]} → {d['date_range'][1]} ({d['n_months']} months), {d['n_categories']} categories.",
        f"- Injected ground-truth anomalies: {d['n_anomalies']} ({d['anomaly_rate']:.2%} of rows), used only for scoring.",
        "", "| category | rows |", "|---|---|",
        *[f"| {k} | {v:,} |" for k, v in d["category_counts"].items()],
    ])

    s = c["split"]
    rows = [f"| {n} | {r['accuracy']:.3f} [{r['accuracy_ci95'][0]:.3f}, {r['accuracy_ci95'][1]:.3f}] | "
            f"{r['macro_f1']:.3f} [{r['macro_f1_ci95'][0]:.3f}, {r['macro_f1_ci95'][1]:.3f}] |"
            for n, r in c["models"].items()]
    mc = c["mcnemar"]
    B["CLASSIFIER"] = "\n".join([
        f"Time-based split: train {s['train_range'][0]}..{s['train_range'][1]} (n={s['n_train']:,}), "
        f"test {s['test_range'][0]}..{s['test_range'][1]} (n={s['n_test']:,}). "
        f"95% CIs from {c['bootstrap_resamples']} bootstrap resamples of the test set.", "",
        "| model | accuracy [95% CI] | macro-F1 [95% CI] |", "|---|---|---|", *rows, "",
        f"**McNemar exact test** (LR vs RF on the same test rows): LR-only correct = {mc['table']['a_correct_b_wrong']}, "
        f"RF-only correct = {mc['table']['a_wrong_b_correct']}, p = {mc['p_value']:.4f}. "
        f"**Shipped: {c['shipped_model']}.** {mc['reason']}", "",
        f"Per-class report ({c['shipped_model']}):", "",
        "| category | precision | recall | F1 | support |", "|---|---|---|---|---|",
        *[f"| {k} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1-score']:.3f} | {int(v['support'])} |"
          for k, v in c["per_class"].items()],
        f"| **overall** | | accuracy **{c['accuracy']:.3f}** | macro-F1 **{c['macro_f1']:.3f}** | {s['n_test']:,} |", "",
        f"Leakage gate (accuracy band {config.ACCURACY_BAND[0]}–{config.ACCURACY_BAND[1]}): **{c['gate']}**.", "",
        "![confusion matrix](results/confusion_matrix.png)",
    ])

    lines = []
    for var, v in a["variants"].items():
        lines += [f"**{var}-z**, AUC-PR = {v['auc_pr']:.3f} (random = base rate {a['base_rate']:.3f})", "",
                  "| \\|z\\| > | precision | recall | F1 | TP | FP | FN | TN |", "|---|---|---|---|---|---|---|---|",
                  *[f"| {r['z']} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | {r['tp']} | {r['fp']} | "
                    f"{r['fn']} | {r['tn']} |" for r in v["sweep"]], ""]
    sv = a["served"]
    B["ANOMALY"] = "\n".join([
        f"Evaluated on {a['n_evaluated']:,} debits after warm-up; {a['n_true_anomalies']} true anomalies.", "",
        *lines,
        f"**Served:** {sv['variant']}-z at \\|z\\| > {sv['threshold']} (higher AUC-PR variant, best-F1 threshold; "
        "threshold chosen on the scored data, so its F1 is optimistic).", "",
        "Per-category precision at the served threshold (this is where the method breaks):", "",
        "| category | flagged | TP | FP | precision | recall |", "|---|---|---|---|---|---|",
        *[f"| {k} | {r['flagged']} | {r['tp']} | {r['fp']} | {_f(r['precision'])} | {_f(r['recall'])} |"
          for k, r in a["per_category"].items()],
        "", "![anomaly PR curve](results/anomaly_pr.png)",
    ])

    fs = f["summary"]
    worse = [k for k, r in f["per_category"].items() if not r["linear_beats_mean"]]
    B["FORECAST"] = "\n".join([
        f"{f['n_weeks']} complete weeks. Holdout = last {f['holdout_weeks']} weeks "
        f"({f['holdout_range'][0]} → {f['holdout_range'][1]}), one-step-ahead walk-forward, no refit on holdout.", "",
        "| Model | mean MAE (INR) | mean MAPE (%) | categories won |", "|---|---|---|---|",
        *[f"| {k} | {v['mean_mae']:,.2f} | {v['mean_mape']:.2f} | {v['categories_won']} |" for k, v in fs.items()], "",
        f"Lowest mean MAE: **{f['overall_best_model']}**. " + (
            f"**The category-mean baseline beats LinearRegression in {len(worse)} of {len(f['per_category'])} "
            f"categories ({', '.join(worse)}).**" if worse else "LinearRegression beats the category mean everywhere."), "",
        "| category | naive | seasonal naive | linear | category mean | served | beat reference (k/8) |",
        "|---|---|---|---|---|---|---|",
        *[f"| {k} | {r['models']['naive']['mae']:,.0f} | {r['models']['seasonal_naive']['mae']:,.0f} | "
          f"{r['models']['linear']['mae']:,.0f} | {r['models']['category_mean']['mae']:,.0f} | {r['winner']} | "
          f"{r['winner_vs_reference']['weeks_won']}/{r['winner_vs_reference']['of']} vs "
          f"{r['winner_vs_reference']['reference']}{'' if r['winner_vs_reference']['robust'] else ' ⚠ not robust'} |"
          for k, r in f["per_category"].items()],
        "", "(per-category cells are holdout MAE in INR)", "", "![forecast](results/forecast_plot.png)",
    ])

    r = m.get("real_validation", {})
    if r.get("status") == "OK":
        ha = r["human_agreement"]
        warn = "" if r["string_source"] == "real" else (
            f"\n> ⚠ **These labelled strings came from `{r['string_source']}`, not real statements. "
            "This is a pipeline test, not a real-data result.**\n")
        B["REAL"] = "\n".join([
            warn,
            "| evaluation set | n | accuracy | macro-F1 |", "|---|---|---|---|",
            f"| synthetic test (time split) | {c['split']['n_test']:,} | {c['accuracy']:.3f} | {c['macro_f1']:.3f} |",
            f"| real, human-labelled | {r['n']} | {r['accuracy']:.3f} | {r['macro_f1']:.3f} |", "",
            f"- **Generalisation gap** (synthetic − real accuracy): **{r['generalisation_gap']:.3f}**",
            f"- **Human agreement** on {ha.get('n_items')} anchor strings × {ha.get('n_raters')} labellers: "
            f"Fleiss' κ = {_f(ha.get('fleiss_kappa'))}, mean pairwise agreement = "
            f"{_f(ha.get('mean_pairwise_agreement'))} (the ceiling; κ is chance-corrected, so it isn't on the accuracy scale)",
            f"- Model accuracy minus human ceiling: {_f(r['accuracy_vs_ceiling'])}",
            f"- At the abstain threshold {r['abstain_threshold']}: coverage {r['coverage_at_threshold']:.3f}, "
            f"accuracy on covered {_f(r['accuracy_on_covered'])}", "",
            "| category | real recall | n |", "|---|---|---|",
            *[f"| {k} | {v:.3f} | {r['per_class_support'][k]} |" for k, v in r["per_class_recall"].items()],
            "", "Every misclassified real string: [results/real_errors.md](results/real_errors.md).",
        ])
    else:
        B["REAL"] = ("**Pending.** The human labels haven't been ingested yet, so `src/validate_real.py` reported "
                     "`SKIPPED`. This section is filled automatically by `python -m src.evaluate --report` once "
                     "`data/real_validation.csv` exists.")

    F = []
    mods = c["models"]
    lr, rf = mods["LogisticRegression"], mods["RandomForest"]
    if rf["macro_f1"] > lr["macro_f1"] and c["shipped_model"] == "LogisticRegression":
        F.append(f"- **RandomForest has the higher macro-F1 ({rf['macro_f1']:.3f} vs {lr['macro_f1']:.3f}) but is not "
                 f"shipped.** McNemar (p = {mc['p_value']:.4f}) says LR makes significantly fewer errors overall. RF "
                 "trades overall accuracy for small-class recall.")
    weakest = sorted(c["per_class"].items(), key=lambda kv: kv[1]["f1-score"])[:2]
    F.append("- **Weakest classes:** " + ", ".join(f"{k} (F1 {v['f1-score']:.3f})" for k, v in weakest)
             + ". Payments to individuals are ambiguous from the text alone.")
    curve = c["calibration"]["curve"]
    gap = sum(b["count"] * (b["accuracy"] - b["mean_confidence"]) for b in curve) / max(1, sum(b["count"] for b in curve))
    if c["calibration"]["ece"] > 0.05:
        F.append(f"- **The shipped classifier is {'under' if gap > 0 else 'over'}confident** (ECE {c['calibration']['ece']:.3f}): "
                 f"on average its accuracy is {abs(gap):.3f} {'above' if gap > 0 else 'below'} its stated confidence. "
                 "Balanced class weights plus L2 regularisation spread probability across 10 classes. The abstain "
                 "threshold is picked from *observed* accuracy, not from the raw probability, so it stays valid.")
    lv, rv = a["variants"]["log"]["auc_pr"], a["variants"]["raw"]["auc_pr"]
    if rv > lv:
        F.append(f"- **My log-z hypothesis was wrong.** I expected z on log(amount) to win because amounts are "
                 f"lognormal. Raw-z scored AUC-PR {rv:.3f} against log-z's {lv:.3f}. Multiplicative outliers are only a "
                 "modest shift in log space when a category's spread is wide.")
    worst_a = sorted([(k, v) for k, v in a["per_category"].items() if v["precision"] is not None],
                     key=lambda kv: kv[1]["precision"])[:2]
    F.append("- **Anomaly precision collapses where a category is multimodal or heavy-tailed:** "
             + ", ".join(f"{k} {v['precision']:.3f}" for k, v in worst_a) + ".")
    if worse:
        F.append(f"- **A constant beat the regression in {len(worse)} of {len(f['per_category'])} forecast categories.** "
                 "Weekly spend in most categories is close to noise around a mean. Lag features let the regression "
                 "chase last week's noise.")
    if f["not_robust_winners"]:
        F.append("- **Not robust:** " + ", ".join(f["not_robust_winners"]) + ". The lowest-MAE model won no more than "
                 "half of the holdout weeks, so its 'win' is one or two lucky weeks.")
    if r.get("status") == "OK":
        F.append(f"- **Real-data gap: {r['generalisation_gap']:.3f}** accuracy lost moving from synthetic to real strings.")
    B["FINDINGS"] = "\n".join(F)

    cal = c["calibration"]
    B["UNCERTAINTY"] = "\n".join([
        f"- **Classifier calibration** ({c['shipped_model']}, synthetic test): ECE = **{cal['ece']:.4f}**, "
        f"Brier (top-1) = {cal['brier']:.4f}. ![reliability](results/calibration.png)",
        f"- **Abstain threshold = {c['abstain_threshold']}** ({c['abstain_rule']}): the model answers on "
        f"{_f(c['abstain_coverage'])} of test transactions with accuracy {_f(c['abstain_accuracy_on_covered'])} "
        "on those; below the threshold `/categorize` returns `uncertain: true`.",
        "", "| threshold | coverage | accuracy on covered |", "|---|---|---|",
        *[f"| {s['threshold']:.2f} | {s['coverage']:.3f} | {_f(s['accuracy_on_covered'])} |" for s in c["abstain_sweep"]],
        "",
        f"- **Forecast 80% intervals** (residual quantiles from a separate {f['calibration_weeks']}-week calibration "
        f"window, {f['calibration_range'][0]} → {f['calibration_range'][1]}): mean holdout coverage "
        f"**{f['interval_coverage_mean']:.3f}** vs target {f['interval_target']:.2f}. "
        f"Per category: " + ", ".join(f"{k} {v['interval']['holdout_coverage']:.3f}" for k, v in f["per_category"].items()),
    ])
    return B


def inject_readme(m: dict):
    if not README.exists():
        print("README.md not found; skipping table injection")
        return
    text = README.read_text(encoding="utf-8")
    for name, body in render_blocks(m).items():
        pat = re.compile(rf"(<!-- METRICS:{name}:START -->)(.*?)(<!-- METRICS:{name}:END -->)", re.S)
        if not pat.search(text):
            print(f"README: marker METRICS:{name} not found; skipped")
            continue
        text = pat.sub(lambda mt: f"{mt.group(1)}\n{body}\n{mt.group(3)}", text)
    README.write_text(text, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true", help="also write metrics.json, plots, markdown, README tables")
    args = ap.parse_args()
    m = build(args.report)
    return train_classifier.gate_exit_code(m["classifier"])


if __name__ == "__main__":
    sys.exit(main())
