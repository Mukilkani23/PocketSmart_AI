"""Small, dependency-light statistics used across the evaluation.

Everything here is numpy + scipy only (statsmodels is deliberately excluded) and
short enough to explain line by line in a viva.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.stats import binomtest


def bootstrap_ci(y_true, y_pred, metric: Callable, n: int, seed: int, alpha: float = 0.05):
    """Percentile bootstrap CI: resample test rows with replacement, recompute metric."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y_true), size=(n, len(y_true)))
    scores = np.array([metric(y_true[i], y_pred[i]) for i in idx])
    lo, hi = np.percentile(scores, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def mcnemar_exact(y_true, pred_a, pred_b) -> dict:
    """Exact McNemar test: do models A and B have different error rates on the SAME rows?

    Only discordant pairs carry information: b = A right & B wrong, c = A wrong & B right.
    Under H0 (equal error rates) each discordant pair is a fair coin flip, so
    b ~ Binomial(b + c, 0.5). Two-sided exact binomial p-value.
    """
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    a_ok, b_ok = pred_a == y_true, pred_b == y_true
    table = {
        "both_correct": int(np.sum(a_ok & b_ok)),
        "a_correct_b_wrong": int(np.sum(a_ok & ~b_ok)),
        "a_wrong_b_correct": int(np.sum(~a_ok & b_ok)),
        "both_wrong": int(np.sum(~a_ok & ~b_ok)),
    }
    b, c = table["a_correct_b_wrong"], table["a_wrong_b_correct"]
    p = 1.0 if b + c == 0 else float(binomtest(b, b + c, 0.5, alternative="two-sided").pvalue)
    return {"table": table, "discordant": b + c, "p_value": p}


def reliability(confidence, correct, n_bins: int) -> dict:
    """Reliability curve, Expected Calibration Error and Brier score on max-probability.

    ECE = sum over bins of (bin share) * |bin accuracy - bin mean confidence|.
    Brier here is the binary Brier score of the top-1 prediction:
    mean((confidence - correct)^2).
    """
    confidence, correct = np.asarray(confidence, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, n_bins + 1)
    bins = np.clip(np.digitize(confidence, edges[1:-1]), 0, n_bins - 1)
    curve, ece = [], 0.0
    for k in range(n_bins):
        m = bins == k
        if not m.any():
            continue
        acc, conf, share = correct[m].mean(), confidence[m].mean(), m.mean()
        ece += share * abs(acc - conf)
        curve.append({"bin_lo": round(float(edges[k]), 2), "bin_hi": round(float(edges[k + 1]), 2),
                      "count": int(m.sum()), "mean_confidence": round(float(conf), 4),
                      "accuracy": round(float(acc), 4)})
    brier = float(np.mean((confidence - correct) ** 2))
    return {"ece": round(float(ece), 4), "brier": round(brier, 4), "curve": curve}


def fleiss_kappa(labels: np.ndarray, categories: list[str]) -> dict:
    """Fleiss' kappa for N items x n raters (same n for every item).

    P_i  = agreement on item i = (sum_j n_ij^2 - n) / (n (n-1))
    P_bar = mean P_i (observed), P_e = sum_j p_j^2 (chance, from overall label shares)
    kappa = (P_bar - P_e) / (1 - P_e)
    Also returns raw percent agreement (all raters agree) and mean pairwise agreement,
    which ARE on the same scale as accuracy. Kappa is not.
    """
    labels = np.asarray(labels)
    n_items, n_raters = labels.shape
    counts = np.array([[np.sum(row == c) for c in categories] for row in labels], float)
    p_i = (np.sum(counts ** 2, axis=1) - n_raters) / (n_raters * (n_raters - 1))
    p_bar = p_i.mean()
    p_j = counts.sum(axis=0) / (n_items * n_raters)
    p_e = float(np.sum(p_j ** 2))
    kappa = (p_bar - p_e) / (1 - p_e) if p_e < 1 else 1.0
    unanimous = float(np.mean([len(set(r)) == 1 for r in labels]))
    return {"fleiss_kappa": round(float(kappa), 4), "mean_pairwise_agreement": round(float(p_bar), 4),
            "unanimous_agreement": round(unanimous, 4), "n_items": int(n_items), "n_raters": int(n_raters)}
