"""PocketSmart API. Models load once at import; the frontend is served at "/".

Run locally:  uvicorn api.main:app --reload     (docs at /docs)
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict, deque

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from api.schemas import (AdviceRequest, AdviceResponse, AnomaliesResponse, AnomalyItem, CategorizeItem,
                         CategorizeRequest, CategorizeResponse, CategoryTotal, ForecastItem, ForecastResponse,
                         HealthResponse, LabelProb, LatencyResponse, LatencyStat, SummaryResponse)
from src import config, gemini
from src.features import normalise

# ---------------------------------------------------------------- load once, at module scope
BOOT = time.time()
_CLF = joblib.load(config.CLASSIFIER_PATH)
PIPE, CLASSES, ABSTAIN = _CLF["pipeline"], np.array(_CLF["pipeline"].classes_), float(_CLF["abstain_threshold"])
MODEL_SHA = hashlib.sha256(config.CLASSIFIER_PATH.read_bytes()).hexdigest()[:12]
FORECAST = joblib.load(config.FORECAST_PATH)
_AN = joblib.load(config.ANOMALY_PATH)
AN_SCORED, AN_SERVED = _AN["scored"], _AN["served"]
TXNS = pd.read_csv(config.TRANSACTIONS_CSV)
DEBITS = TXNS[TXNS["direction"] == "debit"].assign(month=lambda d: d["date"].str[:7])
MONTHS = sorted(DEBITS["month"].unique())
DATA_END = TXNS["date"].max()
METRICS_BYTES = config.METRICS_PATH.read_bytes() if config.METRICS_PATH.exists() else b"{}"
try:
    VERIFIED = bool(json.loads(config.BUILD_VERIFICATION_PATH.read_text())["metrics_verified"])
except Exception:
    VERIFIED = False

app = FastAPI(title="PocketSmart AI", version="1.0",
              description="ML computes (classifier, anomaly z-score, forecaster). Gemini only narrates.")

# ---------------------------------------------------------------- latency middleware (in-memory)
LATENCY: dict[str, deque] = defaultdict(lambda: deque(maxlen=10_000))


@app.middleware("http")
async def record_latency(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    route = request.scope.get("route")
    path = getattr(route, "path", None)
    if path and not path.startswith(("/docs", "/openapi", "/redoc")) and path != "/":
        LATENCY[f"{request.method} {path}"].append((time.perf_counter() - t0) * 1000)
    return response


# ---------------------------------------------------------------- endpoints
@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", model_version=f"{_CLF['model_name']}@{MODEL_SHA}", model_sha256=MODEL_SHA,
                          metrics_verified=VERIFIED, data_range=[TXNS["date"].min(), DATA_END],
                          n_train_rows=int(_CLF["n_train_rows"]), uptime_s=round(time.time() - BOOT, 1),
                          llm=gemini.stats())


def categorize_strings(strings: list[str]) -> list[CategorizeItem]:
    proba = PIPE.predict_proba(strings)
    out = []
    for s, p in zip(strings, proba):
        order = np.argsort(-p)[:3]
        conf = float(p[order[0]])
        out.append(CategorizeItem(merchant=s, normalised=normalise(s), category=str(CLASSES[order[0]]),
                                  confidence=round(conf, 4), uncertain=conf < ABSTAIN,
                                  top_3=[LabelProb(label=str(CLASSES[i]), prob=round(float(p[i]), 4)) for i in order]))
    return out


@app.post("/categorize", response_model=CategorizeResponse)
def categorize(req: CategorizeRequest):
    strings = [s.strip()[:300] for s in req.merchants]
    if any(not s for s in strings):
        raise HTTPException(422, "merchant strings must be non-empty")
    return CategorizeResponse(model=_CLF["model_name"], abstain_threshold=ABSTAIN, results=categorize_strings(strings))


@app.get("/summary", response_model=SummaryResponse)
def summary(month: str | None = Query(None, pattern=r"^\d{4}-(0[1-9]|1[0-2])$", description="YYYY-MM; default latest")):
    month = month or MONTHS[-1]
    if month not in MONTHS:
        raise HTTPException(404, f"no data for {month}; available {MONTHS[0]}..{MONTHS[-1]}")
    i = MONTHS.index(month)
    prev = MONTHS[i - 1] if i > 0 else None
    cur = DEBITS[DEBITS["month"] == month].groupby("category")["amount"].sum()
    old = DEBITS[DEBITS["month"] == prev].groupby("category")["amount"].sum() if prev else pd.Series(dtype=float)
    total, prev_total = float(cur.sum()), (float(old.sum()) if prev else None)
    cats = []
    for c in config.CATEGORY_NAMES:
        t, p = float(cur.get(c, 0.0)), float(old.get(c, 0.0))
        cats.append(CategoryTotal(category=c, total=round(t, 2), share_pct=round(100 * t / total, 2) if total else 0.0,
                                  prev_total=round(p, 2), mom_delta=round(t - p, 2),
                                  mom_delta_pct=round(100 * (t - p) / p, 2) if p else None))
    cats.sort(key=lambda x: -x.total)
    return SummaryResponse(
        month=month, prev_month=prev, total_spend=round(total, 2),
        prev_total_spend=None if prev_total is None else round(prev_total, 2),
        mom_delta_pct=round(100 * (total - prev_total) / prev_total, 2) if prev_total else None,
        n_transactions=int((DEBITS["month"] == month).sum()), top_category=cats[0] if total else None,
        categories=cats, available_months=MONTHS)


def _forecast_item(cat: str, history_weeks: int = 26) -> ForecastItem:
    s, h = FORECAST["served"][cat], FORECAST["history"][cat]
    return ForecastItem(**s, history_weeks=h["weeks"][-history_weeks:], history_actual=h["actual"][-history_weeks:])


@app.get("/forecast", response_model=ForecastResponse)
def forecast(category: str | None = Query(None, description="one of the 10 categories; omit for all")):
    if category is None:
        return ForecastResponse(items=[_forecast_item(c) for c in config.CATEGORY_NAMES])
    match = {c.lower(): c for c in config.CATEGORY_NAMES}.get(category.strip().lower())
    if not match:
        raise HTTPException(404, f"unknown category {category!r}; valid: {config.CATEGORY_NAMES}")
    return ForecastResponse(items=[_forecast_item(match)])


@app.get("/anomalies", response_model=AnomaliesResponse)
def anomalies(days: int = Query(30, ge=1, le=365)):
    var, thr = AN_SERVED["variant"], float(AN_SERVED["threshold"])
    start = (pd.Timestamp(DATA_END) - pd.Timedelta(days=days - 1)).strftime("%Y-%m-%d")
    d = AN_SCORED[(AN_SCORED["date"] >= start) & (AN_SCORED[f"z_{var}"].abs() > thr)]
    d = d.sort_values(f"z_{var}", ascending=False)
    items = []
    for r in d.itertuples():
        z = float(getattr(r, f"z_{var}"))
        typical = float(r.mu_raw if var == "raw" else r.typical_amount)
        side = "above" if z > 0 else "below"
        items.append(AnomalyItem(
            txn_id=r.txn_id, date=r.date, merchant=r.merchant_raw, category=r.category, amount=round(r.amount, 2),
            z_score=round(z, 2), typical_amount=round(typical, 2),
            reason=f"₹{r.amount:,.0f} is {abs(z):.1f} standard deviations {side} your {config.ANOMALY_WINDOW_DAYS}-day "
                   f"{r.category} {'average' if var == 'raw' else 'typical amount'} of ₹{typical:,.0f}"))
    return AnomaliesResponse(days=days, window_end=DATA_END, detector=f"{var}-z rolling {config.ANOMALY_WINDOW_DAYS}d",
                             threshold=thr, items=items)


@app.post("/advice", response_model=AdviceResponse)
def advice(req: AdviceRequest):
    # Send only what the narration needs: no chart history, at most 5 anomalies. Fewer numbers,
    # fewer tokens, and fewer ways for the model to go wrong.
    payload = {}
    try:
        if req.summary:
            payload["summary"] = req.summary.model_dump(exclude={"available_months"})
            payload["summary"]["categories"] = payload["summary"]["categories"][:5]
        if req.forecast:
            payload["forecast"] = [f.model_dump(exclude={"history_weeks", "history_actual"}) for f in req.forecast]
        if req.anomalies:
            a = req.anomalies.model_dump()
            a["items"] = a["items"][:5]
            payload["anomalies"] = a
        if req.categorization:
            payload["categorization"] = [c.model_dump() for c in req.categorization]
        out = gemini.narrate(payload)
    except Exception as e:  # noqa: BLE001  /advice must never 500
        out = {"text": gemini.fallback_text(payload), "source": "fallback", "model": gemini.model_name(),
               "reason": f"{type(e).__name__}"}
    return AdviceResponse(**out, llm_stats=gemini.stats())


@app.get("/metrics")
def metrics():
    """results/metrics.json verbatim: every number in the README, live."""
    return Response(content=METRICS_BYTES, media_type="application/json")


@app.get("/metrics/latency", response_model=LatencyResponse)
def latency():
    eps = {k: LatencyStat(count=len(v), p50_ms=round(float(np.percentile(v, 50)), 2),
                          p95_ms=round(float(np.percentile(v, 95)), 2)) for k, v in sorted(LATENCY.items()) if v}
    return LatencyResponse(since_boot_s=round(time.time() - BOOT, 1), endpoints=eps)


# Frontend last, so API routes take precedence. /docs stays available.
app.mount("/", StaticFiles(directory=config.ROOT / "frontend", html=True), name="frontend")
