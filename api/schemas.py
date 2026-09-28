"""Pydantic v2 request/response schemas for every endpoint."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ---- /categorize ----------------------------------------------------------
class CategorizeRequest(BaseModel):
    merchants: list[str] = Field(..., min_length=1, max_length=100,
                                 examples=[["UPI/9876543210/SWIGGY/YESB0001", "ATM WDL 0912 SALEM TN"]])


class LabelProb(BaseModel):
    label: str
    prob: float


class CategorizeItem(BaseModel):
    merchant: str
    normalised: str
    category: str
    confidence: float
    uncertain: bool = Field(..., description="confidence < abstain_threshold: treat the category as a best guess")
    top_3: list[LabelProb]


class CategorizeResponse(BaseModel):
    model: str
    abstain_threshold: float
    results: list[CategorizeItem]


# ---- /summary -------------------------------------------------------------
class CategoryTotal(BaseModel):
    category: str
    total: float
    share_pct: float
    prev_total: float
    mom_delta: float
    mom_delta_pct: float | None


class SummaryResponse(BaseModel):
    month: str
    prev_month: str | None
    total_spend: float
    prev_total_spend: float | None
    mom_delta_pct: float | None
    n_transactions: int
    top_category: CategoryTotal | None
    categories: list[CategoryTotal]
    available_months: list[str]


# ---- /forecast ------------------------------------------------------------
class ForecastItem(BaseModel):
    category: str
    week_start: str
    prediction: float
    lower_80: float
    upper_80: float
    model_used: Literal["naive", "seasonal_naive", "linear", "category_mean"]
    holdout_mae: float = Field(..., description="mean absolute error of model_used on the 8-week holdout (INR)")
    holdout_mape: float | None
    interval_coverage_holdout: float
    weeks_beat_reference: str
    history_weeks: list[str] = []
    history_actual: list[float] = []


class ForecastResponse(BaseModel):
    items: list[ForecastItem]


# ---- /anomalies -----------------------------------------------------------
class AnomalyItem(BaseModel):
    txn_id: str
    date: str
    merchant: str
    category: str
    amount: float
    z_score: float
    typical_amount: float
    reason: str


class AnomaliesResponse(BaseModel):
    days: int
    window_end: str
    detector: str
    threshold: float
    items: list[AnomalyItem]


# ---- /advice --------------------------------------------------------------
class AdviceRequest(BaseModel):
    """Pass the JSON returned by the other endpoints; any subset is fine."""
    summary: SummaryResponse | None = None
    forecast: list[ForecastItem] | None = None
    anomalies: AnomaliesResponse | None = None
    categorization: list[CategorizeItem] | None = None


class AdviceResponse(BaseModel):
    text: str
    source: Literal["llm", "cache", "fallback"]
    model: str
    reason: str | None = None
    llm_stats: dict


# ---- /health --------------------------------------------------------------
class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_version: str
    model_sha256: str
    metrics_verified: bool
    data_range: list[str]
    n_train_rows: int
    uptime_s: float
    llm: dict


class LatencyStat(BaseModel):
    count: int
    p50_ms: float
    p95_ms: float


class LatencyResponse(BaseModel):
    since_boot_s: float
    endpoints: dict[str, LatencyStat]
