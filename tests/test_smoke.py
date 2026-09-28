"""Six smoke tests: the API boots, every endpoint answers, and /advice survives a dead LLM.

Run:  python -m pytest tests -q      (needs `python -m src.evaluate` to have built models/)
"""
import time

import pytest
from fastapi.testclient import TestClient

from api.main import app
from src import gemini

client = TestClient(app)


def test_health_and_metrics():
    h = client.get("/health").json()
    assert h["status"] == "ok" and len(h["model_sha256"]) == 12
    m = client.get("/metrics").json()
    assert {"classifier", "anomaly", "forecast", "data"} <= m.keys()


def test_categorize_messy_string():
    r = client.post("/categorize", json={"merchants": ["upi/ref9921/SWIGY-INSTAMART/icic", "ATM WDL 0912 SALEM TN"]})
    assert r.status_code == 200
    res = r.json()["results"]
    assert res[1]["category"] == "Miscellaneous"
    assert len(res[0]["top_3"]) == 3 and isinstance(res[0]["uncertain"], bool)


def test_summary_and_bad_month():
    s = client.get("/summary").json()
    assert abs(sum(c["share_pct"] for c in s["categories"]) - 100) < 0.1
    assert client.get("/summary?month=2019-01").status_code == 404
    assert client.get("/summary?month=banana").status_code == 422


def test_forecast_exposes_holdout_error_and_interval():
    items = client.get("/forecast", params={"category": "Food & Dining"}).json()["items"]
    f = items[0]
    assert f["holdout_mae"] > 0 and f["lower_80"] <= f["prediction"] <= f["upper_80"]
    assert 0 <= f["interval_coverage_holdout"] <= 1
    assert client.get("/forecast?category=Crypto").status_code == 404


def test_anomalies_and_latency():
    a = client.get("/anomalies?days=90").json()
    assert all(abs(i["z_score"]) > a["threshold"] for i in a["items"])
    lat = client.get("/metrics/latency").json()["endpoints"]
    assert any(k.startswith("GET /anomalies") for k in lat)


@pytest.mark.parametrize("mode", ["force_fail", "timeout"])
def test_advice_never_500_when_llm_is_down(monkeypatch, mode):
    body = {"summary": client.get("/summary").json(), "forecast": client.get("/forecast").json()["items"][:2],
            "anomalies": client.get("/anomalies").json()}
    gemini._CACHE.clear()
    if mode == "force_fail":
        monkeypatch.setenv("GEMINI_FORCE_FAIL", "1")
    else:  # a hung network: the call never returns within the 5 s budget
        monkeypatch.setenv("GEMINI_FORCE_FAIL", "0")
        monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
        monkeypatch.setattr(gemini, "_call_gemini", lambda p: time.sleep(8))
    t0 = time.time()
    r = client.post("/advice", json=body)
    assert r.status_code == 200
    out = r.json()
    assert out["source"] == "fallback" and len(out["text"]) > 20
    assert time.time() - t0 < 7  # the timeout fired; we didn't wait for the hung call
    assert gemini.invented_numbers(out["text"], body) == []
