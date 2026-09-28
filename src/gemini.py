"""Gemini narration layer. It turns the models' JSON into 2-4 plain-English sentences.

What the LLM DOES: rephrase figures it is given, in plain language.
What it does NOT do: calculate, predict, estimate, or invent any number. Every
number comes from the ML models; the LLM only narrates.

Defences, in order:
1. System prompt forbids computing/predicting/inventing numbers and requires
   hedging whenever an item is marked `uncertain: true`.
2. Number-guard: every number in the reply must match a number in the input
   payload (allowing rounding and fraction->percent). Otherwise the reply is
   rejected and the deterministic template is used instead.
3. Cache: identical payloads (sha256 of canonical JSON + model + prompt version)
   are served from cache/advice_cache.json, with no network call.
4. 5-second hard timeout (thread-level; the SDK socket timeout sits at the API's 10 s minimum) and a
   catch-all: ANY failure returns the deterministic template. narrate() never raises.

CLI:  python -m src.gemini --list-models      (needs GEMINI_API_KEY; verifies the model string)
      python -m src.gemini --demo             (narrates a sample payload; honours GEMINI_FORCE_FAIL)
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import os
import re
import sys
import threading

from dotenv import load_dotenv

from src import config

load_dotenv(config.ROOT / ".env")

SYSTEM_PROMPT = """You are the narration layer of a personal-finance app. You receive JSON produced by
machine-learning models (spending summary, forecasts, anomaly flags). Write 2 to 4 short, plain-English
sentences for the account holder, in second person ("you").

STRICT RULES:
- You MUST NOT calculate, add, subtract, average, estimate, predict, extrapolate or invent any number.
- You may ONLY use numbers that appear in the JSON, copied as given (you may drop decimals from rupee amounts
  and write a fraction such as 0.25 as 25%). If a number you want is not in the JSON, do not state it.
- Do not forecast anything yourself. Only restate forecasts that are in the JSON, and say which model made them.
- If any item has "uncertain": true, you MUST hedge about it explicitly (e.g. "this looks like X, but the
  categoriser isn't confident"). Never present an uncertain categorisation as fact.
- If a forecast includes a holdout MAE, mention that the forecast is typically off by about that much.
- No financial product recommendations, no investment advice. Practical, neutral, non-judgemental tone.
- Write money as rupees with Indian digit grouping and no decimals (e.g. ₹61,234, ₹1,11,062). Write a negative
  change as "down 4.2%" and a positive one as "up 4.2%". Say "standard deviations above normal", not "z-score".
- Never mention JSON field names (like z_score, holdout_mae, mom_delta_pct) or say "JSON".
- Output plain text only: no markdown, no lists, no headings."""

_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_lock = threading.Lock()
_executor = cf.ThreadPoolExecutor(max_workers=2)

STATS = {"llm_calls": 0, "cache_hits": 0, "cache_misses": 0, "fallback_count": 0, "guard_rejections": 0,
         "tokens_in": 0, "tokens_out": 0, "token_counts_approximate": False, "est_cost_inr": 0.0}
CACHE_PATH = config.CACHE_DIR / "advice_cache.json"


def model_name() -> str:
    return os.getenv("GEMINI_MODEL") or config.GEMINI_MODEL_DEFAULT


def _canonical(payload) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def cache_key(payload) -> str:
    return hashlib.sha256(f"{model_name()}|{config.PROMPT_VERSION}|{_canonical(payload)}".encode()).hexdigest()


def _load_cache() -> dict:
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_cache(cache: dict):
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = CACHE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")
        tmp.replace(CACHE_PATH)
    except Exception:
        pass  # a read-only disk must never break /advice


_CACHE = _load_cache()


# --------------------------------------------------------------------------- number guard
def _numbers_in(obj) -> list[float]:
    out = []
    if isinstance(obj, bool) or obj is None:
        return out
    if isinstance(obj, (int, float)):
        return [float(obj)]
    if isinstance(obj, str):
        return [float(m.replace(",", "")) for m in _NUM.findall(obj)]
    if isinstance(obj, dict):
        for k, v in obj.items():
            out += _numbers_in(k) + _numbers_in(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out += _numbers_in(v)
    return out


def _allowed(payload) -> list[float]:
    allowed = set()
    for a in _numbers_in(payload):
        b = abs(a)
        allowed.update({b, round(b), round(b, 1), round(b, 2)})  # sign is spoken as "up"/"down"
        if abs(a) <= 1.5:  # fractions may be spoken as percentages
            allowed.update({round(a * 100), round(a * 100, 1), round(abs(a) * 100), round(abs(a) * 100, 1)})
    return sorted(allowed)


def invented_numbers(text: str, payload) -> list[str]:
    """Numbers in `text` that don't correspond to any number in `payload`."""
    allowed = _allowed(payload)
    bad = []
    for tok in _NUM.findall(text):
        x = abs(float(tok.replace(",", "")))
        if not any(abs(x - c) <= max(0.01, 0.005 * abs(c)) for c in allowed):
            bad.append(tok)
    return bad


# --------------------------------------------------------------------------- fallback
def _inr(x) -> str:
    """Indian digit grouping, e.g. ₹4,04,791 (matches the frontend's en-IN formatting)."""
    n = str(abs(int(round(float(x)))))
    head, tail = n[:-3], n[-3:]
    while len(head) > 2:
        tail, head = head[-2:] + "," + tail, head[:-2]
    return ("-" if float(x) < 0 else "") + "₹" + (head + "," + tail if head else tail)


def fallback_text(payload: dict) -> str:
    """Deterministic narration built only from the payload. Never raises."""
    parts = []
    try:
        s = payload.get("summary") or {}
        if s.get("top_category"):
            top = s["top_category"]
            line = f"In {s.get('month', 'this month')} you spent {_inr(s.get('total_spend', 0))} in total"
            line += f", most of it on {top.get('category')} ({_inr(top.get('total', 0))}, {top.get('share_pct', 0):.1f}% of spend)"
            if s.get("mom_delta_pct") is not None:
                direction = "up" if s["mom_delta_pct"] >= 0 else "down"
                line += f"; total spend is {direction} {abs(s['mom_delta_pct']):.1f}% on the previous month"
            parts.append(line + ".")
        fc = payload.get("forecast") or []
        fc = fc if isinstance(fc, list) else [fc]
        if fc:
            f = max(fc, key=lambda x: x.get("prediction", 0))
            parts.append(f"The {f.get('model_used')} model expects about {_inr(f.get('prediction', 0))} on "
                         f"{f.get('category')} next week (range {_inr(f.get('lower_80', 0))}–{_inr(f.get('upper_80', 0))}; "
                         f"typically off by {_inr(f.get('holdout_mae', 0))}).")
        anp = payload.get("anomalies") or {}
        an = anp.get("items") or []
        if an:
            a = max(an, key=lambda x: x.get("z_score", 0))
            parts.append(f"{anp.get('n_flagged', len(an))} transaction(s) look unusual; the largest is {_inr(a.get('amount', 0))} on "
                         f"{a.get('category')} ({a.get('date')}), {a.get('z_score', 0):.1f} standard deviations above normal.")
        unc = [c for c in (payload.get("categorization") or []) if c.get("uncertain")]
        if unc:
            parts.append(f"The categoriser isn't confident about {len(unc)} merchant string(s), so treat those labels as best guesses.")
    except Exception:
        pass
    return " ".join(parts) or "Not enough data to summarise yet."


# --------------------------------------------------------------------------- LLM call
def _call_gemini(payload: dict) -> tuple[str, int | None, int | None]:
    from google import genai
    from google.genai import types

    # The API rejects server deadlines under 10 s ("Minimum allowed deadline is 10s"), so the SDK timeout
    # is set to that floor only to kill hung sockets. The 5 s user-facing budget is enforced by the
    # thread-level future.result(timeout=...) in narrate().
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"],
                          http_options=types.HttpOptions(timeout=config.GEMINI_HTTP_TIMEOUT_MS))
    cfg = dict(system_instruction=SYSTEM_PROMPT, temperature=0.0, max_output_tokens=400,
               automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))  # no tools used
    level = os.getenv("GEMINI_THINKING_LEVEL", "MINIMAL")  # keep latency inside the 5 s budget
    if level:
        cfg["thinking_config"] = types.ThinkingConfig(thinking_level=level)
    resp = client.models.generate_content(
        model=model_name(), contents="Model output JSON:\n" + _canonical(payload),
        config=types.GenerateContentConfig(**cfg))
    um = getattr(resp, "usage_metadata", None)
    t_in = getattr(um, "prompt_token_count", None)
    t_out = None if um is None else (getattr(um, "candidates_token_count", 0) or 0) + (getattr(um, "thoughts_token_count", 0) or 0)
    return (resp.text or "").strip(), t_in, t_out


def _account(payload: dict, text: str, t_in, t_out):
    if t_in is None:
        STATS["token_counts_approximate"] = True
        t_in, t_out = len(SYSTEM_PROMPT + _canonical(payload)) // 4, len(text) // 4
    STATS["tokens_in"] += t_in
    STATS["tokens_out"] += t_out or 0
    usd = STATS["tokens_in"] / 1e6 * config.FLASH_USD_PER_1M_IN + STATS["tokens_out"] / 1e6 * config.FLASH_USD_PER_1M_OUT
    STATS["est_cost_inr"] = round(usd * config.USD_TO_INR, 4)


def narrate(payload: dict) -> dict:
    """Return {"text", "source": llm|cache|fallback, "model", "reason"}. Never raises."""
    try:
        key = cache_key(payload)
        with _lock:
            hit = _CACHE.get(key)
        if hit:
            STATS["cache_hits"] += 1
            return {"text": hit, "source": "cache", "model": model_name(), "reason": None}
        STATS["cache_misses"] += 1

        if os.getenv("GEMINI_FORCE_FAIL", "0") == "1":
            raise RuntimeError("GEMINI_FORCE_FAIL=1 (simulated outage)")
        if not os.getenv("GEMINI_API_KEY"):
            raise RuntimeError("GEMINI_API_KEY not set")

        STATS["llm_calls"] += 1
        fut = _executor.submit(_call_gemini, payload)
        text, t_in, t_out = fut.result(timeout=config.GEMINI_TIMEOUT_S)
        _account(payload, text, t_in, t_out)
        if not text:
            raise RuntimeError("empty response")
        bad = invented_numbers(text, payload)
        if bad:
            STATS["guard_rejections"] += 1
            raise RuntimeError(f"number-guard rejected reply (numbers not in input: {bad[:5]})")
        with _lock:
            _CACHE[key] = text
            _save_cache(_CACHE)
        return {"text": text, "source": "llm", "model": model_name(), "reason": None}
    except cf.TimeoutError:
        reason = f"timeout after {config.GEMINI_TIMEOUT_S:.0f}s"
    except Exception as e:  # noqa: BLE001  (any failure -> deterministic template)
        reason = f"{type(e).__name__}: {e}"[:200]
    STATS["fallback_count"] += 1
    return {"text": fallback_text(payload), "source": "fallback", "model": model_name(), "reason": reason}


def stats() -> dict:
    return {**STATS, "model": model_name(), "prompt_version": config.PROMPT_VERSION,
            "cost_rates": {"usd_per_1m_in": config.FLASH_USD_PER_1M_IN, "usd_per_1m_out": config.FLASH_USD_PER_1M_OUT,
                           "usd_to_inr": config.USD_TO_INR, "note": "editable constants in src/config.py; verify"}}


SAMPLE = {
    "summary": {"month": "2026-08", "total_spend": 61234.5, "mom_delta_pct": -4.2,
                "top_category": {"category": "Shopping", "total": 18450.0, "share_pct": 30.1}},
    "forecast": [{"category": "Food & Dining", "prediction": 9120.4, "lower_80": 6800.0, "upper_80": 12100.0,
                  "model_used": "linear", "holdout_mae": 1993.0}],
    "anomalies": {"items": [{"date": "2026-08-21", "category": "Health", "amount": 5400.0, "z_score": 4.6}]},
}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    if "--list-models" in sys.argv:
        from google import genai

        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        names = [m.name for m in client.models.list() if "flash" in m.name]
        print("\n".join(names))
        print(f"\nconfigured: {model_name()}  ->  {'FOUND' if any(n.endswith(model_name()) for n in names) else 'NOT FOUND'}")
        return
    out = narrate(SAMPLE)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(stats(), indent=2))


if __name__ == "__main__":
    main()
