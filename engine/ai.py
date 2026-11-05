"""
AI POMOČNIK ZA STRATEGIJE (Claude API)
V Streamlit Secrets dodaj:  ANTHROPIC_API_KEY = "sk-ant-..."
(neobvezno)                 ANTHROPIC_MODEL   = "claude-sonnet-5-5"
"""
import json
import os

import requests

from .builder import EXAMPLES, HELP, parse_json

DEFAULT_MODEL = "claude-sonnet-5-5"


def _cfg():
    key = os.environ.get("ANTHROPIC_API_KEY")
    model = os.environ.get("ANTHROPIC_MODEL")
    try:
        import streamlit as st
        key = key or st.secrets.get("ANTHROPIC_API_KEY")
        model = model or st.secrets.get("ANTHROPIC_MODEL")
    except Exception:
        pass
    return key, model or DEFAULT_MODEL


def enabled():
    return bool(_cfg()[0])


SYSTEM = (
    "You translate trading-strategy ideas (usually written in Slovenian) into a JSON spec for a small, safe rule "
    "language. Reply with ONLY one JSON object, no prose, no code fences.\n\n"
    "JSON keys:\n"
    '  "ime": short name (Slovenian)\n'
    '  "opis": 1-2 sentence description in Slovenian, include assumptions you made\n'
    '  "long": entry condition for long (empty string = no longs)\n'
    '  "short": entry condition for short (empty string = no shorts)\n'
    '  "izhod_long" / "izhod_short": optional exit condition (empty = exit only by SL/TP/time)\n'
    '  "samo_prvi": true = only the first bar where the entry condition becomes true\n'
    '  "sl": expression for the DISTANCE from entry to stop loss in PRICE units (e.g. "2*atr(14)", "close*0.03")\n'
    '  "tp_r": take profit as multiple of the SL distance (0 = no TP)\n'
    '  "max_svečk": time exit in bars (0 = none)\n'
    '  "režimi": market regimes where it should run, subset of ["uptrend","downtrend","range"]\n'
    '  "trgi": markets, subset of ["crypto","index","forex","commodity"]\n\n'
    "Conditions are evaluated on the CLOSE of each bar; entry happens at the next bar's open. "
    "Never reference future bars. Language reference:\n"
    + "\n".join(f"  {k}: {v}" for k, v in HELP.items())
    + "\nOperators: + - * /, > >= < <= ==, and/or/not (or Slovenian in/ali/ne). x[k] = value k bars ago.\n"
    "Only these functions exist. If the idea needs something not available, approximate it with these and say so in 'opis'.\n\n"
    "Example spec:\n" + json.dumps(EXAMPLES["EMA prečkanje + ADX"], ensure_ascii=False)
)


def generate(description, current=None, timeframe="1d", market=""):
    key, model = _cfg()
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY ni nastavljen v Secrets")
    user = f"Časovni okvir: {timeframe}. Trg: {market}.\nOpis strategije / želena sprememba:\n{description}"
    if current and (current.get("long") or current.get("short")):
        user += "\n\nTrenutna strategija (spremeni jo glede na opis):\n" + json.dumps(current, ensure_ascii=False)
    r = requests.post("https://api.anthropic.com/v1/messages", timeout=90,
                      headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                               "content-type": "application/json"},
                      json={"model": model, "max_tokens": 1500, "system": SYSTEM,
                            "messages": [{"role": "user", "content": user}]})
    if r.status_code >= 400:
        raise RuntimeError(f"Claude API {r.status_code}: {r.text[:300]}")
    text = "".join(b.get("text", "") for b in r.json().get("content", []) if b.get("type") == "text")
    return parse_json(text)
