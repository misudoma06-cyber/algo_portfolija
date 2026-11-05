"""
TELEGRAM OBVESTILA
Secrets (Streamlit in GitHub): TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
"""
import os

import requests


def _cfg():
    t, c = os.environ.get("TELEGRAM_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    try:
        import streamlit as st
        t = t or st.secrets.get("TELEGRAM_TOKEN")
        c = c or st.secrets.get("TELEGRAM_CHAT_ID")
    except Exception:
        pass
    return t, c


def has_token():
    return bool(_cfg()[0])


def enabled():
    t, c = _cfg()
    return bool(t and c)


def send(text):
    t, c = _cfg()
    if not (t and c):
        raise RuntimeError("TELEGRAM_TOKEN / TELEGRAM_CHAT_ID nista nastavljena")
    r = requests.post(f"https://api.telegram.org/bot{t}/sendMessage", timeout=20,
                      json={"chat_id": c, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True})
    if r.status_code >= 400:
        raise RuntimeError(f"Telegram {r.status_code}: {r.text[:200]}")
    return True


def find_chats():
    t, _ = _cfg()
    if not t:
        raise RuntimeError("TELEGRAM_TOKEN ni nastavljen")
    r = requests.get(f"https://api.telegram.org/bot{t}/getUpdates", timeout=20).json()
    out = {}
    for u in r.get("result", []):
        ch = (u.get("message") or u.get("channel_post") or {}).get("chat")
        if ch:
            out[ch["id"]] = ch.get("first_name") or ch.get("title") or ch.get("username") or ""
    return out
