"""
ALPACA PAPER TRADING – preprost odjemalec (samo paper račun!)
Ključi: ALPACA_KEY in ALPACA_SECRET (Streamlit Secrets ali GitHub Secrets).
"""
import os

import requests

BASE = "https://paper-api.alpaca.markets"

# Yahoo simbol -> Alpaca simbol. Forex in indeksi na Alpaci niso na voljo (None).
DEFAULT_MAP = {
    "BTC-USD": "BTC/USD", "ETH-USD": "ETH/USD", "SOL-USD": "SOL/USD",
    "QQQ": "QQQ", "SPY": "SPY", "GC=F": "GLD",
}


def alpaca_symbol(yahoo_symbol, market_cfg):
    return market_cfg.get("alpaca") or DEFAULT_MAP.get(yahoo_symbol)


def is_crypto(alp_symbol):
    return "/" in alp_symbol


def _keys():
    k, s = os.environ.get("ALPACA_KEY"), os.environ.get("ALPACA_SECRET")
    try:
        import streamlit as st
        k = k or st.secrets.get("ALPACA_KEY")
        s = s or st.secrets.get("ALPACA_SECRET")
    except Exception:
        pass
    return k, s


def enabled():
    k, s = _keys()
    return bool(k and s)


class Alpaca:
    def __init__(self):
        k, s = _keys()
        if not (k and s):
            raise RuntimeError("ALPACA_KEY / ALPACA_SECRET nista nastavljena")
        self.h = {"APCA-API-KEY-ID": k, "APCA-API-SECRET-KEY": s}

    def _req(self, method, path, **kw):
        r = requests.request(method, BASE + path, headers=self.h, timeout=20, **kw)
        if r.status_code >= 400:
            raise RuntimeError(f"Alpaca {method} {path}: {r.status_code} {r.text}")
        return r.json() if r.text else {}

    def account(self):
        return self._req("GET", "/v2/account")

    def positions(self):
        """{alpaca_simbol: qty} – kripto simboli vrnjeni brez '/' (BTCUSD), zato jih normaliziramo."""
        out = {}
        for p in self._req("GET", "/v2/positions"):
            sym = p["symbol"]
            if p.get("asset_class") == "crypto" and "/" not in sym and sym.endswith("USD"):
                sym = sym[:-3] + "/USD"
            out[sym] = {"qty": float(p["qty"]), "value": float(p["market_value"]),
                        "pnl": float(p["unrealized_pl"]), "price": float(p["current_price"])}
        return out

    def close_position(self, symbol):
        return self._req("DELETE", f"/v2/positions/{symbol.replace('/', '')}")

    def order(self, symbol, qty):
        side = "buy" if qty > 0 else "sell"
        q = ("%.6f" % abs(qty)).rstrip("0").rstrip(".")
        body = {"symbol": symbol, "qty": q, "side": side, "type": "market",
                "time_in_force": "gtc" if is_crypto(symbol) else "day"}
        return self._req("POST", "/v2/orders", json=body)

    def history(self, period="3M"):
        return self._req("GET", "/v2/account/portfolio/history",
                         params={"period": period, "timeframe": "1D"})
