"""
ZAZNAVANJE REŽIMA (na dnevnih svečkah)
- ADX < 20                                  -> range
- cena > EMA200 in EMA50 > EMA200           -> uptrend
- cena < EMA200 in EMA50 < EMA200           -> downtrend
- sicer                                     -> range
regime_series(df) -> režim za vsako svečko v zgodovini (brez gledanja v prihodnost)
detect(df)        -> trenutni režim + dodatni podatki za zavihek "Trgi"
"""
import numpy as np
import pandas as pd

REGIME_LABELS = {"uptrend": "🟢 uptrend", "downtrend": "🔴 downtrend", "range": "🟡 range"}


def _adx(df, n=14):
    h, l, c = df["high"], df["low"], df["close"]
    up, dn = h.diff(), -l.diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / n, adjust=False).mean()
    pdi = 100 * plus_dm.ewm(alpha=1 / n, adjust=False).mean() / atr
    mdi = 100 * minus_dm.ewm(alpha=1 / n, adjust=False).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean(), atr


def _to_daily(df):
    step = pd.Series(df.index).diff().median()
    if step >= pd.Timedelta(hours=20):
        return df
    return df.resample("1D").agg({"open": "first", "high": "max", "low": "min",
                                  "close": "last", "volume": "sum"}).dropna()


def _daily_regimes(d):
    c = d["close"]
    e50 = c.ewm(span=50, adjust=False).mean()
    e200 = c.ewm(span=200, adjust=False).mean()
    a, _ = _adx(d)
    reg = np.where(a < 20, "range",
          np.where((c > e200) & (e50 > e200), "uptrend",
          np.where((c < e200) & (e50 < e200), "downtrend", "range")))
    reg = pd.Series(reg, index=d.index)
    reg.iloc[:200] = "unknown"   # premalo zgodovine
    return reg


def regime_series(df):
    """Režim za vsako svečko. Režim dneva D je izračunan iz podatkov do D-1."""
    d = _to_daily(df)
    reg = _daily_regimes(d).shift(1).fillna("unknown")
    return reg.reindex(df.index, method="ffill").fillna("unknown")


def detect(df):
    d = _to_daily(df)
    c = d["close"]
    reg = _daily_regimes(d)
    a, atr = _adx(d)
    e200 = c.ewm(span=200, adjust=False).mean()
    atr_pct = atr / c
    hist = atr_pct.iloc[-252:]
    vol_rank = (hist < atr_pct.iloc[-1]).mean() * 100
    vol = "visoka" if vol_rank > 75 else "nizka" if vol_rank < 25 else "normalna"

    def chg(days):
        return round((c.iloc[-1] / c.iloc[-days - 1] - 1) * 100, 1) if len(c) > days else None

    # koliko dni že traja trenutni režim
    r = reg.values
    days = 1
    while days < len(r) and r[-days - 1] == r[-1]:
        days += 1

    return {
        "regime": reg.iloc[-1],
        "dni_v_rezimu": days,
        "adx": round(float(a.iloc[-1]), 1),
        "volatilnost": vol,
        "cena": float(c.iloc[-1]),
        "1m_%": chg(21),
        "3m_%": chg(63),
        "nad_EMA200": bool(c.iloc[-1] > e200.iloc[-1]),
    }
