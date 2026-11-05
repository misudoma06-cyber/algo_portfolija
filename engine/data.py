"""
PODATKI (yfinance)
Omejitve yfinance: 15m/30m ~60 dni nazaj, 1h ~730 dni, 1d poljubno.
4h se sestavi iz 1h svečk. Vsi časi so v UTC.
"""
import pandas as pd

PERIODS = {"15m": "59d", "30m": "59d", "1h": "720d", "4h": "720d", "1d": "10y"}
INTERVALS = ["1d", "4h", "1h", "30m", "15m"]


def get_data(symbol, interval="1d", period=None):
    import yfinance as yf
    src = "1h" if interval == "4h" else interval
    period = period or PERIODS.get(interval, "5y")
    df = yf.download(symbol, interval=src, period=period, progress=False, auto_adjust=True)
    if df is None or df.empty:
        raise ValueError(f"Ni podatkov za {symbol}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    idx = pd.to_datetime(df.index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    df.index = idx
    if interval == "4h":
        df = df.resample("4h").agg({"open": "first", "high": "max", "low": "min",
                                    "close": "last", "volume": "sum"}).dropna()
    return df
