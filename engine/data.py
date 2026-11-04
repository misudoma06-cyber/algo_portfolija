"""
PODATKI (yfinance)
Omejitve yfinance: 1h svečke največ ~730 dni nazaj, 1d svečke poljubno.
"""
import pandas as pd

PERIODS = {"1h": "720d", "1d": "5y"}


def get_data(symbol, interval="1d", period=None):
    import yfinance as yf
    period = period or PERIODS.get(interval, "5y")
    df = yf.download(symbol, interval=interval, period=period, progress=False, auto_adjust=True)
    if df is None or df.empty:
        raise ValueError(f"Ni podatkov za {symbol}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df
