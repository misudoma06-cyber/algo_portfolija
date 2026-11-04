"""
TIME-SERIES MOMENTUM (Moskowitz, Ooi, Pedersen)
Long, če je donos zadnjih N svečk pozitiven, short, če negativen.
Pozicija se preveri le vsakih K svečk (manj trejdov, manj stroškov).
Eden najbolj raziskanih edgov na vseh trgih.
"""
import numpy as np
import pandas as pd
from .base import Strategy


class TSMOM(Strategy):
    name = "tsmom"
    label = "Time-series momentum"
    regimes = ["uptrend", "downtrend"]
    markets = ["crypto", "index", "forex", "commodity"]

    def default_params(self):
        return {"lookback": 90, "rebalance": 5}

    def param_grid(self):
        return {"lookback": [30, 60, 90, 120, 180, 250], "rebalance": [1, 5, 10, 20]}

    def generate_signals(self, df):
        p = self.params
        mom = df["close"].pct_change(int(p["lookback"]))
        raw = np.sign(mom).fillna(0)
        if not self.allow_short:
            raw = raw.clip(lower=0)
        k = max(1, int(p["rebalance"]))
        sig = pd.Series(np.nan, index=df.index)
        sig.iloc[::k] = raw.iloc[::k]
        sig = sig.ffill().fillna(0)
        sig.iloc[: int(p["lookback"])] = 0
        return sig
