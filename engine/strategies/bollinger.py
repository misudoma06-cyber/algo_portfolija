"""
BOLLINGER MEAN REVERSION
Long:  zaprtje pod spodnjim pasom (SMA N - k*std)  -> izhod na SMA
Short: zaprtje nad zgornjim pasom (SMA N + k*std)  -> izhod na SMA
Namenjena trgom brez trenda (range) – npr. forex, USD indeks.
"""
import numpy as np
import pandas as pd
from .base import Strategy


class BollingerMR(Strategy):
    name = "bollinger_mr"
    label = "Bollinger mean reversion"
    regimes = ["range"]
    markets = ["forex", "index", "commodity", "crypto"]

    def default_params(self):
        return {"n": 20, "k": 2.0}

    def param_grid(self):
        return {"n": [10, 15, 20, 30, 40], "k": [1.5, 2.0, 2.5, 3.0]}

    def generate_signals(self, df):
        p = self.params
        n = int(p["n"])
        c = df["close"]
        ma = c.rolling(n).mean()
        sd = c.rolling(n).std()
        up, lo = (ma + p["k"] * sd).values, (ma - p["k"] * sd).values
        cv, mv = c.values, ma.values
        sig = np.zeros(len(c))
        pos = 0
        for i in range(n, len(c)):
            if pos == 1 and cv[i] >= mv[i]:
                pos = 0
            elif pos == -1 and cv[i] <= mv[i]:
                pos = 0
            if pos == 0:
                if cv[i] < lo[i]:
                    pos = 1
                elif self.allow_short and cv[i] > up[i]:
                    pos = -1
            sig[i] = pos
        return pd.Series(sig, index=df.index)
