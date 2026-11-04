"""
RSI(2) MEAN REVERSION (Larry Connors)
Long:  cena nad SMA200 in RSI(2) < vstop_long  -> drži, dokler cena ne zapre nad SMA5
Short: cena pod SMA200 in RSI(2) > vstop_short -> drži, dokler cena ne zapre pod SMA5
Najbolje deluje na indeksih (dnevne svečke).
"""
import numpy as np
import pandas as pd
from .base import Strategy


def rsi(close, n):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


class RSI2(Strategy):
    name = "rsi2"
    label = "RSI(2) mean reversion"
    regimes = ["uptrend", "range", "downtrend"]
    markets = ["index", "commodity"]

    def default_params(self):
        return {"rsi_n": 2, "long_entry": 10, "short_entry": 90, "trend": 200, "exit_ma": 5}

    def param_grid(self):
        return {"long_entry": [5, 10, 15, 20], "short_entry": [80, 85, 90, 95], "exit_ma": [3, 5, 10]}

    def generate_signals(self, df):
        p = self.params
        c = df["close"]
        r = rsi(c, p["rsi_n"]).values
        sma_t = c.rolling(p["trend"]).mean().values
        sma_x = c.rolling(p["exit_ma"]).mean().values
        cv = c.values
        sig = np.zeros(len(c))
        pos = 0
        for i in range(p["trend"], len(c)):
            if pos == 1 and cv[i] > sma_x[i]:
                pos = 0
            elif pos == -1 and cv[i] < sma_x[i]:
                pos = 0
            if pos == 0:
                if cv[i] > sma_t[i] and r[i] < p["long_entry"]:
                    pos = 1
                elif self.allow_short and cv[i] < sma_t[i] and r[i] > p["short_entry"]:
                    pos = -1
            sig[i] = pos
        return pd.Series(sig, index=df.index)
