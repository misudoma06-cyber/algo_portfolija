"""
DONCHIAN BREAKOUT (Turtle stil)
Long:  zaprtje nad najvišjo ceno zadnjih N svečk  -> izhod pod najnižjo zadnjih M
Short: zaprtje pod najnižjo ceno zadnjih N svečk  -> izhod nad najvišjo zadnjih M
Klasična trend-following strategija za vse trge.
"""
import numpy as np
import pandas as pd
from .base import Strategy


class Donchian(Strategy):
    name = "donchian"
    label = "Donchian breakout"
    regimes = ["uptrend", "downtrend"]
    markets = ["crypto", "index", "forex", "commodity"]

    def default_params(self):
        return {"entry": 55, "exit": 20}

    def param_grid(self):
        return {"entry": [20, 40, 55, 80, 100], "exit": [10, 20, 30]}

    def generate_signals(self, df):
        p = self.params
        hi_e = df["high"].rolling(p["entry"]).max().shift(1).values
        lo_e = df["low"].rolling(p["entry"]).min().shift(1).values
        hi_x = df["high"].rolling(p["exit"]).max().shift(1).values
        lo_x = df["low"].rolling(p["exit"]).min().shift(1).values
        c = df["close"].values
        sig = np.zeros(len(c))
        pos = 0
        for i in range(p["entry"] + 1, len(c)):
            if pos == 1 and c[i] < lo_x[i]:
                pos = 0
            elif pos == -1 and c[i] > hi_x[i]:
                pos = 0
            if pos == 0:
                if c[i] > hi_e[i]:
                    pos = 1
                elif self.allow_short and c[i] < lo_e[i]:
                    pos = -1
            sig[i] = pos
        return pd.Series(sig, index=df.index)
