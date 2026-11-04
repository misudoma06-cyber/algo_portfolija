"""
CRYPTO TREND (osnovna verzija)
Long:  EMA hitra > EMA počasna in cena nad EMA trend
Short: EMA hitra < EMA počasna in cena pod EMA trend
Sicer: brez pozicije
"""
import pandas as pd
from .base import Strategy


class CryptoTrend(Strategy):
    name = "crypto_trend"
    label = "Crypto trend (EMA)"
    regimes = ["uptrend", "downtrend"]
    markets = ["crypto"]

    def default_params(self):
        return {"fast": 20, "slow": 50, "trend": 100}

    def param_grid(self):
        return {"fast": [10, 20, 30], "slow": [40, 50, 80], "trend": [100, 150, 200]}

    def generate_signals(self, df):
        p = self.params
        c = df["close"]
        fast = c.ewm(span=p["fast"], adjust=False).mean()
        slow = c.ewm(span=p["slow"], adjust=False).mean()
        trend = c.ewm(span=p["trend"], adjust=False).mean()

        sig = pd.Series(0, index=df.index)
        sig[(fast > slow) & (c > trend)] = 1
        if self.allow_short:
            sig[(fast < slow) & (c < trend)] = -1
        sig.iloc[: p["trend"]] = 0  # ogrevanje indikatorjev
        return sig
