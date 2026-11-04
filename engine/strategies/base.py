"""
KALUP ZA VSE STRATEGIJE
Vsaka nova strategija mora:
  - imeti ime (name)
  - povedati, v katerih režimih deluje (regimes)
  - povedati, na katerih trgih deluje (markets)
  - imeti funkcijo generate_signals(df), ki vrne pozicijo za vsako svečko:
        1 = long, -1 = short, 0 = brez pozicije
Signal na svečki t se izvede šele na naslednji svečki (brez goljufanja s prihodnostjo).
"""
import pandas as pd

# Dovoljeni režimi (te uporabljaš v config.py)
REGIMES = ["uptrend", "downtrend", "range", "high_vol", "low_vol"]


class Strategy:
    name = "neimenovana"
    label = "Neimenovana"
    regimes = []      # npr. ["uptrend", "downtrend"]
    markets = []      # npr. ["crypto"], ["index"], ["forex"]
    allow_short = True

    def __init__(self, **params):
        # parametri strategije (npr. dolžine EMA) – lahko jih spremeniš ob klicu
        self.params = {**self.default_params(), **params}

    def default_params(self):
        return {}

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError("Strategija mora imeti generate_signals()")

    def fits(self, regime: str, market: str) -> bool:
        """Ali je strategija primerna za dani režim in trg?"""
        return regime in self.regimes and market in self.markets

    def __repr__(self):
        return f"{self.name} {self.params} | režimi={self.regimes} | trgi={self.markets}"
