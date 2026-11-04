"""
KNJIŽNICA STRATEGIJ – nove strategije dodaš tukaj.
"""
from .base import Strategy, REGIMES
from .crypto_trend import CryptoTrend
from .rsi2 import RSI2
from .donchian import Donchian
from .tsmom import TSMOM
from .bollinger import BollingerMR

ALL_STRATEGIES = {cls.name: cls for cls in [CryptoTrend, RSI2, Donchian, TSMOM, BollingerMR]}
