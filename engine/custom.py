"""
🧩 STRATEGIJE IZ GRADITELJA -> pravi razred Strategy (portfelj, ocena edga, backtest, paper bot)
Pozicija = čas, ko je trejd odprt (vstop, SL/TP/časovni izhod/izhodni pogoj kot v Lab-u).
Opomba: portfeljski backtest računa zaprtje-na-zaprtje, zato je SL med svečo tu le približen –
natančen rezultat SL/TP vidiš v zavihku Lab.
"""
import numpy as np
import pandas as pd

from .strategies import ALL_STRATEGIES
from .strategies.base import Strategy

ALL_REGIMES = ["uptrend", "downtrend", "range"]
ALL_MARKETS = ["crypto", "index", "forex", "commodity"]


def make_class(name, spec):
    from .builder import spec_signals
    from .tradelab import simulate

    def generate_signals(self, df):
        sigs = spec_signals(df, spec)
        tr = simulate(df, sigs, cost=0.0)
        pos = np.zeros(len(df))
        for i, x, sd in zip(tr["_i"], tr["_x"], tr["smer"]):
            pos[int(i):int(x)] = 1 if sd == "long" else -1   # signal i..x-1 -> pozicija i+1..x
        return pd.Series(pos, index=df.index)

    return type("Custom_" + "".join(ch for ch in name if ch.isalnum()), (Strategy,), {
        "name": "custom:" + name,
        "label": "🧩 " + name,
        "regimes": list(spec.get("režimi") or ALL_REGIMES),
        "markets": list(spec.get("trgi") or ALL_MARKETS),
        "__doc__": "\n" + (spec.get("opis") or "Strategija iz graditelja."),
        "generate_signals": generate_signals,
    })


def register_custom(custom):
    """Shranjene strategije iz graditelja doda v ALL_STRATEGIES (stare odstrani)."""
    for k in [k for k in ALL_STRATEGIES if k.startswith("custom:")]:
        ALL_STRATEGIES.pop(k)
    for nm, spec in (custom or {}).items():
        try:
            ALL_STRATEGIES["custom:" + nm] = make_class(nm, spec)
        except Exception:
            pass
    return ALL_STRATEGIES
