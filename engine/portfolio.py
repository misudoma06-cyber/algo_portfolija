"""
PORTFELJ – skupna logika za aplikacijo IN paper trading bota
active_combos()     -> katere (simbol, strategija, režim) kombinacije so aktivne
combo_returns()     -> donosi ene kombinacije (z režimskim filtrom + vol targetingom)
portfolio_returns() -> dnevni donosi celega portfelja
target_positions()  -> današnje ciljne pozicije (za bota)
"""
import pandas as pd

from .backtest import apply_regime_filter
from .regime import regime_series
from .risk import scaled_returns, to_daily, vol_weights
from .strategies import ALL_STRATEGIES


def effective_regime(settings, symbol, auto_regime):
    o = settings.get("overrides", {}).get(symbol, "auto")
    return auto_regime if o == "auto" else o


def active_combos(settings, auto_regimes, only_edge=None):
    """only_edge: množica (simbol, strategija, režim), ki so prestale test – ali None."""
    out = []
    for sym, m in settings["markets"].items():
        reg = effective_regime(settings, sym, auto_regimes.get(sym, "unknown"))
        for name, cls in ALL_STRATEGIES.items():
            if settings["strategies_on"].get(name) and cls().fits(reg, m["trg"]):
                if only_edge is None or (sym, name, reg) in only_edge:
                    out.append((sym, name, reg))
    return out


def combo_signals(df, strategy_name, regime, params=None):
    strat = ALL_STRATEGIES[strategy_name](**(params or {}))
    return apply_regime_filter(strat.generate_signals(df), regime_series(df), regime)


def combo_returns(df, strategy_name, regime, risk, params=None):
    sig = combo_signals(df, strategy_name, regime, params)
    pos = sig.shift(1).fillna(0)
    ret, _ = scaled_returns(df, pos, risk["target_vol"], risk["max_lev"], risk["cost"])
    return ret


def portfolio_returns(data, combos, risk):
    """data: {simbol: df}. Vsaka kombinacija dobi enak delež kapitala."""
    rets = {}
    for sym, name, reg in combos:
        rets[f"{sym} · {name}"] = to_daily(combo_returns(data[sym], name, reg, risk))
    R = pd.DataFrame(rets).fillna(0)
    if R.empty:
        return R, pd.Series(dtype=float)
    return R, R.mean(axis=1)


def target_positions(data, combos, risk):
    """Današnja ciljna izpostavljenost na simbol kot delež kapitala (npr. 0.35 = 35 % long)."""
    n = max(len(combos), 1)
    targets = {}
    for sym, name, reg in combos:
        df = data[sym]
        sig = combo_signals(df, name, reg).iloc[-1]
        w = vol_weights(df, risk["target_vol"], risk["max_lev"]).iloc[-1]
        targets[sym] = targets.get(sym, 0.0) + float(sig) * float(w) / n
    return targets
