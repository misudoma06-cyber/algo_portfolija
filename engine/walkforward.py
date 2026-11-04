"""
WALK-FORWARD TEST – zaščita pred overfittingom
1) Podatke razdeli: prvih X % = učenje (in-sample), ostalo = test (out-of-sample).
2) Na učnem delu preizkusi VSE kombinacije parametrov iz param_grid().
3) Najboljše parametre preveri na testnem delu, ki ga optimizacija ni videla.
4) Stabilnost: kolikšen delež VSEH kombinacij parametrov je na testu dobičkonosen.
   Robustna strategija deluje pri večini parametrov, ne le pri enem "srečnem".
"""
import itertools

import numpy as np
import pandas as pd

from .backtest import apply_regime_filter, run_backtest
from .regime import regime_series
from .strategies import ALL_STRATEGIES


def _score(m):
    # Sharpe kaznovan, če je trejdov premalo
    return m["Sharpe"] if m["Trejdov"] >= 10 else -99


def walk_forward(df, strategy_name, regime="vsi", split=0.7, cost=0.001, max_combos=60):
    cls = ALL_STRATEGIES[strategy_name]
    grid = cls().param_grid()
    if not grid:
        return None
    keys = list(grid)
    combos = list(itertools.product(*grid.values()))
    if len(combos) > max_combos:
        rng = np.random.default_rng(0)
        combos = [combos[i] for i in rng.choice(len(combos), max_combos, replace=False)]

    regs = regime_series(df)
    cut = df.index[int(len(df) * split)]
    rows = []
    for vals in combos:
        params = dict(zip(keys, vals))
        sig = cls(**params).generate_signals(df)   # celotni podatki – indikatorji se ogrejejo
        if regime != "vsi":
            sig = apply_regime_filter(sig, regs, regime)
        ins = run_backtest(df.loc[:cut], sig.loc[:cut], cost)["metrics"]
        oos = run_backtest(df.loc[cut:], sig.loc[cut:], cost)["metrics"]
        rows.append({**params,
                     "IS Sharpe": ins["Sharpe"], "IS donos %": ins["Skupni donos %"], "IS trejdov": ins["Trejdov"],
                     "OOS Sharpe": oos["Sharpe"], "OOS donos %": oos["Skupni donos %"], "OOS trejdov": oos["Trejdov"],
                     "_score": _score(ins)})
    res = pd.DataFrame(rows).sort_values("_score", ascending=False)
    best = res.iloc[0]
    default = cls().params
    dsig = cls().generate_signals(df)
    if regime != "vsi":
        dsig = apply_regime_filter(dsig, regs, regime)
    d_oos = run_backtest(df.loc[cut:], dsig.loc[cut:], cost)["metrics"]

    stable = (res["OOS donos %"] > 0).mean() * 100
    decay = (best["OOS Sharpe"] / best["IS Sharpe"]) if best["IS Sharpe"] > 0 else np.nan
    if stable >= 70 and best["OOS Sharpe"] > 0.5:
        verdict = "✅ robustna"
    elif stable >= 50 and best["OOS Sharpe"] > 0:
        verdict = "🟡 delno robustna"
    else:
        verdict = "❌ verjetno overfit / ni edga"
    return {
        "table": res.drop(columns="_score"),
        "best_params": {k: type(grid[k][0])(best[k]) for k in keys},
        "best_is_sharpe": best["IS Sharpe"], "best_oos_sharpe": best["OOS Sharpe"],
        "best_oos_return": best["OOS donos %"],
        "default_params": {k: default[k] for k in keys if k in default},
        "default_oos": d_oos,
        "stability_%": round(stable, 1),
        "decay": None if np.isnan(decay) else round(float(decay), 2),
        "split_date": cut,
        "verdict": verdict,
    }
