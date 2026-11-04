"""
BACKTEST + METRIKE + TEST NAKLJUČNIH TREJDOV
"""
import numpy as np
import pandas as pd


def bars_per_year(index):
    step = pd.Series(index).diff().median()
    # delnice/indeksi trgujejo ~252 dni na leto, kripto 365 – ocenimo iz podatkov
    span_days = (index[-1] - index[0]).days or 1
    return len(index) / span_days * 365 if step >= pd.Timedelta(hours=20) else pd.Timedelta(days=365) / step


def apply_regime_filter(signals, regimes, regime):
    """Strategija trguje samo, ko je trg v izbranem režimu.
    uptrend -> samo long, downtrend -> samo short, range -> oboje."""
    s = signals.where(regimes == regime, 0)
    if regime == "uptrend":
        s = s.clip(lower=0)
    elif regime == "downtrend":
        s = s.clip(upper=0)
    return s


def _trades(pos, strat_ret):
    trades = []
    p, r = pos.values, strat_ret.values
    i, n = 0, len(p)
    while i < n:
        if p[i] != 0:
            d, j = p[i], i
            while j < n and p[j] == d:
                j += 1
            trades.append({"vstop": pos.index[i], "izstop": pos.index[min(j, n - 1)],
                           "smer": "long" if d > 0 else "short", "svečk": j - i,
                           "donos_%": round((np.prod(1 + r[i:j]) - 1) * 100, 2)})
            i = j
        else:
            i += 1
    return pd.DataFrame(trades)


def metrics(strat_ret, equity, trades, bpy):
    total = equity.iloc[-1] - 1
    years = len(equity) / bpy
    cagr = equity.iloc[-1] ** (1 / years) - 1 if years > 0 and equity.iloc[-1] > 0 else -1
    dd = (equity / equity.cummax() - 1).min()
    sd = strat_ret.std()
    sharpe = strat_ret.mean() / sd * np.sqrt(bpy) if sd > 0 else 0
    n = len(trades)
    if n:
        r = trades["donos_%"]
        win = (r > 0).mean()
        losses = -r[r < 0].sum()
        pf = r[r > 0].sum() / losses if losses > 0 else np.inf
    else:
        win, pf = 0, 0
    return {
        "Skupni donos %": round(total * 100, 2),
        "Letni donos %": round(cagr * 100, 2),
        "Mesečno povp. %": round(((1 + cagr) ** (1 / 12) - 1) * 100, 2) if cagr > -1 else -100,
        "Max drawdown %": round(dd * 100, 2),
        "Sharpe": round(float(sharpe), 2),
        "Trejdov": n,
        "Win rate %": round(win * 100, 1),
        "Profit factor": round(float(pf), 2),
    }


def run_backtest(df, signals, cost=0.001):
    pos = signals.shift(1).fillna(0)           # signal t -> izvedba t+1
    ret = df["close"].pct_change().fillna(0)
    turnover = pos.diff().abs().fillna(pos.abs())
    strat = pos * ret - turnover * cost
    eq = (1 + strat).cumprod()
    tr = _trades(pos, strat)
    return {"pos": pos, "returns": strat, "equity": eq, "trades": tr,
            "metrics": metrics(strat, eq, tr, bars_per_year(df.index)), "cost": cost}


def random_test(df, result, n_random=1000, seed=42, allowed=None):
    """Primerja strategijo z n naključnimi strategijami z enakim številom trejdov,
    enakimi dolžinami in enakim deležem long/short.
    allowed: maska svečk, kjer je bil trejd sploh mogoč (npr. samo v izbranem režimu)."""
    tr = result["trades"]
    if tr.empty:
        return None
    rng = np.random.default_rng(seed)
    n = len(df)
    ret = df["close"].pct_change().fillna(0).values
    lengths = tr["svečk"].values
    long_share = (tr["smer"] == "long").mean()
    cost = result["cost"]
    starts = np.arange(1, n) if allowed is None else np.flatnonzero(allowed.values)
    starts = starts[starts < n - 1]
    if len(starts) == 0:
        return None

    finals = np.empty(n_random)
    for k in range(n_random):
        pos = np.zeros(n)
        for L in lengths:
            s = rng.choice(starts)
            pos[s:s + L] = 1 if rng.random() < long_share else -1
        turnover = np.abs(np.diff(pos, prepend=0))
        finals[k] = np.prod(1 + pos * ret - turnover * cost) - 1

    strat_total = result["equity"].iloc[-1] - 1
    pct = float((finals < strat_total).mean() * 100)
    if len(tr) < 20:
        verdict = "⚪ premalo trejdov"
    else:
        verdict = "✅ edge" if pct >= 95 else "🟡 obetavno" if pct >= 80 else "❌ ni edga"
    return {"finals": finals, "percentile": round(pct, 1), "verdict": verdict,
            "median_%": round(float(np.median(finals)) * 100, 2),
            "top5_%": round(float(np.percentile(finals, 95)) * 100, 2)}
