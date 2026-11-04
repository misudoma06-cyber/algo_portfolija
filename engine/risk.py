"""
RISK MANAGER
1) Volatility targeting: velikost pozicije se prilagaja volatilnosti trga.
   Ko je trg miren, večja pozicija; ko je divji, manjša. Vsaka strategija tako
   prispeva približno enako tveganje, ne glede na to, ali trguje BTC ali EUR/USD.
2) FTMO simulacija: na zgodovinskih dnevnih donosih portfelja začne challenge
   vsakih nekaj dni in preveri, ali bi dosegel cilj, preden bi kršil pravila.
"""
import numpy as np
import pandas as pd

# FTMO 2-step (standardni račun) – pravila v deležih računa
FTMO_RULES = {
    "faza 1": {"target": 0.10, "max_loss": 0.10, "daily_loss": 0.05, "min_days": 4},
    "faza 2": {"target": 0.05, "max_loss": 0.10, "daily_loss": 0.05, "min_days": 4},
}


def to_daily(returns):
    """Donosi poljubnega časovnega okvira -> dnevni donosi."""
    return (1 + returns).resample("1D").prod() - 1


def vol_weights(df, target_vol=0.15, max_lev=3.0, lookback_days=20):
    """Utež pozicije za vsako svečko = ciljna letna vol / dejanska letna vol trga.
    Izračunano samo iz preteklih podatkov (shift), brez gledanja v prihodnost."""
    daily = df["close"].resample("1D").last().dropna()
    dr = daily.pct_change()
    span_days = max((daily.index[-1] - daily.index[0]).days, 1)
    per_year = len(daily) / span_days * 365        # ~252 delnice, ~365 kripto
    vol = dr.ewm(span=lookback_days, min_periods=10).std() * np.sqrt(per_year)
    w = (target_vol / vol).clip(upper=max_lev).shift(1)
    return w.reindex(df.index, method="ffill").fillna(0)


def scaled_returns(df, positions, target_vol=0.15, max_lev=3.0, cost=0.001):
    """Donosi strategije z volatility targetingom (stroški sorazmerni z velikostjo)."""
    w = vol_weights(df, target_vol, max_lev)
    pos = positions * w
    ret = df["close"].pct_change().fillna(0)
    turnover = pos.diff().abs().fillna(pos.abs())
    return pos * ret - turnover * cost, pos


def ftmo_sim(daily_ret, phase="faza 1", start_every=5, max_days=90, rules=None):
    """Za vsak začetni dan simulira challenge.
    Rezultat: PASS (cilj dosežen), FAIL (kršeno pravilo), TIMEOUT (ni dosegel v max_days)."""
    r = rules or FTMO_RULES[phase]
    vals = daily_ret.fillna(0).values
    idx = daily_ret.index
    out = []
    for s in range(0, len(vals) - 5, start_every):
        bal = 1.0
        result, days, worst = "TIMEOUT", 0, 0.0
        trading_days = 0
        for d in range(s, min(s + max_days, len(vals))):
            start_day_bal = bal
            bal *= 1 + vals[d]
            days += 1
            trading_days += vals[d] != 0
            worst = min(worst, bal - 1)
            if (bal - start_day_bal) / 1.0 <= -r["daily_loss"]:
                result = "FAIL (dnevna izguba)"
                break
            if bal <= 1 - r["max_loss"]:
                result = "FAIL (max izguba)"
                break
            if bal >= 1 + r["target"] and trading_days >= r["min_days"]:
                result = "PASS"
                break
        out.append({"začetek": idx[s], "rezultat": result, "dni": days,
                    "končno_%": round((bal - 1) * 100, 2), "najslabše_%": round(worst * 100, 2)})
    res = pd.DataFrame(out)
    if res.empty:
        return res, {}
    n = len(res)
    summary = {
        "Challengov": n,
        "PASS %": round((res["rezultat"] == "PASS").mean() * 100, 1),
        "FAIL %": round(res["rezultat"].str.startswith("FAIL").mean() * 100, 1),
        "TIMEOUT %": round((res["rezultat"] == "TIMEOUT").mean() * 100, 1),
        "Mediana dni do PASS": int(res.loc[res["rezultat"] == "PASS", "dni"].median())
        if (res["rezultat"] == "PASS").any() else None,
    }
    return res, summary
