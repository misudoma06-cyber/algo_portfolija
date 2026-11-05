"""
TRADE LAB
1) Kalkulator premika (zigzag): trajanje, velikost, percentili, verjetnost nadaljevanja, cilji cene
2) Backtest posameznih trejdov s pravim SL/TP MED svečo (high/low) + test naključnih trejdov
3) Izboljšave: MAE/MFE, rezultat po režimu, po dnevih/urah, mreža SL/TP
Strategija je samo funkcija, ki vrne vstope (tabela). Svojo dodaš v LAB_STRATEGIES.

Pravila simulacije:
- vstop na odprtju naslednje sveče (ali na zaprtju signalne, po izbiri)
- če sveča zadene SL in TP hkrati -> šteje SL (konzervativno)
- če cena odpre čez SL/TP (gap) -> izhod po odprtju
- trailing / break-even se premakneta šele na zaprtju sveče (brez gledanja v prihodnost)
- en trejd naenkrat na trg
"""
import numpy as np
import pandas as pd

from .regime import _adx, regime_series
from .strategies import ALL_STRATEGIES

DAYS = ["pon", "tor", "sre", "čet", "pet", "sob", "ned"]
REG_ALL = ["uptrend", "downtrend", "range", "unknown"]
SIG_COLS = ["i", "side", "sl_pct", "tp_pct", "max_bars", "exit_i"]
NO_LIMIT = 10 ** 9


def _atr(df, n=14):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def _empty():
    return pd.DataFrame(columns=SIG_COLS)


def period_mask(index, od=None, do=None):
    m = np.ones(len(index), bool)
    if od:
        m &= index >= pd.Timestamp(od)
    if do:
        m &= index < pd.Timestamp(do) + pd.Timedelta(days=1)
    return m


# =====================================================================
# STRATEGIJE – vsaka vrne tabelo vstopov:
#   i        = indeks sveče s signalom (na njenem zaprtju)
#   side     = 1 long / -1 short
#   sl_pct   = razdalja SL v deležu cene (0.05 = 5 %)
#   tp_pct   = razdalja TP v deležu cene (NaN = brez TP)
#   max_bars = časovni izhod (število svečk)
#   exit_i   = (neobvezno) sveča, na kateri strategija sama zapre
# =====================================================================
def strat_breakout_ab(df, p):
    c = df["close"]
    sig = c.pct_change().rolling(int(p["sigma_n"])).std()
    prev_hi = c.rolling(int(p["n"])).max().shift(1)
    brk = c > prev_hi
    first = brk & ~brk.shift(1, fill_value=False)
    strong = (c / prev_hi - 1) >= p["strength"] * sig
    hn = int(p["high_n"])
    near = c >= (1 - p["max_below_high"]) * c.rolling(hn, min_periods=hn).max()
    ok = (first & strong & near & sig.notna()).values
    R = (sig * np.sqrt(p["r_bars"])).values
    pos = np.flatnonzero(ok)
    if len(pos) == 0:
        return _empty()
    return pd.DataFrame({"i": pos, "side": 1, "sl_pct": R[pos], "tp_pct": p["tp_r"] * R[pos],
                         "max_bars": int(p["max_bars"]), "exit_i": np.nan})


def _from_position(name):
    """Obstoječo strategijo iz knjižnice (pozicija 1/0/-1) pretvori v vstope z SL/TP."""
    def f(df, p):
        s = ALL_STRATEGIES[name]().generate_signals(df).values
        atr_pct = (_atr(df) / df["close"]).values
        n, rows = len(s), []
        for i in range(1, n):
            if s[i] != 0 and s[i] != s[i - 1]:
                j = i + 1
                while j < n and s[j] == s[i]:
                    j += 1
                sl = p["sl_atr"] * atr_pct[i]
                rows.append({"i": i, "side": int(s[i]), "sl_pct": sl,
                             "tp_pct": p["tp_r"] * sl if p["tp_r"] > 0 else np.nan,
                             "max_bars": int(p["max_bars"]) if p["max_bars"] > 0 else NO_LIMIT,
                             "exit_i": j if j < n else np.nan})
        return pd.DataFrame(rows) if rows else _empty()
    return f


LAB_STRATEGIES = {
    "breakout_ab": {
        "label": "Preboj A+B (kripto, samo long)",
        "func": strat_breakout_ab,
        "params": {"n": 20, "strength": 0.51, "high_n": 365, "max_below_high": 0.28,
                   "sigma_n": 20, "r_bars": 10, "tp_r": 2.0, "max_bars": 40},
        "doc": "Zaprtje nad najvišjim zaprtjem zadnjih n svečk (samo prvi dan) · moč ≥ strength·σ nad vrhom · "
               "cena največ max_below_high pod vrhom high_n svečk · R = σ·√r_bars · SL = 1R, TP = tp_r·R · "
               "časovni izhod po max_bars. Deluje v bikovskih trendih.",
    },
}
for _name, _cls in ALL_STRATEGIES.items():
    LAB_STRATEGIES[_name] = {
        "label": f"{_cls.label} (+ SL/TP)",
        "func": _from_position(_name),
        "params": {"sl_atr": 2.0, "tp_r": 3.0, "max_bars": 0},
        "doc": "Vstop in izstop po signalu strategije iz knjižnice · SL = sl_atr × ATR(14) · "
               "TP = tp_r × SL (0 = brez TP) · max_bars 0 = brez časovnega izhoda.",
    }


# =====================================================================
# KOREKCIJE – jedro strategije ostane enako, to so dodatki, ki jih vklapljaš
# =====================================================================
DEFAULT_CORR = {"side": "oboje", "sl_mult": 1.0, "tp_mult": 1.0, "no_tp": False, "max_bars": 0,
                "trail_r": 0.0, "be_r": 0.0, "adx_min": 0, "vol_lo": 0, "vol_hi": 100, "vol_k": 0.0,
                "days": list(range(7)), "hours": (0, 23), "regimes": list(REG_ALL)}


def apply_corrections(df, sigs, corr):
    c = {**DEFAULT_CORR, **(corr or {})}
    s = sigs.copy()
    if s.empty:
        return s
    if c["side"] == "samo long":
        s = s[s["side"] > 0]
    elif c["side"] == "samo short":
        s = s[s["side"] < 0]
    if s.empty:
        return s.reset_index(drop=True)
    s["sl_pct"] = s["sl_pct"].astype(float) * c["sl_mult"]
    s["tp_pct"] = np.nan if c["no_tp"] else s["tp_pct"].astype(float) * c["tp_mult"]
    if c["max_bars"] > 0:
        s["max_bars"] = int(c["max_bars"])
    ii = s["i"].astype(int).values
    when = df.index[ii]
    keep = np.ones(len(s), bool)
    if c["adx_min"] > 0:
        adx, _ = _adx(df)
        keep &= adx.fillna(0).values[ii] >= c["adx_min"]
    if c["vol_lo"] > 0 or c["vol_hi"] < 100:
        ap = _atr(df) / df["close"]
        rank = (ap.rolling(250, min_periods=50).rank(pct=True) * 100).fillna(50).values
        keep &= (rank[ii] >= c["vol_lo"]) & (rank[ii] <= c["vol_hi"])
    if c["vol_k"] > 0:
        v = df["volume"]
        avg = v.rolling(20).mean().values
        keep &= v.values[ii] >= c["vol_k"] * np.nan_to_num(avg[ii], nan=np.inf)
    keep &= np.isin(when.dayofweek, list(c["days"]))
    h0, h1 = c["hours"]
    keep &= (when.hour >= h0) & (when.hour <= h1)
    if set(c["regimes"]) != set(REG_ALL):
        reg = regime_series(df).values[ii]
        keep &= np.isin(reg, list(c["regimes"]))
    return s[keep].reset_index(drop=True)


# =====================================================================
# SIMULACIJA
# =====================================================================
def _arrays(df):
    return (df["open"].values.astype(float), df["high"].values.astype(float),
            df["low"].values.astype(float), df["close"].values.astype(float))


def _sim_one(arr, i, side, sl_pct, tp_pct, max_bars, exit_i, cost, trail_r, be_r, entry_mode):
    o, h, l, c = arr
    n = len(c)
    if not (sl_pct > 0) or i + 1 >= n:
        return None
    e = i + 1
    entry = c[i] if entry_mode == "close" else o[e]
    if not np.isfinite(entry) or entry <= 0:
        return None
    risk = entry * sl_pct
    sl0 = sl = entry - side * risk
    tp = entry + side * tp_pct * entry if tp_pct == tp_pct else side * np.inf
    natural_end = e + int(max_bars) - 1
    last = min(n - 1, natural_end)
    sig_exit = None
    if exit_i == exit_i and exit_i is not None:
        xi = int(exit_i)
        if xi <= last:
            last, sig_exit = xi, xi + 1
    hi_max, lo_min = entry, entry
    exit_px, reason, x = None, None, last
    for j in range(e, last + 1):
        if j > e or entry_mode == "close":           # gap čez SL/TP ob odprtju
            if (o[j] - sl) * side <= 0:
                exit_px, reason, x = o[j], "SL", j
                break
            if (o[j] - tp) * side >= 0:
                exit_px, reason, x = o[j], "TP", j
                break
        hi_max, lo_min = max(hi_max, h[j]), min(lo_min, l[j])
        adverse = l[j] if side > 0 else h[j]
        favor = h[j] if side > 0 else l[j]
        if (adverse - sl) * side <= 0:               # SL ima prednost pred TP v isti sveči
            exit_px, reason, x = sl, "SL", j
            break
        if (favor - tp) * side >= 0:
            exit_px, reason, x = tp, "TP", j
            break
        best = hi_max if side > 0 else lo_min        # premik SL šele na zaprtju sveče
        if trail_r > 0:
            new = best - side * trail_r * risk
            sl = max(sl, new) if side > 0 else min(sl, new)
        if be_r > 0 and (best - entry) * side >= be_r * risk:
            sl = max(sl, entry) if side > 0 else min(sl, entry)
    if exit_px is None:
        if sig_exit is not None and sig_exit < n:
            exit_px, reason, x = o[sig_exit], "signal", sig_exit
        elif sig_exit is not None or natural_end > n - 1:
            exit_px, reason, x = c[n - 1], "odprt", n - 1
        else:
            exit_px, reason, x = c[last], "čas", last
    if reason == "SL" and sl != sl0:
        reason = "SL (premaknjen)"
    R = side * (exit_px - entry) / risk - 2 * cost / sl_pct
    mae = (entry - lo_min) / risk if side > 0 else (hi_max - entry) / risk
    mfe = (hi_max - entry) / risk if side > 0 else (entry - lo_min) / risk
    return {"_i": i, "_e": e, "_x": x, "smer": "long" if side > 0 else "short", "vstopna": entry,
            "izstopna": exit_px, "SL": sl0, "TP": tp if np.isfinite(tp) else np.nan, "razlog": reason,
            "R": R, "donos_%": (side * (exit_px / entry - 1) - 2 * cost) * 100, "svečk": x - e + 1,
            "MAE_R": mae, "MFE_R": mfe, "_sl_pct": sl_pct, "_tp_pct": tp_pct, "_max_bars": max_bars}


def simulate(df, sigs, cost=0.001, trail_r=0.0, be_r=0.0, entry_mode="open"):
    arr = _arrays(df)
    out, busy = [], -1
    if sigs is not None and not sigs.empty:
        for r in sigs.sort_values("i").to_dict("records"):
            i = int(r["i"])
            if i <= busy:
                continue
            t = _sim_one(arr, i, int(r["side"]), float(r["sl_pct"]), float(r["tp_pct"]),
                         int(r["max_bars"]), r["exit_i"], cost, trail_r, be_r, entry_mode)
            if t is None:
                continue
            out.append(t)
            busy = t["_x"]
    cols = ["vstop", "izstop", "smer", "vstopna", "izstopna", "SL", "TP", "razlog", "R", "donos_%",
            "svečk", "MAE_R", "MFE_R", "_i", "_e", "_x", "_sl_pct", "_tp_pct", "_max_bars"]
    if not out:
        return pd.DataFrame(columns=cols)
    t = pd.DataFrame(out)
    t["vstop"] = df.index[t["_e"].values]
    t["izstop"] = df.index[t["_x"].values]
    return t[cols]


# =====================================================================
# STATISTIKA, RAČUN, NAKLJUČNI TEST
# =====================================================================
def trade_stats(t):
    t = t[t["razlog"] != "odprt"]
    n = len(t)
    if n == 0:
        return {"Trejdov": 0}
    r = t["R"]
    loss = -r[r < 0].sum()
    streak = mx = 0
    for v in r:
        streak = streak + 1 if v < 0 else 0
        mx = max(mx, streak)
    return {"Trejdov": n, "Win %": round((r > 0).mean() * 100, 1), "Povp. R": round(r.mean(), 3),
            "Povp. dobitek R": round(r[r > 0].mean(), 2) if (r > 0).any() else 0,
            "Povp. izguba R": round(r[r < 0].mean(), 2) if (r < 0).any() else 0,
            "Skupaj R": round(r.sum(), 1), "Profit factor": round(r[r > 0].sum() / loss, 2) if loss > 0 else np.inf,
            "Max zaporednih izgub": mx, "Povp. svečk": round(t["svečk"].mean(), 1)}


def account_curve(t, risk=0.01, start=10000.0):
    t = t.sort_values("izstop")
    if t.empty:
        return pd.Series(dtype=float), {}
    eq = pd.Series(start * np.cumprod(1 + risk * t["R"].values), index=pd.DatetimeIndex(t["izstop"]))
    years = max((t["izstop"].max() - t["vstop"].min()).days / 365.25, 1 / 12)
    cagr = (eq.iloc[-1] / start) ** (1 / years) - 1 if eq.iloc[-1] > 0 else -1
    dd = (eq / np.maximum.accumulate(np.r_[start, eq.values])[1:] - 1).min()
    return eq, {"Končno €": f"{eq.iloc[-1]:,.0f}", "Letno %": round(cagr * 100, 1),
                "Mesečno %": round(((1 + cagr) ** (1 / 12) - 1) * 100, 2),
                "Max DD %": round(dd * 100, 1), "Letno / DD": round(cagr / abs(dd), 2) if dd < 0 else np.inf}


def _fast_R(arr, e, side, sl_pct, tp_pct, max_bars, cost):
    o, h, l, c = arr
    n = len(c)
    entry = o[e]
    risk = entry * sl_pct
    sl = entry - side * risk
    tp = entry + side * tp_pct * entry if tp_pct == tp_pct else side * np.inf
    last = min(n - 1, e + int(max_bars) - 1)
    hh, ll = h[e:last + 1], l[e:last + 1]
    s_hit = (ll <= sl) if side > 0 else (hh >= sl)
    t_hit = (hh >= tp) if side > 0 else (ll <= tp)
    si = int(s_hit.argmax()) if s_hit.any() else NO_LIMIT
    ti = int(t_hit.argmax()) if t_hit.any() else NO_LIMIT
    if si <= ti and si < NO_LIMIT:
        px = o[e + si]
        exit_px = (min(sl, px) if side > 0 else max(sl, px)) if si > 0 else sl
    elif ti < NO_LIMIT:
        px = o[e + ti]
        exit_px = (max(tp, px) if side > 0 else min(tp, px)) if ti > 0 else tp
    else:
        exit_px = c[last]
    return side * (exit_px - entry) / risk - 2 * cost / sl_pct


def random_test(df, trades, allowed=None, n_random=1000, cost=0.001, trail_r=0.0, be_r=0.0, seed=42):
    """Naključni vstopi: enako število trejdov, enak delež long/short, enaki SL/TP in časovni izhodi."""
    t = trades[trades["razlog"] != "odprt"]
    if len(t) < 5:
        return None
    arr = _arrays(df)
    n = len(df)
    cand = np.flatnonzero(allowed if allowed is not None else np.ones(n, bool))
    cand = cand[cand < n - 2]
    if len(cand) == 0:
        return None
    rng = np.random.default_rng(seed)
    sl_p, tp_p = t["_sl_pct"].values, t["_tp_pct"].values
    cap = np.where(t["_max_bars"].values >= NO_LIMIT, t["svečk"].values, t["_max_bars"].values)
    long_share = (t["smer"] == "long").mean()
    dynamic = trail_r > 0 or be_r > 0
    means = np.empty(n_random)
    for k in range(n_random):
        picks = rng.choice(cand, len(t))
        sides = np.where(rng.random(len(t)) < long_share, 1, -1)
        order = rng.permutation(len(t))
        rs = np.empty(len(t))
        for m in range(len(t)):
            q = order[m]
            if dynamic:
                res = _sim_one(arr, picks[m], sides[m], sl_p[q], tp_p[q], cap[q], np.nan, cost, trail_r, be_r, "open")
                rs[m] = res["R"] if res else 0.0
            else:
                rs[m] = _fast_R(arr, picks[m] + 1, sides[m], sl_p[q], tp_p[q], cap[q], cost)
        means[k] = rs.mean()
    strat = t["R"].mean()
    pct = float((means < strat).mean() * 100)
    if len(t) < 20:
        verdict = "⚪ premalo trejdov"
    else:
        verdict = "✅ edge" if pct >= 95 else "🟡 obetavno" if pct >= 80 else "❌ ni edga"
    return {"means": means, "percentile": round(pct, 1), "verdict": verdict,
            "median_R": round(float(np.median(means)), 3), "top5_R": round(float(np.percentile(means, 95)), 3)}


def group_stats(t, key):
    t = t[t["razlog"] != "odprt"]
    if t.empty:
        return pd.DataFrame()
    g = t.groupby(key)["R"]
    out = pd.DataFrame({"Trejdov": g.count(), "Win %": g.apply(lambda r: (r > 0).mean() * 100),
                        "Povp. R": g.mean(), "Skupaj R": g.sum()}).round(2)
    return out


def sl_tp_grid(df, sigs_base, corr, cost, entry_mode, pmask,
               sl_list=(0.5, 0.75, 1.0, 1.5, 2.0), tp_list=(1.0, 1.5, 2.0, 3.0, 4.0)):
    """SL = množitelj × SL strategije, TP = večkratnik (v R) novega SL."""
    rows = []
    for a in sl_list:
        for b in tp_list:
            s = sigs_base.copy()
            if s.empty:
                continue
            s["sl_pct"] = s["sl_pct"].astype(float) * a
            s["tp_pct"] = s["sl_pct"] * b
            t = simulate(df, s, cost, corr.get("trail_r", 0), corr.get("be_r", 0), entry_mode)
            t = t[pmask[t["_e"].values.astype(int)]] if len(t) else t
            st_ = trade_stats(t)
            rows.append({"SL ×": a, "TP (R)": b, "Trejdov": st_.get("Trejdov", 0),
                         "Win %": st_.get("Win %", 0), "Povp. R": st_.get("Povp. R", 0),
                         "Skupaj R": st_.get("Skupaj R", 0)})
    return pd.DataFrame(rows)


# =====================================================================
# KALKULATOR PREMIKA (zigzag)
# =====================================================================
def zigzag(df, pct):
    h, l = df["high"].values, df["low"].values
    piv = [(0, l[0], "L")]
    trend, ext, ext_i = 1, h[0], 0
    for i in range(1, len(h)):
        if trend > 0:
            if h[i] > ext:
                ext, ext_i = h[i], i
            elif l[i] <= ext * (1 - pct):
                piv.append((ext_i, ext, "H"))
                trend, ext, ext_i = -1, l[i], i
        else:
            if l[i] < ext:
                ext, ext_i = l[i], i
            elif h[i] >= ext * (1 + pct):
                piv.append((ext_i, ext, "L"))
                trend, ext, ext_i = 1, h[i], i
    return piv, (ext_i, ext, "H" if trend > 0 else "L"), trend


def auto_pct(df):
    s = float(np.nanstd(df["close"].pct_change().values))
    return max(round(s * np.sqrt(20), 4), 0.002)


def move_calculator(df, pct=None):
    pct = pct or auto_pct(df)
    piv, cur, trend = zigzag(df, pct)
    pts = piv[1:]                                      # prvi pivot je umeten
    legs = [{"začetek": df.index[i1], "konec": df.index[i2], "smer": "gor" if t1 == "L" else "dol",
             "velikost_%": abs(p2 / p1 - 1) * 100, "svečk": i2 - i1}
            for (i1, p1, t1), (i2, p2, _) in zip(pts[:-1], pts[1:])]
    legs = pd.DataFrame(legs, columns=["začetek", "konec", "smer", "velikost_%", "svečk"])
    last_i, last_p, _ = piv[-1]
    ci, cp, _ = cur
    d = "gor" if trend > 0 else "dol"
    sign = 1 if trend > 0 else -1
    size = abs(cp / last_p - 1) * 100
    bars = len(df) - 1 - last_i
    price = float(df["close"].iloc[-1])
    same = legs[legs["smer"] == d]
    S, D = same["velikost_%"].values, same["svečk"].values
    surv = S[S >= size]
    tg = []
    for q in (25, 50, 75, 90):
        v = float(np.percentile(surv, q)) if len(surv) >= 3 else np.nan
        tg.append({"Verjetnost %": 100 - q, "Velikost premika %": round(v, 2),
                   "Ciljna cena": round(last_p * (1 + sign * v / 100), 6) if v == v else np.nan,
                   "Od trenutne cene %": round((last_p * (1 + sign * v / 100) / price - 1) * 100, 2) if v == v else np.nan})
    n_ge = max((S >= size).sum(), 1)
    cont = (S >= size + pct * 100).sum() / n_ge * 100 if len(S) else np.nan
    Dsurv = D[D >= bars]
    rem = float(np.median(Dsurv) - bars) if len(Dsurv) >= 3 else np.nan

    dist = []
    for dd in ("gor", "dol"):
        x = legs[legs["smer"] == dd]
        if x.empty:
            continue
        for col, lab in (("velikost_%", "velikost %"), ("svečk", "trajanje (svečk)")):
            v = x[col].values
            dist.append({"Smer": dd, "Mera": lab, "Št.": len(v), "Povp.": round(v.mean(), 2),
                         **{f"p{q}": round(float(np.percentile(v, q)), 2) for q in (10, 25, 50, 75, 90)}})
    pivots = pd.DataFrame([{"čas": df.index[i], "cena": p, "tip": t} for i, p, t in pts] +
                          [{"čas": df.index[ci], "cena": cp, "tip": "trenutno"}])
    return {"pct": pct, "legs": legs, "pivots": pivots, "targets": pd.DataFrame(tg),
            "dist": pd.DataFrame(dist), "n_same": len(S), "n_surv": len(surv),
            "cur": {"smer": d, "začetek": df.index[last_i], "od_cene": last_p, "ekstrem": cp,
                    "velikost_%": round(size, 2), "svečk": bars, "cena": price,
                    "povratek_%": round(abs(price / cp - 1) * 100, 2),
                    "do_obrata_%": round(pct * 100 - abs(price / cp - 1) * 100, 2),
                    "nadaljevanje_%": round(cont, 1) if cont == cont else None,
                    "še_svečk_med": rem}}


# =====================================================================
# VSE SKUPAJ
# =====================================================================
def run_lab(df, name, params, corr, entry_mode="open", od=None, do=None, risk=0.01,
            zz_pct=None, cost=0.001, n_random=1000):
    corr = {**DEFAULT_CORR, **(corr or {})}
    func = LAB_STRATEGIES[name]["func"]
    base = func(df, params)
    sigs = apply_corrections(df, base, corr)
    pm = period_mask(df.index, od, do)
    trades = simulate(df, sigs, cost, corr["trail_r"], corr["be_r"], entry_mode)
    if len(trades):
        trades = trades[pm[trades["_e"].values.astype(int)]].reset_index(drop=True)
    stats = trade_stats(trades)
    eq, acc = account_curve(trades[trades["razlog"] != "odprt"], risk)
    allowed = pm.copy()
    rnd = random_test(df, trades, allowed, n_random, cost, corr["trail_r"], corr["be_r"])

    regs = regime_series(df)
    tt = trades.copy()
    if len(tt):
        tt["režim"] = regs.values[tt["_i"].values.astype(int)]
        tt["dan"] = [DAYS[d] for d in pd.DatetimeIndex(tt["vstop"]).dayofweek]
        tt["ura"] = pd.DatetimeIndex(tt["vstop"]).hour
    by_regime = group_stats(tt, "režim") if len(tt) else pd.DataFrame()
    by_day = group_stats(tt, "dan") if len(tt) else pd.DataFrame()
    if len(by_day):
        by_day = by_day.reindex([d for d in DAYS if d in by_day.index])
    intraday = pd.Series(df.index).diff().median() < pd.Timedelta(hours=20)
    by_hour = group_stats(tt, "ura") if (len(tt) and intraday) else pd.DataFrame()

    grid_base = apply_corrections(df, base, {**corr, "sl_mult": 1.0, "tp_mult": 1.0, "no_tp": False})
    grid = sl_tp_grid(df, grid_base, corr, cost, entry_mode, pm)

    end_mask = period_mask(df.index, None, do)
    mv = move_calculator(df[end_mask], zz_pct or None) if end_mask.sum() > 30 else None
    return {"trades": trades, "stats": stats, "equity": eq, "account": acc, "random": rnd,
            "by_regime": by_regime, "by_day": by_day, "by_hour": by_hour, "grid": grid,
            "move": mv, "pmask": pm, "n_signals": len(base), "n_after_corr": len(sigs)}
