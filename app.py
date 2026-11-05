"""
ALGO PORTFELJ – aplikacija
Zagon lokalno: streamlit run app.py
"""
import io
import json
import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from engine import ai, alpaca, storage
from engine import builder as B
from engine.backtest import apply_regime_filter, bars_per_year, metrics, random_test, run_backtest
import time

from engine.custom import register_custom
from engine.data import INTERVALS, get_data
from engine.portfolio import active_combos, combo_returns, portfolio_returns
from engine.regime import REGIME_LABELS, detect, regime_series
from engine.risk import FTMO_RULES, ftmo_sim, to_daily
from engine.strategies import ALL_STRATEGIES
from engine import journal as J
from engine import notify
from engine.tradelab import (DAYS, DEFAULT_CORR, LAB_STRATEGIES, REG_ALL, account_curve, all_lab_strategies,
                             apply_corrections, ftmo_trades, lab_walkforward, move_calculator, multi_market,
                             period_mask, run_lab, simulate)
from engine.walkforward import walk_forward

st.set_page_config(page_title="Algo portfelj", page_icon="📈", layout="wide")
BASE = os.path.dirname(os.path.abspath(__file__))
REGIMES = ["uptrend", "downtrend", "range"]

# ---------------------------------------------------------------- stanje
if "settings" not in st.session_state:
    try:
        st.session_state.settings, st.session_state.settings_src = storage.load_settings(
            os.path.join(BASE, "settings.json"))
    except Exception as e:
        with open(os.path.join(BASE, "settings.json"), encoding="utf-8") as f:
            st.session_state.settings = json.load(f)
        st.session_state.settings_src = f"privzeto ({e})"
        for k, v in {"overrides": {}, "edge": [],
                     "risk": {"target_vol": 0.15, "max_lev": 3.0, "cost": 0.001},
                     "paper": {"enabled": False, "only_edge": True, "capital_use": 1.0}}.items():
            st.session_state.settings.setdefault(k, v)
S = st.session_state.settings
S.setdefault("custom_strategies", {})
S.setdefault("lab_saved", {})

# naloženo stanje gradnikov (iz "Naloži nastavitev" / graditelja) se uveljavi pred risanjem
if "pending" in st.session_state:
    for _k, _v in st.session_state.pop("pending").items():
        st.session_state[_k] = _v


register_custom(S["custom_strategies"])
for _nm in S["custom_strategies"]:
    S.setdefault("strategies_on", {}).setdefault("custom:" + _nm, True)


def ss(key, default):
    """Gradnik z začetno vrednostjo v session_state (da ga lahko kasneje naložimo iz shranjenih)."""
    if key not in st.session_state:
        st.session_state[key] = default
    return key



def save():
    if storage.enabled():
        try:
            storage.save_settings(S)
            st.toast("✅ Nastavitve shranjene na GitHub")
        except Exception as e:
            st.error(f"Shranjevanje ni uspelo: {e}")
    else:
        st.toast("Shranjeno samo začasno (GH_TOKEN ni nastavljen)")


@st.cache_data(ttl=3600, show_spinner=False)
def load(symbol, interval):
    return get_data(symbol, interval)


def auto_regimes():
    if "auto_regimes" not in st.session_state:
        out = {}
        for sym in S["markets"]:
            try:
                out[sym] = detect(load(sym, "1d"))["regime"]
            except Exception:
                out[sym] = "unknown"
        st.session_state.auto_regimes = out
    return st.session_state.auto_regimes


def metric_row(m):
    cols = st.columns(len(m))
    for c, (k, v) in zip(cols, m.items()):
        c.metric(k, v)


@st.cache_data(ttl=3600, show_spinner=False)
def load_live(symbol, interval, bucket):
    """bucket se spremeni vsakih N sekund -> sveži podatki."""
    return get_data(symbol, interval)


def period_inputs(key):
    a, b, c = st.columns(3)
    od = a.date_input("Od (prazno = začetek)", key=ss(f"{key}_od", None), format="DD.MM.YYYY")
    do = b.date_input("Do (prazno = konec)", key=ss(f"{key}_do", None), format="DD.MM.YYYY")
    m = c.text_input("…ali cel mesec (LLLL-MM)", placeholder="npr. 2024-08", key=ss(f"{key}_m", ""))
    if m.strip():
        try:
            p = pd.Period(m.strip(), "M")
            od, do = p.start_time.date(), p.end_time.date()
        except Exception:
            st.warning("Mesec vpiši kot 2024-08")
    return od, do


def corrections_ui(key):
    c = dict(DEFAULT_CORR)
    a, b, cc, d = st.columns(4)
    c["side"] = a.selectbox("Smer", ["oboje", "samo long", "samo short"], key=ss(f"{key}_side", "oboje"))
    c["sl_mult"] = b.number_input("SL × (množitelj)", 0.1, 5.0, step=0.1, key=ss(f"{key}_slm", 1.0))
    c["tp_mult"] = cc.number_input("TP × (množitelj)", 0.1, 10.0, step=0.1, key=ss(f"{key}_tpm", 1.0))
    c["no_tp"] = d.toggle("Brez TP", key=ss(f"{key}_notp", False))
    a, b, cc, d = st.columns(4)
    c["max_bars"] = a.number_input("Časovni izhod (svečk, 0 = strategija)", 0, 100000, key=ss(f"{key}_mb", 0))
    c["trail_r"] = b.number_input("Trailing stop (R, 0 = izklop)", 0.0, 10.0, step=0.25, key=ss(f"{key}_tr", 0.0))
    c["be_r"] = cc.number_input("SL na break-even po +R (0 = izklop)", 0.0, 10.0, step=0.25, key=ss(f"{key}_be", 0.0))
    c["adx_min"] = d.number_input("Min ADX (0 = izklop)", 0, 60, key=ss(f"{key}_adx", 0))
    a, b, cc = st.columns(3)
    c["vol_lo"], c["vol_hi"] = a.slider("Volatilnost – percentil ATR %", 0, 100, key=ss(f"{key}_vol", (0, 100)),
                                        help="Npr. 0–50 = trguj samo, ko je trg mirnejši od običajnega")
    c["vol_k"] = b.number_input("Volumen ≥ × povprečja 20 (0 = izklop)", 0.0, 5.0, step=0.1, key=ss(f"{key}_vk", 0.0))
    c["regimes"] = cc.multiselect("Samo v režimih", REG_ALL, key=ss(f"{key}_reg", list(REG_ALL)))
    a, b = st.columns(2)
    c["days"] = a.multiselect("Dnevi vstopa", list(range(7)), format_func=lambda d: DAYS[d],
                              key=ss(f"{key}_days", list(range(7))))
    c["hours"] = b.slider("Ure vstopa (UTC, za intraday)", 0, 23, key=ss(f"{key}_hrs", (0, 23)),
                          help="London ≈ 7–16, New York ≈ 13–21 UTC")
    return c


LAB_KEYS = ["lab_sym", "lab_strat", "lab_entry", "lab_od", "lab_do", "lab_m", "lab_risk", "lab_zz", "lab_kind",
            "lab_side", "lab_slm", "lab_tpm", "lab_notp", "lab_mb", "lab_tr", "lab_be", "lab_adx", "lab_vol",
            "lab_vk", "lab_reg", "lab_days", "lab_hrs", "interval"]


def _jsonable(v):
    if hasattr(v, "isoformat"):
        return {"__date__": v.isoformat()}
    if isinstance(v, tuple):
        return list(v)
    return v


def _from_json(k, v):
    if isinstance(v, dict) and "__date__" in v:
        return pd.Timestamp(v["__date__"]).date()
    if k in ("lab_vol", "lab_hrs") and isinstance(v, list):
        return tuple(v)
    return v


def lab_chart(df, trades=None, pivots=None, targets=None, kind="sveče", height=600, title="", max_bars=6000):
    if len(df) > max_bars:
        df = df.iloc[-max_bars:]
    x0 = df.index[0]
    fig = go.Figure()
    if kind == "sveče":
        fig.add_trace(go.Candlestick(x=df.index, open=df["open"], high=df["high"], low=df["low"],
                                     close=df["close"], name="cena"))
    else:
        fig.add_trace(go.Scatter(x=df.index, y=df["close"], name="cena", line=dict(width=1.2)))
    if pivots is not None and len(pivots):
        p = pivots[pivots["čas"] >= x0]
        fig.add_trace(go.Scatter(x=p["čas"], y=p["cena"], mode="lines+markers", name="zigzag",
                                 line=dict(color="orange", width=1.5), marker=dict(size=4)))
    if trades is not None and len(trades):
        t = trades[trades["vstop"] >= x0].tail(400)
        if len(t):
            def seg(col):
                xs, ys = [], []
                for a, b, v in zip(t["vstop"], t["izstop"], t[col]):
                    if v == v:
                        xs += [a, b, None]
                        ys += [v, v, None]
                return xs, ys
            xs, ys = seg("SL")
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="SL", line=dict(color="#d62728", width=1, dash="dash")))
            xs, ys = seg("TP")
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="TP", line=dict(color="#2ca02c", width=1, dash="dash")))
            px, py = [], []
            for a, b, v1, v2 in zip(t["vstop"], t["izstop"], t["vstopna"], t["izstopna"]):
                px += [a, b, None]
                py += [v1, v2, None]
            fig.add_trace(go.Scatter(x=px, y=py, mode="lines", name="pot trejda",
                                     line=dict(color="gray", width=1, dash="dot")))
            for side, sym, col in (("long", "triangle-up", "#2ca02c"), ("short", "triangle-down", "#d62728")):
                s = t[t["smer"] == side]
                if len(s):
                    fig.add_trace(go.Scatter(x=s["vstop"], y=s["vstopna"], mode="markers", name=f"vstop {side}",
                                             marker=dict(symbol=sym, size=11, color=col, line=dict(width=1, color="black"))))
            fig.add_trace(go.Scatter(
                x=t["izstop"], y=t["izstopna"], mode="markers", name="izstop",
                marker=dict(symbol="x", size=9, color=np.where(t["R"] > 0, "#2ca02c", "#d62728")),
                text=[f"{r} · {v:+.2f}R" for r, v in zip(t["razlog"], t["R"])], hoverinfo="text+x+y"))
    if targets is not None and len(targets.dropna()):
        tg = targets.dropna()
        span = (df.index[-1] - x0) * 0.06
        y50 = tg.loc[tg["Verjetnost %"] == 50, "Ciljna cena"]
        y25 = tg.loc[tg["Verjetnost %"] == 25, "Ciljna cena"]
        y10 = tg.loc[tg["Verjetnost %"] == 10, "Ciljna cena"]
        if len(y50) and len(y25):
            fig.add_shape(type="rect", x0=df.index[-1], x1=df.index[-1] + span, y0=float(y50.iloc[0]),
                          y1=float(y25.iloc[0]), fillcolor="rgba(255,165,0,0.25)", line_width=0)
            fig.add_annotation(x=df.index[-1] + span, y=float(y50.iloc[0]), text="cilj 50 %", showarrow=False,
                               xanchor="left", font=dict(size=10))
            fig.add_annotation(x=df.index[-1] + span, y=float(y25.iloc[0]), text="cilj 25 %", showarrow=False,
                               xanchor="left", font=dict(size=10))
        if len(y10):
            fig.add_shape(type="line", x0=df.index[-1], x1=df.index[-1] + span, y0=float(y10.iloc[0]),
                          y1=float(y10.iloc[0]), line=dict(color="orange", dash="dot"))
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=40, b=0), title=title,
                      xaxis_rangeslider_visible=False, legend=dict(orientation="h", y=1.02))
    return fig


# ---------------------------------------------------------------- stranska vrstica
with st.sidebar:
    st.title("📈 Algo portfelj")
    interval = st.radio("Časovni okvir", INTERVALS, horizontal=True, key=ss("interval", "1d"),
                        help="1d = 10 let, 4h/1h = ~2 leti, 30m/15m = ~60 dni (omejitev Yahoo)")
    n_random = st.select_slider("Naključnih trejdov za test", [200, 500, 1000], value=1000)

    st.markdown("**⚖️ Risk manager**")
    R = S["risk"]
    R["target_vol"] = st.slider("Ciljna letna volatilnost na strategijo (%)", 5, 40,
                                int(R["target_vol"] * 100), 1,
                                help="Velikost pozicije se prilagaja volatilnosti trga. "
                                     "Višje = večji donosi IN večji drawdowni.") / 100
    R["max_lev"] = st.slider("Max vzvod na strategijo", 0.5, 5.0, float(R["max_lev"]), 0.5)
    R["cost"] = st.number_input("Stroški na stran (%)", 0.0, 1.0, float(R["cost"]) * 100, 0.01) / 100

    st.divider()
    if st.button("💾 Shrani nastavitve", type="primary", width="stretch"):
        save()
    st.caption(("🟢 Trajno shranjevanje na GitHub" if storage.enabled()
                else "⚪ Shranjevanje je začasno – v Secrets dodaj GH_TOKEN in GH_REPO")
               + f" · naloženo: {st.session_state.settings_src}")

tab_m, tab_s, tab_p, tab_b, tab_l, tab_g, tab_j, tab_t = st.tabs(
    ["🌍 Trgi", "🧠 Strategije", "💼 Portfelj + FTMO", "🔬 Backtest", "🎯 Lab", "🧩 Graditelj", "📓 Dnevnik",
     "🤖 Paper trading"])
LABS = all_lab_strategies(S["custom_strategies"])

# ================================================================ TRGI
with tab_m:
    st.subheader("Trgi in režimi")
    st.caption("Program sam zazna režim (dnevne svečke: EMA50/EMA200 + ADX). "
               "V stolpcu **Tvoj režim** ga lahko povoziš z lastno oceno (novice, makro).")
    rows = []
    with st.spinner("Nalagam trge..."):
        for sym, m in S["markets"].items():
            try:
                d = detect(load(sym, "1d"))
                rows.append({"Simbol": sym, "Ime": m["ime"], "Trg": m["trg"],
                             "Avto režim": REGIME_LABELS[d["regime"]], "Dni": d["dni_v_rezimu"],
                             "Tvoj režim": S["overrides"].get(sym, "auto"),
                             "1M %": d["1m_%"], "3M %": d["3m_%"], "ADX": d["adx"],
                             "Volatilnost": d["volatilnost"], "Alpaca": alpaca.alpaca_symbol(sym, m) or "–",
                             "_auto": d["regime"]})
            except Exception as e:
                rows.append({"Simbol": sym, "Ime": m["ime"], "Trg": m["trg"],
                             "Avto režim": f"napaka: {e}", "Tvoj režim": "auto", "_auto": "unknown"})
    mdf = pd.DataFrame(rows)
    st.session_state.auto_regimes = dict(zip(mdf["Simbol"], mdf["_auto"]))
    edited = st.data_editor(
        mdf.drop(columns="_auto"), hide_index=True, width="stretch",
        disabled=[c for c in mdf.columns if c != "Tvoj režim"],
        column_config={"Tvoj režim": st.column_config.SelectboxColumn(options=["auto"] + REGIMES, required=True)})
    new_over = {k: v for k, v in zip(edited["Simbol"], edited["Tvoj režim"]) if v != "auto"}
    if new_over != S["overrides"]:
        S["overrides"] = new_over
        st.rerun()

    with st.expander("➕ Dodaj / odstrani trg"):
        c1, c2, c3, c4 = st.columns([2, 2, 2, 2])
        sym = c1.text_input("Simbol (Yahoo)", placeholder="npr. XRP-USD, ^GDAXI, GBPUSD=X")
        ime = c2.text_input("Ime")
        trg = c3.selectbox("Vrsta trga", ["crypto", "index", "forex", "commodity"])
        alp = c4.text_input("Alpaca simbol (neobvezno)", placeholder="npr. XRP/USD, IWM")
        if st.button("Dodaj") and sym:
            S["markets"][sym.strip()] = {"ime": ime or sym, "trg": trg, **({"alpaca": alp} if alp else {})}
            st.session_state.pop("auto_regimes", None)
            st.rerun()
        rem = st.selectbox("Odstrani", [""] + list(S["markets"]))
        if rem and st.button("Odstrani izbrani"):
            S["markets"].pop(rem, None)
            S["overrides"].pop(rem, None)
            st.rerun()

# ================================================================ STRATEGIJE
with tab_s:
    st.subheader("Knjižnica strategij")
    for name, cls in ALL_STRATEGIES.items():
        s = cls()
        c1, c2 = st.columns([1, 6])
        S["strategies_on"][name] = c1.toggle("Vklop", S["strategies_on"].get(name, True), key=f"on_{name}")
        doc = ((cls.__doc__ if name.startswith("custom:") else sys.modules[cls.__module__].__doc__) or "").strip().splitlines()
        if name.startswith("custom:"):
            doc = [""] + doc
        c2.markdown(f"**{s.label}** · režimi: {', '.join(s.regimes)} · trgi: {', '.join(s.markets)}")
        if len(doc) > 1:
            c2.caption(" · ".join(l.strip() for l in doc[1:] if l.strip()))

    st.divider()
    st.markdown("### Kje ima katera strategija edge?")
    st.caption("Vsaka strategija se testira na vsakem ustreznem trgu, **ločeno po režimih**. "
               "✅ edge = boljša od 95 % naključnih trejdov in vsaj 20 trejdov. "
               "Seznam ✅ kombinacij se shrani in ga uporablja paper trading bot. "
               "Trend strategije na 1d naredijo malo trejdov – za oceno jih poženi tudi na **1h**.")
    if st.button("▶️ Ovrednoti vse strategije", type="primary"):
        combos = [(n, sym, r) for n, cls in ALL_STRATEGIES.items() if S["strategies_on"].get(n)
                  for sym, m in S["markets"].items() if m["trg"] in cls.markets for r in cls.regimes]
        prog, out = st.progress(0.0), []
        for i, (n, sym, r) in enumerate(combos):
            prog.progress((i + 1) / len(combos), f"{n} · {sym} · {r}")
            try:
                df = load(sym, interval)
                regs = regime_series(df)
                sig = apply_regime_filter(ALL_STRATEGIES[n]().generate_signals(df), regs, r)
                res = run_backtest(df, sig, S["risk"]["cost"])
                rnd = random_test(df, res, n_random, allowed=(regs == r))
                out.append({"Strategija": n, "Simbol": sym, "Režim": r, **res["metrics"],
                            "Boljša od naklj. %": rnd["percentile"] if rnd else None,
                            "Ocena": rnd["verdict"] if rnd else "ni trejdov"})
            except Exception as e:
                out.append({"Strategija": n, "Simbol": sym, "Režim": r, "Ocena": f"napaka: {e}"})
        prog.empty()
        e = pd.DataFrame(out)
        st.session_state.edge = e
        S["edge"] = e[e["Ocena"] == "✅ edge"][["Simbol", "Strategija", "Režim"]].values.tolist()
        S["edge_interval"] = interval
        save()

    if "edge" in st.session_state:
        e = st.session_state.edge
        only = st.checkbox("Pokaži samo ✅ edge", False)
        st.dataframe(e[e["Ocena"] == "✅ edge"] if only else e, hide_index=True, width="stretch")
    elif S.get("edge"):
        st.info(f"Shranjene ✅ kombinacije ({S.get('edge_interval', '?')}): "
                + ", ".join(f"{a}·{b}·{c}" for a, b, c in S["edge"]))

# ================================================================ PORTFELJ + FTMO
with tab_p:
    st.subheader("Aktivni portfelj")
    auto = auto_regimes()
    only_edge = st.checkbox("Uporabi samo kombinacije z ✅ edge", bool(S.get("edge")),
                            help="Iz zavihka Strategije")
    edge_set = {tuple(x) for x in S.get("edge", [])} if only_edge else None
    active = active_combos(S, auto, edge_set)
    if not active:
        st.info("Trenutno nobena (vklopljena / ✅) strategija ne ustreza režimom trgov.")
    else:
        st.dataframe(pd.DataFrame(active, columns=["Simbol", "Strategija", "Režim"]),
                     hide_index=True, width="stretch")
        st.caption("Vsaka kombinacija dobi enak del kapitala, velikost pozicije določa risk manager "
                   "(ciljna volatilnost). Strategija trguje samo v zgodovinskih obdobjih, ko je bil "
                   "trg v istem režimu kot zdaj.")
        if st.button("▶️ Backtest portfelja + FTMO simulacija", type="primary"):
            data = {sym: load(sym, interval) for sym in {a[0] for a in active}}
            Rt, port = portfolio_returns(data, active, S["risk"])
            st.session_state.port = (Rt, port)

    if "port" in st.session_state:
        Rt, port = st.session_state.port
        eq = (1 + port).cumprod()
        m = metrics(port, eq, pd.DataFrame(columns=["donos_%"]), bars_per_year(port.index))
        for k in ["Trejdov", "Win rate %", "Profit factor"]:
            m.pop(k)
        metric_row(m)
        fig = go.Figure()
        for c in Rt.columns:
            fig.add_trace(go.Scatter(x=Rt.index, y=(1 + Rt[c]).cumprod(), name=c, line=dict(width=1), opacity=0.5))
        fig.add_trace(go.Scatter(x=eq.index, y=eq, name="PORTFELJ", line=dict(width=3)))
        fig.update_layout(height=420, margin=dict(l=0, r=0, t=30, b=0), yaxis_title="kapital (1 = start)")
        st.plotly_chart(fig, width="stretch")

        st.markdown("### 🏆 FTMO simulacija")
        st.caption("Na zgodovini portfelja začne nov challenge vsakih 5 dni. "
                   "PASS = cilj dosežen pred kršitvijo pravil (max dnevna izguba 5 %, max skupna 10 %).")
        c1, c2 = st.columns(2)
        phase = c1.radio("Faza", list(FTMO_RULES), horizontal=True)
        max_days = c2.slider("Največ dni za challenge", 30, 180, 90, 10)
        res, summ = ftmo_sim(port, phase, 5, max_days)
        if summ:
            metric_row(summ)
            fig2 = go.Figure(go.Bar(x=res["začetek"], y=res["končno_%"],
                                    marker_color=np.where(res["rezultat"] == "PASS", "#2ca02c",
                                                          np.where(res["rezultat"].str.startswith("FAIL"), "#d62728", "#bbbbbb"))))
            fig2.update_layout(height=260, margin=dict(l=0, r=0, t=30, b=0),
                               title="Rezultat challenga glede na začetni dan (zeleno PASS, rdeče FAIL, sivo TIMEOUT)")
            st.plotly_chart(fig2, width="stretch")
            st.info("💡 Če je FAIL % visok → zmanjšaj ciljno volatilnost v stranski vrstici. "
                    "Če je TIMEOUT % visok → portfelj je prepočasen (premalo strategij z edgem ali prenizka volatilnost).")
        st.markdown("**Korelacija med strategijami** (nižja = boljša razpršenost)")
        st.dataframe(Rt.corr().round(2), width="stretch")

# ================================================================ BACKTEST
with tab_b:
    st.subheader("Posamezen backtest")
    c1, c2, c3 = st.columns(3)
    sym = c1.selectbox("Trg", list(S["markets"]), format_func=lambda x: f"{x} – {S['markets'][x]['ime']}")
    n = c2.selectbox("Strategija", list(ALL_STRATEGIES), format_func=lambda x: ALL_STRATEGIES[x].label)
    regf = c3.selectbox("Trguj samo v režimu", ["vsi"] + REGIMES)
    strat = ALL_STRATEGIES[n]()
    st.markdown("**Parametri**")
    pcols = st.columns(max(1, len(strat.params)))
    params = {k: pc.number_input(k, value=v, key=f"p_{n}_{k}") for pc, (k, v) in zip(pcols, strat.params.items())}
    od_b, do_b = period_inputs("bt")

    b1, b2 = st.columns(2)
    if b1.button("▶️ Zaženi backtest", type="primary"):
        dfull = load(sym, interval)
        sig = ALL_STRATEGIES[n](**params).generate_signals(dfull)   # indikatorji na celi zgodovini
        regs = regime_series(dfull)
        pm = period_mask(dfull.index, od_b, do_b)
        df, sig, regs = dfull[pm], sig[pm], regs[pm]
        allowed = None
        if regf != "vsi":
            sig = apply_regime_filter(sig, regs, regf)
            allowed = regs == regf
        if len(df) < 5:
            st.error("V izbranem obdobju ni dovolj podatkov.")
        else:
            res = run_backtest(df, sig, S["risk"]["cost"])
            rnd = random_test(df, res, n_random, allowed=allowed)
            metric_row(res["metrics"])
            if rnd:
                st.markdown(f"### {rnd['verdict']} — boljša od **{rnd['percentile']} %** naključnih "
                            f"(mediana naključnih {rnd['median_%']} %, top 5 % {rnd['top5_%']} %)")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=res["equity"].index, y=res["equity"], name="strategija"))
            fig.add_trace(go.Scatter(x=df.index, y=df["close"] / df["close"].iloc[0], name="buy & hold", opacity=0.5))
            colors = {"uptrend": "rgba(0,180,0,0.08)", "downtrend": "rgba(220,0,0,0.08)", "range": "rgba(220,180,0,0.08)"}
            for _, g in regs.groupby((regs != regs.shift()).cumsum()):
                if g.iloc[0] in colors:
                    fig.add_vrect(x0=g.index[0], x1=g.index[-1], fillcolor=colors[g.iloc[0]], line_width=0, layer="below")
            fig.update_layout(height=420, margin=dict(l=0, r=0, t=30, b=0),
                              title="Equity (ozadje: zeleno = uptrend, rdeče = downtrend, rumeno = range)")
            st.plotly_chart(fig, width="stretch")
            if rnd:
                h = go.Figure(go.Histogram(x=rnd["finals"] * 100, nbinsx=50, name="naključne"))
                h.add_vline(x=(res["equity"].iloc[-1] - 1) * 100, line_color="red", line_width=3)
                h.update_layout(height=280, margin=dict(l=0, r=0, t=30, b=0),
                                title="Strategija (rdeča črta) proti naključnim trejdom – donos %")
                st.plotly_chart(h, width="stretch")
            with st.expander(f"Seznam trejdov ({len(res['trades'])})"):
                st.dataframe(res["trades"], hide_index=True, width="stretch")

    if b2.button("🧪 Walk-forward test robustnosti"):
        with st.spinner("Testiram vse kombinacije parametrov..."):
            wf = walk_forward(load(sym, interval), n, regf, 0.7, S["risk"]["cost"])
        if wf is None:
            st.warning("Strategija nima določenih parametrov za test.")
        else:
            st.markdown(f"### {wf['verdict']}")
            st.caption(f"Učenje do {wf['split_date']:%d.%m.%Y}, test po tem datumu (podatki, ki jih optimizacija ni videla).")
            metric_row({"Stabilnost %": wf["stability_%"],
                        "Najboljši IS Sharpe": wf["best_is_sharpe"],
                        "Isti parametri OOS Sharpe": wf["best_oos_sharpe"],
                        "Privzeti OOS Sharpe": wf["default_oos"]["Sharpe"],
                        "Ohranjeno (OOS/IS)": wf["decay"]})
            st.markdown(f"**Najboljši parametri na učenju:** `{wf['best_params']}` · "
                        f"**privzeti:** `{wf['default_params']}`")
            st.info("Kako brati: **Stabilnost** = delež vseh nastavitev parametrov, ki so na testu v plusu "
                    "(>70 % je dobro). **Ohranjeno** = koliko Sharpa ostane izven učenja (>0,5 je dobro). "
                    "Če najboljši parametri na testu padejo, privzeti pa zdržijo – ostani pri privzetih.")
            st.dataframe(wf["table"], hide_index=True, width="stretch")

# ================================================================ LAB
with tab_l:
    st.subheader("🎯 Trade lab – premiki, SL/TP backtest, izboljšave, live")
    if st.session_state.get("lab_strat") not in LABS:
        st.session_state["lab_strat"] = "breakout_ab"
    if st.session_state.get("lab_sym") not in S["markets"]:
        st.session_state["lab_sym"] = list(S["markets"])[0]
    c1, c2, c3 = st.columns([2, 3, 2])
    lsym = c1.selectbox("Trg", list(S["markets"]), key="lab_sym",
                        format_func=lambda x: f"{x} – {S['markets'][x]['ime']}")
    lname = c2.selectbox("Strategija", list(LABS), key="lab_strat", format_func=lambda x: LABS[x]["label"])
    entry_mode = c3.radio("Vstop", ["open", "close"], key=ss("lab_entry", "open"), horizontal=True,
                          format_func=lambda x: {"open": "odprtje naslednje sveče",
                                                 "close": "zaprtje signalne sveče"}[x])
    st.caption(f"Časovni okvir: **{interval}** (stranska vrstica) · " + LABS[lname]["doc"])
    od_l, do_l = period_inputs("lab")
    pp = LABS[lname]["params"]
    if pp:
        with st.expander("⚙️ Parametri strategije"):
            pc = st.columns(4)
            lparams = {k: pc[j % 4].number_input(k, key=ss(f"lp_{lname}_{k}", v)) for j, (k, v) in enumerate(pp.items())}
    else:
        lparams = {}
        st.caption("🧩 Strategija iz graditelja – pravila urejaš v zavihku **Graditelj**.")
    with st.expander("🔧 Korekcije (jedro strategije ostane enako – dodatke vklapljaš tu)"):
        corr = corrections_ui("lab")
    c1, c2, c3 = st.columns(3)
    risk_pct = c1.number_input("Tveganje na trejd (% računa)", 0.1, 5.0, step=0.1, key=ss("lab_risk", 1.0)) / 100
    zz = c2.number_input("Zigzag prag % (0 = avto)", 0.0, 80.0, step=0.5, key=ss("lab_zz", 0.0)) / 100
    chart_kind = c3.radio("Graf", ["sveče", "linija"], horizontal=True, key=ss("lab_kind", "sveče"))

    if st.button("▶️ Zaženi analizo", type="primary", key="lab_run"):
        with st.spinner("Računam (backtest, naključni test, mreža SL/TP)..."):
            try:
                dfl = load(lsym, interval)
                st.session_state.lab = {"res": run_lab(dfl, lname, lparams, corr, entry_mode, od_l, do_l, risk_pct,
                                                       zz or None, S["risk"]["cost"], n_random, strategies=LABS),
                                        "df": dfl, "sym": lsym, "name": lname, "tf": interval,
                                        "params": lparams, "corr": corr, "entry_mode": entry_mode,
                                        "od": od_l, "do": do_l, "risk": risk_pct}
            except Exception as e:
                st.error(f"Napaka: {e}")
    L = st.session_state.get("lab")
    if L:
        st.caption(f"Rezultati za **{L['sym']} · {LABS.get(L['name'], {'label': L['name']})['label']} · {L['tf']}** "
                   f"(po spremembi nastavitev ponovno klikni Zaženi)")

    # ---------- shranjene nastavitve
    with st.expander(f"💾 Shranjene nastavitve ({len(S['lab_saved'])})"):
        c1, c2 = st.columns([3, 1])
        default_name = f"{LABS[lname]['label'].split(' (')[0]} · {lsym} · {interval}"
        sname = c1.text_input("Ime za shranjevanje", value=default_name, key="lab_save_name")
        if c2.button("💾 Shrani trenutno", width="stretch"):
            keys = LAB_KEYS + [k for k in st.session_state if str(k).startswith(f"lp_{lname}_")]
            cfg = {k: _jsonable(st.session_state[k]) for k in keys if k in st.session_state}
            rec = {"cfg": cfg, "shranjeno": pd.Timestamp.now().strftime("%d.%m.%Y %H:%M"),
                   "trg": lsym, "strategija": LABS[lname]["label"], "tf": interval}
            if L and L["sym"] == lsym and L["name"] == lname and L["tf"] == interval:
                stt = L["res"]["stats"]
                rnd = L["res"]["random"]
                rec["rezultat"] = {"Trejdov": stt.get("Trejdov"), "Win %": stt.get("Win %"),
                                   "Povp. R": stt.get("Povp. R"), "Letno %": L["res"]["account"].get("Letno %"),
                                   "Max DD %": L["res"]["account"].get("Max DD %"),
                                   "Ocena": rnd["verdict"] if rnd else None}
            S["lab_saved"][sname] = json.loads(json.dumps(rec, default=float))
            save()
        if S["lab_saved"]:
            rows = [{"Ime": k, "Trg": v.get("trg"), "Strategija": v.get("strategija"), "TF": v.get("tf"),
                     **(v.get("rezultat") or {}), "Shranjeno": v.get("shranjeno")} for k, v in S["lab_saved"].items()]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            c1, c2, c3 = st.columns([3, 1, 1])
            pick = c1.selectbox("Izberi", list(S["lab_saved"]), key="lab_pick")
            if c2.button("📂 Naloži", width="stretch"):
                cfg = S["lab_saved"][pick]["cfg"]
                pend = {k: _from_json(k, v) for k, v in cfg.items()}
                bad = []
                if pend.get("lab_sym") not in S["markets"]:
                    bad.append("trg")
                    pend.pop("lab_sym", None)
                if pend.get("lab_strat") not in LABS:
                    bad.append("strategija")
                    pend.pop("lab_strat", None)
                if bad:
                    st.warning("Ni več na voljo: " + ", ".join(bad))
                st.session_state["pending"] = pend
                st.session_state.pop("lab", None)
                st.rerun()
            if c3.button("🗑 Izbriši", width="stretch"):
                S["lab_saved"].pop(pick, None)
                save()
                st.rerun()
        else:
            st.caption("Še nič shranjenega. Nastavi strategijo, korekcije in obdobje, klikni Zaženi in nato Shrani.")

    # ---------- obvestila
    with st.expander("🔔 Telegram obvestila (nov signal / izhod)"):
        if not notify.has_token():
            st.markdown(
                "**Nastavitev (enkrat, ~5 min):**\n"
                "1. V Telegramu poišči **@BotFather**, pošlji `/newbot`, izberi ime → dobiš **token**.\n"
                "2. Odpri svojega novega bota in mu pošlji poljubno sporočilo (npr. `zivjo`).\n"
                "3. V Streamlit **Secrets** dodaj `TELEGRAM_TOKEN = \"...\"` in shrani.\n"
                "4. Vrni se sem in klikni **Poišči moj chat ID** → dodaj `TELEGRAM_CHAT_ID = \"...\"` v Secrets.\n"
                "5. Oba (TELEGRAM_TOKEN, TELEGRAM_CHAT_ID) dodaj še na GitHubu: Settings → Secrets → Actions.")
        else:
            c1, c2 = st.columns(2)
            if c1.button("🔎 Poišči moj chat ID"):
                try:
                    ch = notify.find_chats()
                    st.write(ch or "Ni sporočil – najprej pošlji botu sporočilo v Telegramu.")
                except Exception as e:
                    st.error(str(e))
            if c2.button("📨 Pošlji testno sporočilo", disabled=not notify.enabled()):
                try:
                    notify.send("✅ Algo portfelj: obvestila delujejo.")
                    st.success("Poslano.")
                except Exception as e:
                    st.error(str(e))
        S.setdefault("alerts", {})
        if S["lab_saved"]:
            on_now = [k for k, v in S["alerts"].items() if v.get("on") and k in S["lab_saved"]]
            sel = st.multiselect("Obvestila za shranjene nastavitve", list(S["lab_saved"]), default=on_now,
                                 key="alerts_sel")
            if st.button("💾 Shrani obvestila"):
                S["alerts"] = {k: {"on": True} for k in sel}
                save()
            st.caption("Preverjanje teče vsakih 15 min na GitHubu (tudi ko je aplikacija zaprta). "
                       "Sporoči nov signal na zadnji zaprti sveči in izhod odprtega trejda (TP/SL/čas).")
        else:
            st.caption("Najprej shrani vsaj eno nastavitev (💾 Shranjene nastavitve).")

    lt1, lt2, lt3, lt4, lt5, lt6, lt7 = st.tabs(["📏 Kalkulator premika", "📊 Backtest SL/TP", "🛠 Izboljšave",
                                                 "🌐 Več trgov", "🏆 FTMO", "📈 Graf", "🔴 Live"])

    # ---------- 1) kalkulator premika
    with lt1:
        if not L or not L["res"]["move"]:
            st.info("Klikni **Zaženi analizo**.")
        else:
            mv = L["res"]["move"]
            cur = mv["cur"]
            st.caption(f"Zigzag prag {mv['pct'] * 100:.2f} % – premik se šteje za končan, ko se cena vrne za več kot prag. "
                       f"Zgodovinskih premikov v isti smeri: {mv['n_same']} (od tega večjih od trenutnega: {mv['n_surv']}).")
            metric_row({"Trenutni premik": f"{'⬆️' if cur['smer'] == 'gor' else '⬇️'} {cur['smer']}",
                        "Velikost do zdaj %": cur["velikost_%"], "Traja svečk": cur["svečk"],
                        "Povratek od ekstrema %": cur["povratek_%"],
                        "Še do obrata %": cur["do_obrata_%"]})
            metric_row({"Verjetnost nadaljevanja (še ≥ 1 prag) %": cur["nadaljevanje_%"],
                        "Mediana še svečk": None if cur["še_svečk_med"] != cur["še_svečk_med"] else round(cur["še_svečk_med"], 1),
                        "Začetek premika": f"{cur['začetek']:%d.%m.%Y %H:%M}", "Od cene": round(cur["od_cene"], 4),
                        "Trenutna cena": round(cur["cena"], 4)})
            st.markdown("**Cilji cene** (verjetnost, da premik doseže vsaj to velikost, glede na to, da je že tako velik)")
            st.dataframe(mv["targets"], hide_index=True, width="stretch")
            st.markdown("**Porazdelitev vseh premikov** (percentili)")
            st.dataframe(mv["dist"], hide_index=True, width="stretch")
            legs = mv["legs"]
            same = legs[legs["smer"] == cur["smer"]]
            if len(same):
                h = go.Figure(go.Histogram(x=same["velikost_%"], nbinsx=40, name="premiki"))
                h.add_vline(x=cur["velikost_%"], line_color="red", line_width=3)
                h.update_layout(height=260, margin=dict(l=0, r=0, t=30, b=0),
                                title=f"Velikost premikov '{cur['smer']}' (%) – rdeča = trenutni")
                st.plotly_chart(h, width="stretch")
            st.caption("⚠️ Zadnji (trenutni) premik se lahko še spremeni – zigzag ga potrdi šele ob obratu.")

    # ---------- 2) backtest
    with lt2:
        if not L:
            st.info("Klikni **Zaženi analizo**.")
        else:
            r = L["res"]
            st.caption(f"Signalov strategije: {r['n_signals']} · po korekcijah: {r['n_after_corr']} · "
                       f"izvedenih trejdov v obdobju: {len(r['trades'])} (en trejd naenkrat). "
                       "SL/TP se preverjata med svečo (high/low); če sveča zadene oba → šteje SL.")
            if r["stats"].get("Trejdov", 0) == 0:
                st.warning("V izbranem obdobju ni zaprtih trejdov.")
            else:
                metric_row(r["stats"])
                metric_row(r["account"])
                rnd = r["random"]
                if rnd:
                    st.markdown(f"### {rnd['verdict']} — povp. R boljši od **{rnd['percentile']} %** naključnih "
                                f"(mediana naključnih {rnd['median_R']}R, top 5 % {rnd['top5_R']}R)")
                    h = go.Figure(go.Histogram(x=rnd["means"], nbinsx=50, name="naključni"))
                    h.add_vline(x=r["stats"]["Povp. R"], line_color="red", line_width=3)
                    h.update_layout(height=250, margin=dict(l=0, r=0, t=30, b=0),
                                    title="Povprečni R: strategija (rdeča) proti 1000 naključnim izbiram vstopov")
                    st.plotly_chart(h, width="stretch")
                eq = r["equity"]
                if len(eq):
                    f = go.Figure(go.Scatter(x=eq.index, y=eq.values, name="račun", line=dict(width=2)))
                    f.update_layout(height=320, margin=dict(l=0, r=0, t=30, b=0),
                                    title=f"Račun 10.000 € pri {risk_pct * 100:.1f} % tveganja na trejd")
                    st.plotly_chart(f, width="stretch")
                h2 = go.Figure(go.Histogram(x=r["trades"]["R"], nbinsx=40))
                h2.update_layout(height=220, margin=dict(l=0, r=0, t=30, b=0), title="Porazdelitev rezultatov (R)")
                st.plotly_chart(h2, width="stretch")
            op = r["trades"][r["trades"]["razlog"] == "odprt"]
            if len(op):
                st.info(f"🟢 Odprt trejd: {op.iloc[-1]['smer']} od {op.iloc[-1]['vstop']:%d.%m.%Y}, "
                        f"vstop {op.iloc[-1]['vstopna']:.4f}, SL {op.iloc[-1]['SL']:.4f}, TP {op.iloc[-1]['TP']:.4f}, "
                        f"trenutno {op.iloc[-1]['R']:+.2f}R")
            with st.expander(f"Seznam trejdov ({len(r['trades'])})"):
                show = r["trades"].drop(columns=[c for c in r["trades"].columns if c.startswith("_")])
                st.dataframe(show.round({k: 4 for k in show.select_dtypes("number").columns}), hide_index=True, width="stretch")

    # ---------- 3) izboljšave
    with lt3:
        if not L or L["res"]["stats"].get("Trejdov", 0) == 0:
            st.info("Klikni **Zaženi analizo** (potrebni so zaprti trejdi).")
        else:
            r = L["res"]
            t = r["trades"][r["trades"]["razlog"] != "odprt"]
            st.markdown("#### MAE / MFE – kako daleč gre trejd proti tebi in v tvojo korist")
            win, los = t[t["R"] > 0], t[t["R"] <= 0]
            tips = []
            if len(win):
                for x in (0.25, 0.5, 0.75):
                    tips.append(f"{(win['MAE_R'] < x).mean() * 100:.0f} % zmagovalcev ni nikoli šlo več kot {x}R proti tebi")
            if len(los):
                for x in (0.5, 1.0):
                    tips.append(f"{(los['MFE_R'] >= x).mean() * 100:.0f} % poražencev je bilo prej vsaj +{x}R v plusu")
            st.markdown("\n".join(f"- {s}" for s in tips))
            st.caption("Če veliko zmagovalcev nikoli ne gre daleč proti tebi → SL je morda preširok. "
                       "Če je veliko poražencev bilo prej v plusu → preizkusi break-even ali trailing v Korekcijah.")
            f = go.Figure()
            f.add_trace(go.Scatter(x=t["MAE_R"], y=t["MFE_R"], mode="markers",
                                   marker=dict(color=np.where(t["R"] > 0, "#2ca02c", "#d62728"), size=7, opacity=0.7),
                                   text=[f"{v:+.2f}R · {rz}" for v, rz in zip(t["R"], t["razlog"])]))
            f.update_layout(height=360, margin=dict(l=0, r=0, t=30, b=0), xaxis_title="MAE (R proti tebi)",
                            yaxis_title="MFE (R v tvojo korist)", title="Vsak trejd: zeleno = dobiček, rdeče = izguba")
            st.plotly_chart(f, width="stretch")

            c1, c2 = st.columns(2)
            c1.markdown("#### Po stanju trga (režim ob vstopu)")
            c1.dataframe(r["by_regime"], width="stretch")
            c2.markdown("#### Po dnevih vstopa")
            c2.dataframe(r["by_day"], width="stretch")
            if len(r["by_hour"]):
                st.markdown("#### Po urah vstopa (UTC)")
                st.dataframe(r["by_hour"].T, width="stretch")

            st.markdown("#### Mreža SL / TP")
            st.caption("SL × = množitelj SL strategije · TP (R) = TP kot večkratnik novega SL. "
                       "⚠️ Ne izberi samo najboljše celice – iskana je **stabilna regija** (več sosednjih zelenih), "
                       "sicer je to overfitting. Preveri jo še z walk-forward / drugim obdobjem.")
            g = r["grid"]
            if len(g):
                for metric_name in ("Povp. R", "Skupaj R"):
                    pv = g.pivot(index="SL ×", columns="TP (R)", values=metric_name)
                    hm = go.Figure(go.Heatmap(z=pv.values, x=[f"TP {c}R" for c in pv.columns],
                                              y=[f"SL ×{i}" for i in pv.index], colorscale="RdYlGn", zmid=0,
                                              text=np.round(pv.values, 2), texttemplate="%{text}"))
                    hm.update_layout(height=320, margin=dict(l=0, r=0, t=30, b=0), title=metric_name)
                    st.plotly_chart(hm, width="stretch")
                with st.expander("Tabela mreže"):
                    st.dataframe(g, hide_index=True, width="stretch")

            st.markdown("#### 🧪 Walk-forward (zaščita pred overfittingom)")
            st.caption("Mrežo SL/TP najprej preizkusi na prvem delu obdobja (učenje), nato najboljšo nastavitev "
                       "na delu, ki ga optimizacija ni videla (test).")
            split = st.slider("Delež za učenje", 0.5, 0.85, 0.7, 0.05, key="lab_wf_split")
            if st.button("🧪 Zaženi walk-forward", key="lab_wf"):
                with st.spinner("Računam..."):
                    dfw = L["df"]
                    basew = LABS[L["name"]]["func"](dfw, L["params"])
                    sb = apply_corrections(dfw, basew, {**L["corr"], "sl_mult": 1.0, "tp_mult": 1.0, "no_tp": False})
                    sc = apply_corrections(dfw, basew, L["corr"])
                    st.session_state.lab_wf_res = lab_walkforward(
                        dfw, sb, sc, L["corr"], S["risk"]["cost"], L["entry_mode"],
                        period_mask(dfw.index, L["od"], L["do"]), split)
            wf = st.session_state.get("lab_wf_res")
            if wf:
                st.markdown(f"### {wf['verdict']}")
                st.caption(f"Učenje do {wf['split_date']:%d.%m.%Y}, test po tem.")
                b = wf["best"]
                metric_row({"Stabilnost (celic v plusu na testu) %": wf["stability_%"],
                            "Najboljša na učenju": f"SL ×{b['SL ×']} · TP {b['TP (R)']}R" if b is not None else "—",
                            "… njen povp. R na testu": b["Povp. R test"] if b is not None else None,
                            "Tvoja nastavitev – R učenje": wf["current_is"].get("Povp. R"),
                            "Tvoja nastavitev – R test": wf["current_oos"].get("Povp. R")})
                st.info("Robustno = večina celic mreže je v plusu tudi na testu in najboljša nastavitev ne pade. "
                        "Če najboljša na testu pade, tvoja pa zdrži – ostani pri svoji.")
                st.dataframe(wf["table"], hide_index=True, width="stretch")

    # ---------- več trgov
    with lt4:
        st.caption("Ista strategija s temi nastavitvami (parametri, korekcije, obdobje) na več trgih naenkrat. "
                   "Yahoo simboli, ločeni z vejico.")
        crypto15 = ("BTC-USD, ETH-USD, XRP-USD, ADA-USD, DOGE-USD, LTC-USD, BCH-USD, LINK-USD, DOT-USD, "
                    "UNI7083-USD, BNB-USD, TRX-USD, XLM-USD, ETC-USD, EOS-USD")
        c1, c2 = st.columns([4, 1])
        if c2.button("15 kovancev A+B", width="stretch"):
            st.session_state["pending"] = {"mm_syms": crypto15}
            st.rerun()
        if c2.button("Moji trgi", width="stretch"):
            st.session_state["pending"] = {"mm_syms": ", ".join(S["markets"])}
            st.rerun()
        syms = c1.text_area("Trgi", key=ss("mm_syms", ", ".join(S["markets"])), height=80)
        if st.button("🌐 Testiraj na vseh", type="primary", key="mm_run"):
            lst = [s.strip() for s in syms.replace("\n", ",").split(",") if s.strip()]
            data, prog = {}, st.progress(0.0)
            for k, s_ in enumerate(lst):
                prog.progress((k + 1) / len(lst), f"Nalagam {s_}")
                try:
                    data[s_] = load(s_, interval)
                except Exception as e:
                    data[s_] = e
            prog.empty()
            with st.spinner("Testiram..."):
                st.session_state.lab_mm = multi_market(data, LABS[lname]["func"], lparams, corr, entry_mode,
                                                       od_l, do_l, S["risk"]["cost"], min(n_random, 500))
                st.session_state.lab_mm_label = f"{LABS[lname]['label']} · {interval}"
        mm = st.session_state.get("lab_mm")
        if mm:
            st.markdown(f"**{st.session_state.get('lab_mm_label', '')}** · skupaj: {mm['verdict']}"
                        + (f" — povp. R boljši od **{mm['percentile']} %** naključnih" if mm["percentile"] is not None else ""))
            metric_row({k: v for k, v in mm["pooled"].items() if k in ("Trejdov", "Win %", "Povp. R", "Skupaj R",
                                                                        "Profit factor", "Max zaporednih izgub")})
            st.dataframe(mm["table"], hide_index=True, width="stretch")
            if len(mm["trades"]):
                eq, acc = account_curve(mm["trades"], risk_pct)
                metric_row(acc)
                f = go.Figure(go.Scatter(x=eq.index, y=eq.values, line=dict(width=2)))
                f.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0),
                                title=f"Vsi trgi skupaj, {risk_pct * 100:.1f} % tveganja na trejd "
                                      "(brez omejitve števila hkratnih pozicij)")
                st.plotly_chart(f, width="stretch")
                st.caption("Skupne trejde lahko preizkusiš v zavihku 🏆 FTMO.")

    # ---------- FTMO
    with lt5:
        st.caption("FTMO 2-step na trejdih: faza 1 cilj +10 %, faza 2 +5 %, max dnevna izguba 5 %, max skupna 10 %, "
                   "vsaj 4 dni trgovanja. Tveganje je od začetnega računa. Med trejdom se upošteva najslabša točka (MAE). "
                   "Simulacija začne challenge pri vsakem trejdu.")
        src_opts = ["Lab (en trg)"] + (["Več trgov (skupaj)"] if st.session_state.get("lab_mm") else [])
        c1, c2, c3, c4 = st.columns(4)
        fsrc = c1.radio("Trejdi", src_opts, key="ftmo_src")
        frisk = c2.number_input("Tveganje na trejd %", 0.1, 5.0, step=0.1, key=ss("ftmo_risk", 1.0)) / 100
        fguard = c3.toggle("Varovalka (−3 % → ½, −6 % → ¼, blizu cilja ½)", key=ss("ftmo_guard", True))
        fdays = c4.number_input("Časovna omejitev dni (0 = brez)", 0, 2000, key=ss("ftmo_days", 0))
        if st.button("🏆 Simuliraj FTMO", type="primary", key="ftmo_run"):
            trs = (st.session_state["lab_mm"]["trades"] if fsrc.startswith("Več") else (L["res"]["trades"] if L else None))
            if trs is None or len(trs) < 5:
                st.warning("Premalo trejdov – najprej zaženi analizo.")
            else:
                st.session_state.lab_ftmo = ftmo_trades(trs, frisk, fguard, fdays)
        fr = st.session_state.get("lab_ftmo")
        if fr and fr[1]:
            res, summ = fr
            metric_row(summ)
            colr = np.where(res["rezultat"] == "PASS (obe fazi)", "#2ca02c",
                            np.where(res["rezultat"].str.contains("FAIL"), "#d62728", "#bbbbbb"))
            f = go.Figure(go.Bar(x=res["začetek"], y=res["dni"], marker_color=colr,
                                 text=res["rezultat"], hoverinfo="text+x+y"))
            f.update_layout(height=280, margin=dict(l=0, r=0, t=30, b=0),
                            title="Dni do konca glede na začetek (zeleno PASS, rdeče FAIL, sivo nedokončano)")
            st.plotly_chart(f, width="stretch")
            st.dataframe(res["rezultat"].value_counts().rename("število"), width="stretch")

    # ---------- graf
    with lt6:
        if not L:
            st.info("Klikni **Zaženi analizo**.")
        else:
            r = L["res"]
            dfc = L["df"][r["pmask"]] if r["pmask"].sum() > 5 else L["df"]
            mv = r["move"]
            st.plotly_chart(lab_chart(dfc, r["trades"], mv["pivots"] if mv else None,
                                      mv["targets"] if mv else None, chart_kind, 650,
                                      f"{L['sym']} · {L['tf']} – vstopi ▲▼, izstopi ✕, SL rdeče, TP zeleno, zigzag oranžno"),
                            width="stretch")
            st.caption("Povečaš z vlečenjem miške, dvoklik = ponastavi. Oranžno polje desno = ciljna cona "
                       "trenutnega premika (50–25 % verjetnost), pikčasta črta = 10 %.")

    # ---------- 5) live
    with lt7:
        a, b, c = st.columns(3)
        live_on = a.toggle("🔴 Live (samodejno osveževanje)", key="live_on")
        every = b.selectbox("Osveži vsakih", [30, 60, 300, 900], index=1, key="live_every",
                            format_func=lambda s: f"{s} s" if s < 60 else f"{s // 60} min")
        nbars = c.number_input("Prikaži zadnjih svečk", 50, 3000, 200, 50, key="live_n")

        def live_view():
            try:
                bucket = int(time.time() // every) if live_on else 0
                dfv = load_live(lsym, interval, bucket)
            except Exception as e:
                st.error(f"Podatki: {e}")
                return
            sigs = apply_corrections(dfv, LABS[lname]["func"](dfv, lparams), corr)
            tr = simulate(dfv, sigs, S["risk"]["cost"], corr["trail_r"], corr["be_r"], entry_mode)
            mvv = move_calculator(dfv, zz or None)
            last = dfv.iloc[-1]
            chg = (last["close"] / dfv["close"].iloc[-2] - 1) * 100
            st.caption(f"Zadnja posodobitev: {time.strftime('%H:%M:%S')} · zadnja sveča {dfv.index[-1]:%d.%m.%Y %H:%M} UTC "
                       f"(lahko še ni zaprta – signal velja šele ob zaprtju)")
            cur = mvv["cur"]
            metric_row({f"{lsym} cena": f"{last['close']:.4f}", "Sprememba %": round(chg, 2),
                        "Premik": f"{'⬆️' if cur['smer'] == 'gor' else '⬇️'} {cur['velikost_%']} %",
                        "Verjetnost nadaljevanja %": cur["nadaljevanje_%"], "Še do obrata %": cur["do_obrata_%"]})
            op = tr[tr["razlog"] == "odprt"]
            if len(op):
                o = op.iloc[-1]
                px = last["close"]
                st.success(f"🟢 ODPRT TREJD: {o['smer']} od {o['vstop']:%d.%m.%Y %H:%M} · vstop {o['vstopna']:.4f} · "
                           f"SL {o['SL']:.4f} ({(o['SL'] / px - 1) * 100:+.2f} %) · "
                           f"TP {o['TP']:.4f} ({(o['TP'] / px - 1) * 100:+.2f} %) · zdaj {o['R']:+.2f}R")
            else:
                n = len(dfv)
                new = sigs[sigs["i"].astype(int) >= n - 2] if len(sigs) else sigs
                if len(new):
                    s0 = new.iloc[-1]
                    st.warning(f"⚡ NOV SIGNAL na sveči {dfv.index[int(s0['i'])]:%d.%m.%Y %H:%M}: "
                               f"{'long' if s0['side'] > 0 else 'short'} · SL {s0['sl_pct'] * 100:.2f} % · "
                               f"TP {s0['tp_pct'] * 100:.2f} % – vstop ob odprtju naslednje sveče")
                else:
                    last_sig = dfv.index[int(sigs["i"].iloc[-1])] if len(sigs) else None
                    st.info("Ni odprtega trejda in ni novega signala." +
                            (f" Zadnji signal: {last_sig:%d.%m.%Y %H:%M}." if last_sig is not None else ""))
            st.plotly_chart(lab_chart(dfv.iloc[-int(nbars):], tr, mvv["pivots"], mvv["targets"], chart_kind, 520,
                                      f"{lsym} · {interval} · live"), width="stretch")

        st.fragment(run_every=every if live_on else None)(live_view)()

# ================================================================ GRADITELJ
G_MAP = {"ime": "g_ime", "opis": "g_opis", "long": "g_long", "short": "g_short", "izhod_long": "g_xl",
         "izhod_short": "g_xs", "samo_prvi": "g_prvi", "sl": "g_sl", "tp_r": "g_tp", "max_svečk": "g_mb",
         "režimi": "g_reg", "trgi": "g_trg"}


def spec_to_pending(spec):
    spec = B.normalize(spec)
    return {G_MAP[k]: spec[k] for k in G_MAP}


def current_spec():
    return B.normalize({k: st.session_state.get(v) for k, v in G_MAP.items()})


with tab_g:
    st.subheader("🧩 Graditelj strategij")
    st.caption("Strategijo zapišeš kot pravila – brez programiranja in brez GitHub commita. "
               "Shranjena strategija se pojavi v zavihku **🎯 Lab** (z oznako 🧩), kjer jo testiraš z vsemi orodji.")

    c1, c2, c3 = st.columns([2, 2, 1])
    src = c1.selectbox("Začni iz", ["—"] + [f"shranjena: {k}" for k in S["custom_strategies"]] +
                       [f"primer: {k}" for k in B.EXAMPLES] + ["prazna"], key="g_src")
    if c3.button("📂 Odpri", width="stretch") and src != "—":
        if src.startswith("shranjena: "):
            spec = S["custom_strategies"][src[len("shranjena: "):]]
        elif src.startswith("primer: "):
            spec = B.EXAMPLES[src[len("primer: "):]]
        else:
            spec = B.EMPTY_SPEC
        st.session_state["pending"] = spec_to_pending(spec)
        st.rerun()

    # ---------- AI
    with st.expander("🤖 AI pomočnik – opiši strategijo po slovensko", expanded=False):
        if not ai.enabled():
            st.info("Za AI v Streamlit **Secrets** dodaj vrstico `ANTHROPIC_API_KEY = \"sk-ant-...\"` "
                    "(ključ dobiš na console.anthropic.com → API Keys; plačaš po porabi, ena strategija ≈ 1–2 centa). "
                    "Brez ključa: opiši strategijo v chatu s Claudom, on ti da JSON, ki ga prilepiš spodaj v **Uvoz**.")
        desc = st.text_area("Opis (ali sprememba trenutne strategije)", key="g_ai_text", height=110,
                            placeholder="npr. Kupi, ko cena zapre nad 20-dnevnim vrhom in je RSI(14) pod 70, "
                                        "samo ko je cena nad EMA200. SL 2×ATR, TP 3R, izhod po 30 dneh.")
        modify = st.checkbox("Spremeni trenutno strategijo (namesto nove)", key="g_ai_mod")
        if st.button("✨ Ustvari s pomočjo AI", disabled=not ai.enabled()):
            with st.spinner("AI piše pravila..."):
                try:
                    spec = ai.generate(desc, current_spec() if modify else None, interval,
                                       st.session_state.get("lab_sym", ""))
                    st.session_state["pending"] = spec_to_pending(spec)
                    st.session_state["g_ai_ok"] = True
                    st.rerun()
                except Exception as e:
                    st.error(f"AI: {e}")
        if st.session_state.pop("g_ai_ok", False):
            st.success("AI je izpolnil pravila spodaj – preveri jih in klikni ✅ Preveri.")

    # ---------- uvoz / izvoz
    with st.expander("📋 Uvoz / izvoz (JSON) – za delo s Claudom v chatu"):
        c1, c2 = st.columns(2)
        imp = c1.text_area("Prilepi JSON, ki ti ga je dal Claude", key="g_import", height=160)
        if c1.button("⬇️ Uvozi"):
            try:
                st.session_state["pending"] = spec_to_pending(B.parse_json(imp))
                st.rerun()
            except Exception as e:
                c1.error(f"Uvoz: {e}")
        c2.markdown("Trenutna strategija (kopiraj in pošlji Claudu):")
        c2.code(json.dumps(current_spec(), ensure_ascii=False, indent=1), language="json")

    # ---------- urejevalnik
    st.markdown("#### Pravila")
    c1, c2 = st.columns([1, 2])
    c1.text_input("Ime", key=ss("g_ime", ""))
    c2.text_input("Opis", key=ss("g_opis", ""))
    c1, c2 = st.columns(2)
    c1.text_area("🟢 Vstop LONG (prazno = brez longov)", key=ss("g_long", ""), height=90)
    c2.text_area("🔴 Vstop SHORT (prazno = brez shortov)", key=ss("g_short", ""), height=90)
    c1.text_input("Izhod long (neobvezno)", key=ss("g_xl", ""))
    c2.text_input("Izhod short (neobvezno)", key=ss("g_xs", ""))
    c1, c2, c3, c4 = st.columns(4)
    c1.text_input("SL – razdalja v ceni", key=ss("g_sl", "2*atr(14)"), help="npr. 2*atr(14) ali close*0.03")
    c2.number_input("TP (× SL, 0 = brez)", 0.0, 20.0, step=0.5, key=ss("g_tp", 2.0))
    c3.number_input("Časovni izhod (svečk, 0 = brez)", 0, 100000, key=ss("g_mb", 0))
    c4.toggle("Samo prva sveča signala", key=ss("g_prvi", True),
              help="Vstop samo na prvi sveči, ko pogoj postane izpolnjen (ne vsak dan znova)")

    c1, c2 = st.columns(2)
    c1.multiselect("Režimi, v katerih jo portfelj vklopi", ["uptrend", "downtrend", "range"],
                   key=ss("g_reg", ["uptrend", "downtrend", "range"]))
    c2.multiselect("Trgi", ["crypto", "index", "forex", "commodity"],
                   key=ss("g_trg", ["crypto", "index", "forex", "commodity"]))

    with st.expander("📖 Seznam funkcij in primeri"):
        st.table(pd.DataFrame({"Zapis": list(B.HELP), "Pomen": list(B.HELP.values())}))
        st.markdown("Primeri pogojev:\n"
                    "- `close > highest(close,20)[1]` – zaprtje nad najvišjim zaprtjem prejšnjih 20 svečk\n"
                    "- `cross_up(ema(9), ema(21)) in adx(14) > 20`\n"
                    "- `rsi(2) < 10 in close > sma(200)`\n"
                    "- `close < bb_lo(20,2) in dan < 5` – pod Bollingerjem, samo pon–pet\n"
                    "- `roc(90) > 0 in std(20) < 0.04` – 3-mesečni momentum pozitiven, volatilnost < 4 %\n"
                    "- `ura >= 13 in ura <= 16` – samo v newyorški seji (UTC)")

    spec = current_spec()
    c1, c2, c3, c4 = st.columns(4)
    if c1.button("✅ Preveri", type="primary", width="stretch"):
        try:
            dfc = load(st.session_state.get("lab_sym") or list(S["markets"])[0], interval)
            nl, ns, err = B.check_spec(dfc, spec)
            if err:
                st.error(err)
            else:
                st.success(f"Pravila so pravilna · {st.session_state.get('lab_sym')} {interval}: "
                           f"{nl} long in {ns} short signalov v zgodovini.")
        except Exception as e:
            st.error(str(e))
    if c2.button("💾 Shrani strategijo", width="stretch"):
        if not spec["ime"].strip():
            st.error("Vpiši ime.")
        elif not (spec["long"].strip() or spec["short"].strip()):
            st.error("Vpiši vsaj en pogoj za vstop.")
        else:
            S["custom_strategies"][spec["ime"].strip()] = spec
            S["strategies_on"].setdefault("custom:" + spec["ime"].strip(), True)
            register_custom(S["custom_strategies"])
            save()
            st.success(f"Shranjeno kot 🧩 {spec['ime']} – najdeš jo v zavihku Lab, med strategijami portfelja "
                       f"(Strategije / Portfelj / Backtest) in jo lahko trguje paper bot.")
    if c3.button("🎯 Odpri v Lab", width="stretch"):
        nm = spec["ime"].strip() or "neimenovana"
        S["custom_strategies"][nm] = spec
        st.session_state["pending"] = {"lab_strat": "custom:" + nm}
        st.session_state.pop("lab", None)
        st.toast("Izbrano v zavihku 🎯 Lab – odpri ga in klikni Zaženi analizo")
        st.rerun()
    if c4.button("🗑 Izbriši shranjeno", width="stretch"):
        if S["custom_strategies"].pop(spec["ime"].strip(), None) is not None:
            S["strategies_on"].pop("custom:" + spec["ime"].strip(), None)
            register_custom(S["custom_strategies"])
            save()
            st.rerun()
        else:
            st.warning("Strategija s tem imenom ni shranjena.")
    if S["custom_strategies"]:
        st.caption("Shranjene: " + ", ".join(f"🧩 {k}" for k in S["custom_strategies"]))

# ================================================================ DNEVNIK
with tab_j:
    st.subheader("📓 Dnevnik trejdov")
    st.caption("Vpisuj svoje prave (ali paper) trejde. Nova vrstica = klik na + spodaj v tabeli. "
               "Smer: long / short. R se izračuna sam iz vstopa, SL in izstopa.")
    if "journal" not in st.session_state:
        try:
            st.session_state.journal = J.load()
        except Exception as e:
            st.session_state.journal = J.empty()
            st.warning(f"Dnevnika ni bilo mogoče naložiti: {e}")
    names = sorted(set([v["label"] for v in LABS.values()] + list(S["lab_saved"])))
    ed = st.data_editor(
        st.session_state.journal, num_rows="dynamic", width="stretch", key="journal_ed",
        column_config={
            "datum_vstopa": st.column_config.DatetimeColumn("Vstop (datum)", format="DD.MM.YYYY HH:mm"),
            "datum_izstopa": st.column_config.DatetimeColumn("Izstop (datum)", format="DD.MM.YYYY HH:mm"),
            "trg": st.column_config.SelectboxColumn("Trg", options=list(S["markets"]) + ["drugo"]),
            "smer": st.column_config.SelectboxColumn("Smer", options=["long", "short"]),
            "vstop": st.column_config.NumberColumn("Vstopna cena", format="%.5f"),
            "SL": st.column_config.NumberColumn("SL", format="%.5f"),
            "TP": st.column_config.NumberColumn("TP", format="%.5f"),
            "izstop": st.column_config.NumberColumn("Izstopna cena", format="%.5f"),
            "strategija": st.column_config.SelectboxColumn("Strategija", options=names + ["diskrecijsko"]),
            "opomba": st.column_config.TextColumn("Opomba"),
        })
    c1, c2 = st.columns(2)
    if c1.button("💾 Shrani dnevnik", type="primary"):
        st.session_state.journal = J.clean(ed)
        if storage.enabled():
            try:
                J.save(ed)
                st.success("Shranjeno na GitHub.")
            except Exception as e:
                st.error(f"Shranjevanje: {e}")
        else:
            st.warning("GH_TOKEN ni nastavljen – shranjeno samo začasno. Prenesi CSV.")
    c2.download_button("⬇️ Prenesi CSV", J.clean(ed).to_csv(index=False), "dnevnik.csv", "text/csv")

    jd = J.enrich(ed)
    if len(jd):
        metric_row(J.stats(jd))
        z = jd[jd["status"] == "zaprt"].dropna(subset=["R"]).sort_values("datum_izstopa")
        if len(z):
            c1, c2 = st.columns(2)
            f = go.Figure(go.Scatter(x=z["datum_izstopa"], y=z["R"].cumsum(), mode="lines+markers"))
            f.update_layout(height=280, margin=dict(l=0, r=0, t=30, b=0), title="Skupaj R skozi čas")
            c1.plotly_chart(f, width="stretch")
            by = z.groupby("strategija")["R"].agg(["count", "mean", "sum"]).round(2)
            by.columns = ["Trejdov", "Povp. R", "Skupaj R"]
            c2.markdown("**Po strategijah**")
            c2.dataframe(by, width="stretch")
        st.dataframe(jd[["datum_vstopa", "trg", "smer", "R", "donos_%", "status", "strategija"]].round({"R": 2, "donos_%": 2}),
                     hide_index=True, width="stretch")

        st.markdown("#### Primerjava s signali strategije")
        if S["lab_saved"]:
            c1, c2 = st.columns([3, 1])
            pick = c1.selectbox("Shranjena nastavitev", list(S["lab_saved"]), key="j_pick")
            tol = c2.number_input("Toleranca (dni)", 0, 30, 3, key="j_tol")
            if st.button("🔍 Primerjaj"):
                try:
                    from engine.labcfg import cfg_to_run
                    run = cfg_to_run(S["lab_saved"][pick]["cfg"], S["custom_strategies"])
                    dfj = load(run["sym"], run["interval"])
                    sg = apply_corrections(dfj, run["func"](dfj, run["params"]), run["corr"])
                    trj = simulate(dfj, sg, S["risk"]["cost"], run["corr"]["trail_r"], run["corr"]["be_r"],
                                   run["entry_mode"])
                    cmp_ = J.compare(jd, trj, run["sym"], tol)
                    if cmp_.empty:
                        st.info(f"V dnevniku ni trejdov na {run['sym']}.")
                    else:
                        st.dataframe(cmp_, hide_index=True, width="stretch")
                        st.caption("✅ po signalu = vstopil si ob signalu strategije · razlika R = tvoj R − R strategije "
                                   "(pokaže, ali tvoje ročne odločitve izboljšajo ali poslabšajo sistem).")
                except Exception as e:
                    st.error(str(e))
        else:
            st.caption("Za primerjavo najprej shrani nastavitev v Lab-u.")

# ================================================================ PAPER TRADING
with tab_t:
    st.subheader("🤖 Paper trading (Alpaca)")
    st.caption("Bot teče vsak dan ob ~22:15 v GitHub Actions: prebere te nastavitve, izbere strategije "
               "glede na režime, izračuna pozicije z risk managerjem in jih uskladi na Alpaca PAPER računu. "
               "Trguje samo trge, ki obstajajo na Alpaci (kripto, ameriški ETF-i).")
    P = S["paper"]
    c1, c2, c3 = st.columns(3)
    P["enabled"] = c1.toggle("Bot vklopljen", P.get("enabled", False))
    P["only_edge"] = c2.toggle("Samo ✅ edge kombinacije", P.get("only_edge", True))
    P["capital_use"] = c3.slider("Delež kapitala v uporabi", 0.1, 1.0, float(P.get("capital_use", 1.0)), 0.1)
    if st.button("💾 Shrani nastavitve bota"):
        save()
    if not storage.enabled():
        st.warning("Bot bere nastavitve z GitHuba – brez GH_TOKEN v Secrets se spremembe tu ne bodo prenesle do bota.")

    st.markdown("#### Kaj bi bot naredil danes")
    if st.button("🔍 Izračunaj današnje ciljne pozicije"):
        from trader.run_paper import plan
        logs = []
        targets, prices, combos, _ = plan(S, log=logs.append)
        st.code("\n".join(logs) or "—")
        st.dataframe(pd.DataFrame([{"Alpaca simbol": k, "Cilj (% kapitala)": round(v * 100, 1),
                                    "Cena": prices.get(k)} for k, v in targets.items()]),
                     hide_index=True, width="stretch")

    st.markdown("#### Račun")
    if not alpaca.enabled():
        st.info("Za prikaz računa v Streamlit Secrets dodaj ALPACA_KEY in ALPACA_SECRET.")
    else:
        try:
            api = alpaca.Alpaca()
            acc = api.account()
            metric_row({"Kapital $": f"{float(acc['equity']):,.0f}",
                        "Gotovina $": f"{float(acc['cash']):,.0f}",
                        "Danes $": f"{float(acc['equity']) - float(acc['last_equity']):,.0f}"})
            pos = api.positions()
            if pos:
                st.dataframe(pd.DataFrame([{"Simbol": k, **v} for k, v in pos.items()]),
                             hide_index=True, width="stretch")
            hist = api.history("3M")
            if hist.get("equity"):
                hdf = pd.DataFrame({"t": pd.to_datetime(hist["timestamp"], unit="s"), "eq": hist["equity"]}).dropna()
                st.plotly_chart(go.Figure(go.Scatter(x=hdf["t"], y=hdf["eq"])).update_layout(
                    height=280, margin=dict(l=0, r=0, t=30, b=0), title="Kapital (3 mesece)"), width="stretch")
        except Exception as e:
            st.error(f"Alpaca: {e}")

    st.markdown("#### Dnevnik bota")
    log_txt, _ = storage.read_file("trades_log.csv") if storage.enabled() else (None, None)
    if log_txt:
        st.dataframe(pd.read_csv(io.StringIO(log_txt)).iloc[::-1], hide_index=True, width="stretch")
    else:
        st.caption("Še ni zapisov.")
