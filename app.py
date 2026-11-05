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

from engine import alpaca, storage
from engine.backtest import apply_regime_filter, bars_per_year, metrics, random_test, run_backtest
import time

from engine.data import INTERVALS, get_data
from engine.portfolio import active_combos, combo_returns, portfolio_returns
from engine.regime import REGIME_LABELS, detect, regime_series
from engine.risk import FTMO_RULES, ftmo_sim, to_daily
from engine.strategies import ALL_STRATEGIES
from engine.tradelab import (DAYS, DEFAULT_CORR, LAB_STRATEGIES, REG_ALL, apply_corrections,
                             move_calculator, period_mask, run_lab, simulate)
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
    od = a.date_input("Od (prazno = začetek)", value=None, key=f"{key}_od", format="DD.MM.YYYY")
    do = b.date_input("Do (prazno = konec)", value=None, key=f"{key}_do", format="DD.MM.YYYY")
    m = c.text_input("…ali cel mesec (LLLL-MM)", placeholder="npr. 2024-08", key=f"{key}_m")
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
    c["side"] = a.selectbox("Smer", ["oboje", "samo long", "samo short"], key=f"{key}_side")
    c["sl_mult"] = b.number_input("SL × (množitelj)", 0.1, 5.0, 1.0, 0.1, key=f"{key}_slm")
    c["tp_mult"] = cc.number_input("TP × (množitelj)", 0.1, 10.0, 1.0, 0.1, key=f"{key}_tpm")
    c["no_tp"] = d.toggle("Brez TP", key=f"{key}_notp")
    a, b, cc, d = st.columns(4)
    c["max_bars"] = a.number_input("Časovni izhod (svečk, 0 = strategija)", 0, 100000, 0, key=f"{key}_mb")
    c["trail_r"] = b.number_input("Trailing stop (R, 0 = izklop)", 0.0, 10.0, 0.0, 0.25, key=f"{key}_tr")
    c["be_r"] = cc.number_input("SL na break-even po +R (0 = izklop)", 0.0, 10.0, 0.0, 0.25, key=f"{key}_be")
    c["adx_min"] = d.number_input("Min ADX (0 = izklop)", 0, 60, 0, key=f"{key}_adx")
    a, b, cc = st.columns(3)
    c["vol_lo"], c["vol_hi"] = a.slider("Volatilnost – percentil ATR %", 0, 100, (0, 100), key=f"{key}_vol",
                                        help="Npr. 0–50 = trguj samo, ko je trg mirnejši od običajnega")
    c["vol_k"] = b.number_input("Volumen ≥ × povprečja 20 (0 = izklop)", 0.0, 5.0, 0.0, 0.1, key=f"{key}_vk")
    c["regimes"] = cc.multiselect("Samo v režimih", REG_ALL, default=REG_ALL, key=f"{key}_reg")
    a, b = st.columns(2)
    c["days"] = a.multiselect("Dnevi vstopa", list(range(7)), default=list(range(7)),
                              format_func=lambda d: DAYS[d], key=f"{key}_days")
    c["hours"] = b.slider("Ure vstopa (UTC, za intraday)", 0, 23, (0, 23), key=f"{key}_hrs",
                          help="London ≈ 7–16, New York ≈ 13–21 UTC")
    return c


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
    interval = st.radio("Časovni okvir", INTERVALS, horizontal=True,
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

tab_m, tab_s, tab_p, tab_b, tab_l, tab_t = st.tabs(
    ["🌍 Trgi", "🧠 Strategije", "💼 Portfelj + FTMO", "🔬 Backtest", "🎯 Lab", "🤖 Paper trading"])

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
        doc = (sys.modules[cls.__module__].__doc__ or "").strip().splitlines()
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
    c1, c2, c3 = st.columns([2, 3, 2])
    lsym = c1.selectbox("Trg", list(S["markets"]), key="lab_sym",
                        format_func=lambda x: f"{x} – {S['markets'][x]['ime']}")
    lname = c2.selectbox("Strategija", list(LAB_STRATEGIES), key="lab_strat",
                         format_func=lambda x: LAB_STRATEGIES[x]["label"])
    entry_mode = c3.radio("Vstop", ["open", "close"], key="lab_entry", horizontal=True,
                          format_func=lambda x: {"open": "odprtje naslednje sveče",
                                                 "close": "zaprtje signalne sveče"}[x])
    st.caption(f"Časovni okvir: **{interval}** (stranska vrstica) · " + LAB_STRATEGIES[lname]["doc"])
    od_l, do_l = period_inputs("lab")
    with st.expander("⚙️ Parametri strategije"):
        pp = LAB_STRATEGIES[lname]["params"]
        pc = st.columns(4)
        lparams = {k: pc[j % 4].number_input(k, value=v, key=f"lp_{lname}_{k}") for j, (k, v) in enumerate(pp.items())}
    with st.expander("🔧 Korekcije (jedro strategije ostane enako – dodatke vklapljaš tu)"):
        corr = corrections_ui("lab")
    c1, c2, c3 = st.columns(3)
    risk_pct = c1.number_input("Tveganje na trejd (% računa)", 0.1, 5.0, 1.0, 0.1, key="lab_risk") / 100
    zz = c2.number_input("Zigzag prag % (0 = avto)", 0.0, 80.0, 0.0, 0.5, key="lab_zz") / 100
    chart_kind = c3.radio("Graf", ["sveče", "linija"], horizontal=True, key="lab_kind")

    if st.button("▶️ Zaženi analizo", type="primary", key="lab_run"):
        with st.spinner("Računam (backtest, naključni test, mreža SL/TP)..."):
            try:
                dfl = load(lsym, interval)
                st.session_state.lab = {"res": run_lab(dfl, lname, lparams, corr, entry_mode, od_l, do_l, risk_pct,
                                                       zz or None, S["risk"]["cost"], n_random),
                                        "df": dfl, "sym": lsym, "name": lname, "tf": interval}
            except Exception as e:
                st.error(f"Napaka: {e}")
    L = st.session_state.get("lab")
    if L:
        st.caption(f"Rezultati za **{L['sym']} · {LAB_STRATEGIES[L['name']]['label']} · {L['tf']}** "
                   f"(po spremembi nastavitev ponovno klikni Zaženi)")
    lt1, lt2, lt3, lt4, lt5 = st.tabs(["📏 Kalkulator premika", "📊 Backtest SL/TP", "🛠 Izboljšave",
                                       "📈 Graf", "🔴 Live"])

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

    # ---------- 4) graf
    with lt4:
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
    with lt5:
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
            sigs = apply_corrections(dfv, LAB_STRATEGIES[lname]["func"](dfv, lparams), corr)
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
