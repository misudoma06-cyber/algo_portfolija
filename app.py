"""
ALGO PORTFELJ – aplikacija
Zagon lokalno: streamlit run app.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from engine.backtest import apply_regime_filter, random_test, run_backtest, metrics, bars_per_year
from engine.data import get_data
from engine.regime import REGIME_LABELS, detect, regime_series
from engine.strategies import ALL_STRATEGIES

st.set_page_config(page_title="Algo portfelj", page_icon="📈", layout="wide")
BASE = os.path.dirname(os.path.abspath(__file__))
REGIMES = ["uptrend", "downtrend", "range"]

# ---------------------------------------------------------------- stanje
if "settings" not in st.session_state:
    with open(os.path.join(BASE, "settings.json"), encoding="utf-8") as f:
        st.session_state.settings = json.load(f)
    st.session_state.settings.setdefault("overrides", {})
S = st.session_state.settings


@st.cache_data(ttl=3600, show_spinner=False)
def load(symbol, interval):
    return get_data(symbol, interval)


def effective_regime(symbol, auto):
    o = S["overrides"].get(symbol, "auto")
    return auto if o == "auto" else o


# ---------------------------------------------------------------- stranska vrstica
with st.sidebar:
    st.title("📈 Algo portfelj")
    interval = st.radio("Časovni okvir backtesta", ["1d", "1h"], horizontal=True,
                        help="1d = 5 let podatkov, 1h = ~2 leti")
    cost = st.number_input("Stroški na stran (%)", 0.0, 1.0, 0.10, 0.01) / 100
    n_random = st.select_slider("Naključnih trejdov za test", [200, 500, 1000], value=1000)
    st.divider()
    st.download_button("💾 Shrani nastavitve", json.dumps(S, ensure_ascii=False, indent=1),
                       "settings.json", "application/json")
    up = st.file_uploader("📂 Naloži nastavitve", type="json")
    if up is not None and st.button("Uporabi naložene"):
        st.session_state.settings = json.load(up)
        st.session_state.settings.setdefault("overrides", {})
        st.rerun()
    st.caption("Nastavitve se po ponovnem zagonu strežnika ponastavijo – shrani jih.")

tab_m, tab_s, tab_p, tab_b = st.tabs(["🌍 Trgi", "🧠 Strategije", "💼 Portfelj", "🔬 Backtest"])

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
                             "Avto režim": REGIME_LABELS[d["regime"]],
                             "Dni": d["dni_v_rezimu"],
                             "Tvoj režim": S["overrides"].get(sym, "auto"),
                             "1M %": d["1m_%"], "3M %": d["3m_%"],
                             "ADX": d["adx"], "Volatilnost": d["volatilnost"],
                             "_auto": d["regime"]})
            except Exception as e:
                rows.append({"Simbol": sym, "Ime": m["ime"], "Trg": m["trg"],
                             "Avto režim": f"napaka: {e}", "Tvoj režim": "auto", "_auto": "unknown"})
    mdf = pd.DataFrame(rows)
    st.session_state.auto_regimes = dict(zip(mdf["Simbol"], mdf["_auto"]))

    edited = st.data_editor(
        mdf.drop(columns="_auto"), hide_index=True, width="stretch",
        disabled=[c for c in mdf.columns if c != "Tvoj režim"],
        column_config={
            "Tvoj režim": st.column_config.SelectboxColumn(options=["auto"] + REGIMES, required=True),
            "1M %": st.column_config.NumberColumn(format="%.1f"),
            "3M %": st.column_config.NumberColumn(format="%.1f"),
        })
    new_over = dict(zip(edited["Simbol"], edited["Tvoj režim"]))
    if new_over != {k: S["overrides"].get(k, "auto") for k in new_over}:
        S["overrides"] = {k: v for k, v in new_over.items() if v != "auto"}
        st.rerun()

    with st.expander("➕ Dodaj / odstrani trg"):
        c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
        sym = c1.text_input("Simbol (Yahoo)", placeholder="npr. XRP-USD, ^GDAXI, GBPUSD=X")
        ime = c2.text_input("Ime")
        trg = c3.selectbox("Vrsta trga", ["crypto", "index", "forex", "commodity"])
        if c4.button("Dodaj") and sym:
            S["markets"][sym.strip()] = {"ime": ime or sym, "trg": trg}
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
        c1, c2 = st.columns([1, 5])
        S["strategies_on"][name] = c1.toggle("Vklop", S["strategies_on"].get(name, True), key=f"on_{name}")
        doc = (sys.modules[cls.__module__].__doc__ or "").strip().splitlines()
        c2.markdown(f"**{s.label}** · režimi: {', '.join(s.regimes)} · trgi: {', '.join(s.markets)}")
        if len(doc) > 1:
            c2.caption(" · ".join(l.strip() for l in doc[1:] if l.strip()))

    st.divider()
    st.markdown("### Kje ima katera strategija edge?")
    st.caption("Vsaka strategija se testira na vsakem ustreznem trgu, **ločeno po režimih** "
               "(trguje samo v svečkah, ko je bil trg v tem režimu). Edge = boljša od 95 % naključnih trejdov.")
    if st.button("▶️ Ovrednoti vse strategije", type="primary"):
        combos = [(n, sym, r) for n, cls in ALL_STRATEGIES.items()
                  for sym, m in S["markets"].items() if m["trg"] in cls.markets
                  for r in cls.regimes]
        prog = st.progress(0.0)
        out = []
        for i, (n, sym, r) in enumerate(combos):
            prog.progress((i + 1) / len(combos), f"{n} · {sym} · {r}")
            try:
                df = load(sym, interval)
                regs = regime_series(df)
                sig = apply_regime_filter(ALL_STRATEGIES[n]().generate_signals(df), regs, r)
                res = run_backtest(df, sig, cost)
                rnd = random_test(df, res, n_random, allowed=(regs == r))
                out.append({"Strategija": n, "Simbol": sym, "Režim": r, **res["metrics"],
                            "Boljša od naklj. %": rnd["percentile"] if rnd else None,
                            "Ocena": rnd["verdict"] if rnd else "ni trejdov"})
            except Exception as e:
                out.append({"Strategija": n, "Simbol": sym, "Režim": r, "Ocena": f"napaka: {e}"})
        prog.empty()
        st.session_state.edge = pd.DataFrame(out)
        st.session_state.edge_interval = interval

    if "edge" in st.session_state:
        e = st.session_state.edge
        st.caption(f"Časovni okvir: {st.session_state.edge_interval}")
        only = st.checkbox("Pokaži samo ✅ edge", False)
        st.dataframe(e[e["Ocena"] == "✅ edge"] if only else e, hide_index=True, width="stretch")

# ================================================================ PORTFELJ
with tab_p:
    st.subheader("Aktivni portfelj")
    auto = st.session_state.get("auto_regimes", {})
    active = []
    for sym, m in S["markets"].items():
        reg = effective_regime(sym, auto.get(sym, "unknown"))
        for n, cls in ALL_STRATEGIES.items():
            s = cls()
            if S["strategies_on"].get(n) and s.fits(reg, m["trg"]):
                active.append((sym, n, reg))

    if not active:
        st.info("Trenutno nobena vklopljena strategija ne ustreza režimom trgov.")
    else:
        st.dataframe(pd.DataFrame(active, columns=["Simbol", "Strategija", "Režim"]),
                     hide_index=True, width="stretch")
        only_edge = st.checkbox("Uporabi samo kombinacije z ✅ edge (iz zavihka Strategije)", False)
        if only_edge and "edge" in st.session_state:
            ok = st.session_state.edge
            ok = set(map(tuple, ok[ok["Ocena"] == "✅ edge"][["Simbol", "Strategija", "Režim"]].values))
            active = [a for a in active if a in ok]
            st.write(f"Po filtru ostane **{len(active)}** kombinacij.")

        if active and st.button("▶️ Backtest portfelja", type="primary"):
            st.caption("Vsaka kombinacija dobi enak del kapitala. Strategija trguje samo v zgodovinskih "
                       "obdobjih, ko je bil trg v istem režimu kot zdaj.")
            rets, fig = {}, go.Figure()
            for sym, n, reg in active:
                df = load(sym, interval)
                sig = apply_regime_filter(ALL_STRATEGIES[n]().generate_signals(df), regime_series(df), reg)
                res = run_backtest(df, sig, cost)
                r = res["returns"]
                r.index = r.index.normalize() if interval == "1d" else r.index
                rets[f"{sym} · {n}"] = r
                fig.add_trace(go.Scatter(x=res["equity"].index, y=res["equity"], name=f"{sym} · {n}",
                                         line=dict(width=1), opacity=0.5))
            R = pd.DataFrame(rets).fillna(0)
            port = R.mean(axis=1)
            eq = (1 + port).cumprod()
            fig.add_trace(go.Scatter(x=eq.index, y=eq, name="PORTFELJ", line=dict(width=3, color="black")))
            fig.update_layout(height=450, margin=dict(l=0, r=0, t=30, b=0), yaxis_title="kapital (1 = start)")
            m = metrics(port, eq, pd.DataFrame(columns=["donos_%"]), bars_per_year(R.index))
            m.pop("Trejdov"); m.pop("Win rate %"); m.pop("Profit factor")
            cols = st.columns(len(m))
            for c, (k, v) in zip(cols, m.items()):
                c.metric(k, v)
            st.plotly_chart(fig, width="stretch")
            st.markdown("**Korelacija med strategijami** (nižja = boljša razpršenost)")
            st.dataframe(R.corr().round(2), width="stretch")

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

    if st.button("▶️ Zaženi backtest", type="primary"):
        df = load(sym, interval)
        strat = ALL_STRATEGIES[n](**params)
        sig = strat.generate_signals(df)
        regs = regime_series(df)
        allowed = None
        if regf != "vsi":
            sig = apply_regime_filter(sig, regs, regf)
            allowed = regs == regf
        res = run_backtest(df, sig, cost)
        rnd = random_test(df, res, n_random, allowed=allowed)

        cols = st.columns(len(res["metrics"]))
        for c, (k, v) in zip(cols, res["metrics"].items()):
            c.metric(k, v)
        if rnd:
            st.markdown(f"### {rnd['verdict']} — boljša od **{rnd['percentile']} %** naključnih "
                        f"(mediana naključnih {rnd['median_%']} %, top 5 % {rnd['top5_%']} %)")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=res["equity"].index, y=res["equity"], name="strategija"))
        fig.add_trace(go.Scatter(x=df.index, y=df["close"] / df["close"].iloc[0], name="buy & hold", opacity=0.5))
        colors = {"uptrend": "rgba(0,180,0,0.08)", "downtrend": "rgba(220,0,0,0.08)", "range": "rgba(220,180,0,0.08)"}
        blocks = (regs != regs.shift()).cumsum()
        for _, g in regs.groupby(blocks):
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
