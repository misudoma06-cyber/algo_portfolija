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
from engine.data import get_data
from engine.portfolio import active_combos, combo_returns, portfolio_returns
from engine.regime import REGIME_LABELS, detect, regime_series
from engine.risk import FTMO_RULES, ftmo_sim, to_daily
from engine.strategies import ALL_STRATEGIES
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


# ---------------------------------------------------------------- stranska vrstica
with st.sidebar:
    st.title("📈 Algo portfelj")
    interval = st.radio("Časovni okvir backtesta", ["1d", "1h"], horizontal=True,
                        help="1d = 5 let podatkov, 1h = ~2 leti")
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

tab_m, tab_s, tab_p, tab_b, tab_t = st.tabs(
    ["🌍 Trgi", "🧠 Strategije", "💼 Portfelj + FTMO", "🔬 Backtest", "🤖 Paper trading"])

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

    b1, b2 = st.columns(2)
    if b1.button("▶️ Zaženi backtest", type="primary"):
        df = load(sym, interval)
        sig = ALL_STRATEGIES[n](**params).generate_signals(df)
        regs = regime_series(df)
        allowed = None
        if regf != "vsi":
            sig = apply_regime_filter(sig, regs, regf)
            allowed = regs == regf
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
