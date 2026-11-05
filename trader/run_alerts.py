"""
TELEGRAM BOT – teče vsakih 15 min v GitHub Actions (.github/workflows/alerts.yml)
Za vsako shranjeno Lab nastavitev z vklopljenim obvestilom:
 - NOV SIGNAL na zadnji zaprti sveči (vstop ob odprtju naslednje)
 - IZHOD odprtega trejda (TP / SL / čas / signal)
Stanje (kaj je že bilo poslano) se hrani v alerts_state.json na veji "data".
Ročni preizkus brez pošiljanja:  python trader/run_alerts.py --dry
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import notify, storage  # noqa: E402
from engine.data import get_data  # noqa: E402
from engine.labcfg import cfg_to_run, closed_bars  # noqa: E402
from engine.tradelab import apply_corrections, simulate  # noqa: E402


def check(name, rec, settings, state, send, cost):
    run = cfg_to_run(rec["cfg"], settings.get("custom_strategies"))
    df = closed_bars(get_data(run["sym"], run["interval"]), run["interval"])
    c = run["corr"]
    sigs = apply_corrections(df, run["func"](df, run["params"]), c)
    tr = simulate(df, sigs, cost, c["trail_r"], c["be_r"], run["entry_mode"])
    st = state.setdefault(name, {})
    head = f"<b>{name}</b>\n{run['sym']} · {run['interval']} · {run['label']}"
    msgs = []

    # izhod prej odprtega trejda
    if st.get("open_entry"):
        t = tr[pd.DatetimeIndex(tr["vstop"]) == pd.Timestamp(st["open_entry"])] if len(tr) else tr
        if len(t) and t.iloc[-1]["razlog"] != "odprt":
            x = t.iloc[-1]
            msgs.append(f"🏁 IZHOD ({x['razlog']}) {x['smer']}\n{head}\nvstop {x['vstopna']:.5g} → izstop "
                        f"{x['izstopna']:.5g}\nrezultat {x['R']:+.2f}R ({x['donos_%']:+.2f} %)")
            st.pop("open_entry")
    op = tr[tr["razlog"] == "odprt"]
    if len(op):
        o = op.iloc[-1]
        if st.get("open_entry") != str(o["vstop"]):
            st["open_entry"] = str(o["vstop"])

    # nov signal na zadnji zaprti sveči
    if len(sigs):
        last = sigs.iloc[-1]
        i = int(last["i"])
        tkey = str(df.index[i])
        busy = len(op) and int(op.iloc[-1]["_i"]) < i
        if i == len(df) - 1 and st.get("last_sig") != tkey and not busy:
            px = float(df["close"].iloc[i])
            side = int(last["side"])
            sl = px * (1 - side * last["sl_pct"])
            tp = px * (1 + side * last["tp_pct"]) if last["tp_pct"] == last["tp_pct"] else None
            msgs.append(f"⚡ NOV SIGNAL {'LONG' if side > 0 else 'SHORT'}\n{head}\nzaprtje {df.index[i]:%d.%m.%Y %H:%M} "
                        f"UTC @ {px:.5g}\nSL ≈ {sl:.5g} ({-last['sl_pct'] * 100:.2f} %)"
                        + (f"\nTP ≈ {tp:.5g} ({last['tp_pct'] * 100:+.2f} %)" if tp else "\nTP: brez")
                        + f"\ntveganje {run['risk'] * 100:.1f} % računa · vstop ob odprtju naslednje sveče")
            st["last_sig"] = tkey
    for m in msgs:
        send(m)
    return msgs


def main(dry=False):
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    settings, src = storage.load_settings(os.path.join(here, "settings.json"))
    alerts = {k: v for k, v in settings.get("alerts", {}).items() if v.get("on")}
    print(f"Nastavitve: {src} · aktivnih obvestil: {len(alerts)}")
    if not alerts:
        return
    txt, _ = storage.read_file("alerts_state.json")
    state = json.loads(txt) if txt else {}
    send = print if dry else notify.send
    cost = settings.get("risk", {}).get("cost", 0.001)
    for name in alerts:
        rec = settings.get("lab_saved", {}).get(name)
        if not rec:
            print(f"  ! {name}: shranjena nastavitev ne obstaja")
            continue
        try:
            msgs = check(name, rec, settings, state, send, cost)
            print(f"  {name}: {len(msgs)} sporočil")
        except Exception as e:
            print(f"  ! {name}: {e}")
    if not dry and storage.enabled():
        storage.write_file("alerts_state.json", json.dumps(state, ensure_ascii=False, indent=1), "bot: alerts state")


if __name__ == "__main__":
    main(dry="--dry" in sys.argv)
