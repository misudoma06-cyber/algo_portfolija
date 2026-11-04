"""
PAPER TRADING BOT – teče vsak dan v GitHub Actions (.github/workflows/paper.yml)
1) prebere nastavitve (iz aplikacije, shranjene na GitHubu)
2) zazna režime, izbere aktivne strategije (po želji samo tiste z ✅ edge)
3) izračuna ciljne pozicije z risk managerjem
4) na Alpaca PAPER računu uskladi pozicije
5) zapiše dnevnik v trades_log.csv in equity_log.csv (veja "data")
Ročni preizkus brez trgovanja:  python trader/run_paper.py --dry
"""
import io
import os
import sys
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import storage  # noqa: E402
from engine.alpaca import Alpaca, alpaca_symbol, is_crypto  # noqa: E402
from engine.data import get_data  # noqa: E402
from engine.portfolio import active_combos, target_positions  # noqa: E402
from engine.regime import detect  # noqa: E402

MIN_TRADE = 0.01      # ne trguj, če je sprememba manjša od 1 % kapitala
MAX_GROSS = 1.0       # skupna izpostavljenost največ 100 % kapitala (brez vzvoda)


def plan(settings, log=print):
    """Izračuna ciljne pozicije (delež kapitala na Alpaca simbol). Ne trguje."""
    data, auto = {}, {}
    for sym in settings["markets"]:
        try:
            data[sym] = get_data(sym, "1d")
            auto[sym] = detect(data[sym])["regime"]
        except Exception as e:
            log(f"  ! {sym}: {e}")
    paper = settings.get("paper", {})
    only = {tuple(x) for x in settings.get("edge", [])} if paper.get("only_edge", True) else None
    combos = [c for c in active_combos(settings, auto, only) if c[0] in data]
    log(f"Aktivne kombinacije ({len(combos)}): {combos}")
    raw = target_positions(data, combos, settings["risk"]) if combos else {}

    targets, prices = {}, {}
    for sym, w in raw.items():
        alp = alpaca_symbol(sym, settings["markets"][sym])
        if not alp:
            log(f"  - {sym}: ni na voljo na Alpaci, preskočeno")
            continue
        if is_crypto(alp) and w < 0:
            log(f"  - {sym}: short na kripto na Alpaci ni mogoč -> 0")
            w = 0.0
        targets[alp] = targets.get(alp, 0.0) + w
        prices[alp] = float(data[sym]["close"].iloc[-1])
    gross = sum(abs(v) for v in targets.values())
    cap = MAX_GROSS * paper.get("capital_use", 1.0)
    if gross > cap:
        targets = {k: v * cap / gross for k, v in targets.items()}
    return targets, prices, combos, auto


def rebalance(api, targets, prices, log=print, dry=False):
    acc = api.account()
    equity = float(acc["equity"])
    pos = api.positions()
    trades = []
    for sym in sorted(set(targets) | set(pos)):
        tgt_val = targets.get(sym, 0.0) * equity
        cur = pos.get(sym, {"qty": 0.0, "value": 0.0})
        price = prices.get(sym) or cur.get("price")
        diff_val = tgt_val - cur["value"]
        closing = tgt_val == 0 and cur["qty"] != 0
        if abs(diff_val) < MIN_TRADE * equity and not closing:
            continue
        action = ""
        try:
            flip = cur["qty"] != 0 and (tgt_val == 0 or (tgt_val > 0) != (cur["qty"] > 0))
            if flip:
                action = "zapri"
                if not dry:
                    api.close_position(sym)
                cur_qty = 0.0
            else:
                cur_qty = cur["qty"]
            if tgt_val != 0:
                tgt_qty = tgt_val / price
                if is_crypto(sym):
                    tgt_qty = round(tgt_qty, 6)
                elif tgt_qty < 0:
                    tgt_qty = int(tgt_qty)            # short samo cele delnice
                else:
                    tgt_qty = round(tgt_qty, 4)
                q = round(tgt_qty - cur_qty, 6)
                if q != 0:
                    action += (" + " if action else "") + ("kupi" if q > 0 else "prodaj") + f" {abs(q)}"
                    if not dry:
                        api.order(sym, q)
            status = "DRY" if dry else "OK"
        except Exception as e:
            status = f"NAPAKA: {e}"[:200]
        log(f"  {sym}: cilj {tgt_val:,.0f} $ (zdaj {cur['value']:,.0f} $) -> {action} [{status}]")
        trades.append({"čas": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"), "simbol": sym,
                       "cilj_$": round(tgt_val, 2), "prej_$": round(cur["value"], 2),
                       "akcija": action, "status": status})
    return equity, trades


def append_csv(path, rows):
    if not rows:
        return
    old, _ = storage.read_file(path)
    df = pd.DataFrame(rows)
    if old:
        df = pd.concat([pd.read_csv(io.StringIO(old)), df], ignore_index=True)
    storage.write_file(path, df.to_csv(index=False), f"bot: {path}")


def main(dry=False):
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    settings, src = storage.load_settings(os.path.join(here, "settings.json"))
    print(f"Nastavitve: {src}")
    if not settings.get("paper", {}).get("enabled") and not dry:
        print("Paper trading je IZKLOPLJEN v aplikaciji – končujem.")
        return
    targets, prices, combos, auto = plan(settings)
    print("Ciljne pozicije (delež kapitala):", {k: round(v, 3) for k, v in targets.items()})
    api = Alpaca()
    equity, trades = rebalance(api, targets, prices, dry=dry)
    print(f"Kapital: {equity:,.2f} $")
    if not dry and storage.enabled():
        append_csv("trades_log.csv", trades)
        append_csv("equity_log.csv", [{"datum": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                                       "kapital": equity, "kombinacij": len(combos)}])


if __name__ == "__main__":
    main(dry="--dry" in sys.argv)
