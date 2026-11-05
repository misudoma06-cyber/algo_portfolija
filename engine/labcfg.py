"""
Shranjena nastavitev Lab-a (cfg) -> vse, kar potrebuješ za izračun (uporablja aplikacija in Telegram bot).
"""
import pandas as pd

from .tradelab import DEFAULT_CORR, all_lab_strategies

CORR_MAP = {"lab_side": "side", "lab_slm": "sl_mult", "lab_tpm": "tp_mult", "lab_notp": "no_tp", "lab_mb": "max_bars",
            "lab_tr": "trail_r", "lab_be": "be_r", "lab_adx": "adx_min", "lab_vk": "vol_k", "lab_reg": "regimes",
            "lab_days": "days", "lab_hrs": "hours"}
TD = {"1d": "1D", "4h": "4h", "1h": "1h", "30m": "30min", "15m": "15min"}


def cfg_to_run(cfg, custom=None):
    labs = all_lab_strategies(custom)
    name = cfg.get("lab_strat", "breakout_ab")
    if name not in labs:
        raise KeyError(f"Strategija '{name}' ne obstaja več")
    pre = f"lp_{name}_"
    params = {**labs[name]["params"], **{k[len(pre):]: v for k, v in cfg.items() if k.startswith(pre)}}
    corr = dict(DEFAULT_CORR)
    for k, c in CORR_MAP.items():
        if k in cfg:
            corr[c] = cfg[k]
    if "lab_vol" in cfg:
        corr["vol_lo"], corr["vol_hi"] = cfg["lab_vol"]
    corr["hours"] = tuple(corr["hours"])
    return {"sym": cfg.get("lab_sym"), "name": name, "label": labs[name]["label"], "func": labs[name]["func"],
            "params": params, "corr": corr, "entry_mode": cfg.get("lab_entry", "open"),
            "interval": cfg.get("interval", "1d"), "risk": float(cfg.get("lab_risk", 1.0)) / 100}


def closed_bars(df, interval):
    """Odstrani zadnjo svečo, če še ni zaprta."""
    now = pd.Timestamp.utcnow().tz_localize(None)
    if len(df) and df.index[-1] + pd.Timedelta(TD.get(interval, "1D")) > now:
        return df.iloc[:-1]
    return df
