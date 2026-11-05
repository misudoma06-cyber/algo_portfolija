"""
DNEVNIK TREJDOV – tvoji pravi (ali paper) trejdi, shranjeni v journal.csv na veji "data".
"""
import io

import numpy as np
import pandas as pd

from . import storage

COLS = ["datum_vstopa", "trg", "smer", "vstop", "SL", "TP", "datum_izstopa", "izstop", "strategija", "opomba"]


def empty():
    df = pd.DataFrame({c: pd.Series(dtype="object") for c in COLS})
    for c in ("vstop", "SL", "TP", "izstop"):
        df[c] = pd.Series(dtype="float")
    for c in ("datum_vstopa", "datum_izstopa"):
        df[c] = pd.Series(dtype="datetime64[ns]")
    return df


def clean(df):
    df = df.copy()
    for c in COLS:
        if c not in df:
            df[c] = np.nan
    for c in ("vstop", "SL", "TP", "izstop"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("datum_vstopa", "datum_izstopa"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    for c in ("trg", "smer", "strategija", "opomba"):
        df[c] = df[c].astype("object").where(df[c].notna(), "")
    return df[COLS].dropna(how="all", subset=["datum_vstopa", "trg", "vstop"]).reset_index(drop=True)


def load():
    txt, _ = storage.read_file("journal.csv") if storage.enabled() else (None, None)
    return clean(pd.read_csv(io.StringIO(txt))) if txt else empty()


def save(df):
    storage.write_file("journal.csv", clean(df).to_csv(index=False), "dnevnik trejdov")


def enrich(df):
    d = clean(df)
    side = np.where(d["smer"].astype(str).str.lower().str.startswith("s"), -1, 1)
    risk = (d["vstop"] - d["SL"]).abs()
    d["R"] = np.where(risk > 0, side * (d["izstop"] - d["vstop"]) / risk, np.nan)
    d["donos_%"] = side * (d["izstop"] / d["vstop"] - 1) * 100
    d["status"] = np.where(d["izstop"].notna(), "zaprt", "odprt")
    return d


def stats(d):
    z = d[d["status"] == "zaprt"].dropna(subset=["R"])
    if z.empty:
        return {"Zaprtih": 0}
    r = z["R"]
    loss = -r[r < 0].sum()
    return {"Zaprtih": len(z), "Odprtih": int((d["status"] == "odprt").sum()), "Win %": round((r > 0).mean() * 100, 1),
            "Povp. R": round(r.mean(), 2), "Skupaj R": round(r.sum(), 1),
            "Profit factor": round(r[r > 0].sum() / loss, 2) if loss > 0 else None}


def compare(d, strat_trades, symbol, tol_days=3):
    """Vsak tvoj trejd na tem trgu poveže z najbližjim vstopom strategije (± tol_days)."""
    mine = d[d["trg"] == symbol]
    if mine.empty:
        return pd.DataFrame()
    st = strat_trades.copy()
    st["vstop"] = pd.DatetimeIndex(st["vstop"])
    rows = []
    for _, m in mine.iterrows():
        row = {"datum": m["datum_vstopa"], "tvoja smer": m["smer"], "tvoj R": m.get("R")}
        if len(st) and pd.notna(m["datum_vstopa"]):
            diff = (st["vstop"] - m["datum_vstopa"]).abs()
            j = diff.idxmin()
            if diff[j] <= pd.Timedelta(days=tol_days):
                s = st.loc[j]
                row.update({"signal strategije": s["vstop"], "smer signala": s["smer"], "R signala": round(s["R"], 2),
                            "razlika R": round((m.get("R") or 0) - s["R"], 2) if pd.notna(m.get("R")) else None,
                            "ujemanje": "✅ po signalu" if s["smer"] == str(m["smer"]).lower() else "⚠️ druga smer"})
            else:
                row["ujemanje"] = "❌ ni signala"
        rows.append(row)
    return pd.DataFrame(rows)
