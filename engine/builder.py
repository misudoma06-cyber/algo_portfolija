"""
GRADITELJ STRATEGIJ – strategija kot besedilo pravil (brez programiranja)

Primer:
    long:  close > highest(close,20)[1] in rsi(2) < 10
    sl:    2*atr(14)            (razdalja do SL v ceni)
    tp_r:  2                    (TP = 2 × razdalja SL; 0 = brez TP)

Jezik je varen: dovoljene so samo spodnje funkcije, števila, + - * /, primerjave in in/ali/ne.
[k] pomeni vrednost pred k svečami (close[1] = prejšnje zaprtje).
"""
import ast
import json
import re

import numpy as np
import pandas as pd

from .regime import _adx

HELP = {
    "open, high, low, close, volume": "cena / volumen trenutne sveče",
    "x[k]": "vrednost pred k svečami, npr. close[1], highest(close,20)[1]",
    "ema(n) / ema(src,n)": "eksponentno povprečje (privzeto close)",
    "sma(n) / sma(src,n)": "navadno povprečje",
    "rsi(n) / rsi(src,n)": "RSI",
    "atr(n)": "povprečni razpon sveče (ATR)",
    "adx(n)": "moč trenda (ADX)",
    "highest(n) / highest(src,n)": "najvišja vrednost zadnjih n svečk (privzeto high)",
    "lowest(n) / lowest(src,n)": "najnižja vrednost zadnjih n svečk (privzeto low)",
    "std(n)": "standardni odklon donosov zadnjih n svečk (delež, npr. 0.03 = 3 %)",
    "roc(n) / roc(src,n)": "sprememba v % v zadnjih n svečkah",
    "bb_up(n,k) / bb_lo(n,k)": "Bollinger zgornji / spodnji pas",
    "macd(f,s) / macd_sig(f,s,g)": "MACD linija / signalna linija",
    "stoch(n)": "stohastik %K (0–100)",
    "avg_vol(n)": "povprečni volumen",
    "cross_up(a,b) / cross_down(a,b)": "a prečka b navzgor / navzdol na tej sveči",
    "abs(x), max(a,b), min(a,b), sqrt(x)": "matematika",
    "dan, ura": "dan v tednu (0 = pon … 6 = ned), ura (UTC)",
    "in / ali / ne": "logični pogoji (lahko tudi and / or / not)",
}

EXAMPLES = {
    "Preboj A+B (kripto)": {
        "ime": "Preboj A+B (graditelj)",
        "opis": "20-dnevni preboj zaprtja, močno zaprtje, blizu letnega vrha. SL 1R = σ·√10, TP 2R, 40 dni.",
        "long": "close > highest(close,20)[1] in close[1] <= highest(close,20)[2] "
                "in close/highest(close,20)[1] - 1 >= 0.51*std(20) in close >= 0.72*highest(close,365)",
        "short": "", "izhod_long": "", "izhod_short": "",
        "samo_prvi": False, "sl": "close*std(20)*sqrt(10)", "tp_r": 2.0, "max_svečk": 40,
    },
    "RSI(2) odboj v trendu": {
        "ime": "RSI2 odboj",
        "opis": "Nakup ob močnem kratkem padcu v dolgoročnem trendu, izhod nad SMA5.",
        "long": "close > sma(200) in rsi(2) < 10", "short": "",
        "izhod_long": "close > sma(5)", "izhod_short": "",
        "samo_prvi": True, "sl": "3*atr(14)", "tp_r": 0, "max_svečk": 20,
    },
    "EMA prečkanje + ADX": {
        "ime": "EMA cross ADX",
        "opis": "Prečkanje EMA20/EMA50 v smeri EMA200, samo ko je ADX nad 20.",
        "long": "cross_up(ema(20), ema(50)) in close > ema(200) in adx(14) > 20",
        "short": "cross_down(ema(20), ema(50)) in close < ema(200) in adx(14) > 20",
        "izhod_long": "cross_down(ema(20), ema(50))", "izhod_short": "cross_up(ema(20), ema(50))",
        "samo_prvi": False, "sl": "2*atr(14)", "tp_r": 3.0, "max_svečk": 0,
    },
}

EMPTY_SPEC = {"ime": "", "opis": "", "long": "", "short": "", "izhod_long": "", "izhod_short": "",
              "samo_prvi": True, "sl": "2*atr(14)", "tp_r": 2.0, "max_svečk": 0,
              "režimi": ["uptrend", "downtrend", "range"], "trgi": ["crypto", "index", "forex", "commodity"]}

PRICE = {"open", "high", "low", "close", "volume"}
NO_LIMIT = 10 ** 9


class SpecError(Exception):
    pass


def _prep(text):
    t = (text or "").strip()
    t = re.sub(r"\bin\b", " and ", t)
    t = re.sub(r"\bali\b", " or ", t)
    t = re.sub(r"\bne\b", " not ", t)
    return t.replace("\n", " ")


class Env:
    def __init__(self, df):
        self.df = df
        self.idx = df.index
        self.cache = {}

    def s(self, x):
        if isinstance(x, pd.Series):
            return x
        return pd.Series(float(x), index=self.idx)

    # ---------- funkcije
    def _src_n(self, args, default):
        if len(args) == 1:
            return self.df[default], int(args[0])
        return self.s(args[0]), int(args[1])

    def f_ema(self, *a):
        src, n = self._src_n(a, "close")
        return src.ewm(span=n, adjust=False).mean()

    def f_sma(self, *a):
        src, n = self._src_n(a, "close")
        return src.rolling(n).mean()

    def f_rsi(self, *a):
        src, n = self._src_n(a, "close")
        d = src.diff()
        up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
        dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
        return 100 - 100 / (1 + up / dn.replace(0, np.nan))

    def f_atr(self, n=14):
        df = self.df
        tr = pd.concat([df["high"] - df["low"], (df["high"] - df["close"].shift()).abs(),
                        (df["low"] - df["close"].shift()).abs()], axis=1).max(axis=1)
        return tr.ewm(alpha=1 / int(n), adjust=False).mean()

    def f_adx(self, n=14):
        return _adx(self.df, int(n))[0]

    def f_highest(self, *a):
        src, n = self._src_n(a, "high")
        return src.rolling(n, min_periods=n).max()

    def f_lowest(self, *a):
        src, n = self._src_n(a, "low")
        return src.rolling(n, min_periods=n).min()

    def f_std(self, n=20):
        return self.df["close"].pct_change().rolling(int(n)).std()

    def f_roc(self, *a):
        src, n = self._src_n(a, "close")
        return src.pct_change(n) * 100

    def f_bb_up(self, n=20, k=2):
        c = self.df["close"]
        return c.rolling(int(n)).mean() + k * c.rolling(int(n)).std()

    def f_bb_lo(self, n=20, k=2):
        c = self.df["close"]
        return c.rolling(int(n)).mean() - k * c.rolling(int(n)).std()

    def f_macd(self, f=12, s=26):
        c = self.df["close"]
        return c.ewm(span=int(f), adjust=False).mean() - c.ewm(span=int(s), adjust=False).mean()

    def f_macd_sig(self, f=12, s=26, g=9):
        return self.f_macd(f, s).ewm(span=int(g), adjust=False).mean()

    def f_stoch(self, n=14):
        n = int(n)
        lo, hi = self.df["low"].rolling(n).min(), self.df["high"].rolling(n).max()
        return (self.df["close"] - lo) / (hi - lo).replace(0, np.nan) * 100

    def f_avg_vol(self, n=20):
        return self.df["volume"].rolling(int(n)).mean()

    def f_cross_up(self, a, b):
        a, b = self.s(a), self.s(b)
        return (a > b) & (a.shift(1) <= b.shift(1))

    def f_cross_down(self, a, b):
        a, b = self.s(a), self.s(b)
        return (a < b) & (a.shift(1) >= b.shift(1))

    def f_abs(self, x):
        return self.s(x).abs()

    def f_sqrt(self, x):
        return np.sqrt(x) if not isinstance(x, pd.Series) else np.sqrt(x)

    def f_max(self, a, b):
        return np.maximum(self.s(a), self.s(b))

    def f_min(self, a, b):
        return np.minimum(self.s(a), self.s(b))

    # ---------- evalvacija drevesa
    def ev(self, node):
        if isinstance(node, ast.Expression):
            return self.ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name):
            n = node.id.lower()
            if n in PRICE:
                return self.df[n]
            if n == "dan":
                return pd.Series(self.idx.dayofweek, index=self.idx).astype(float)
            if n == "ura":
                return pd.Series(self.idx.hour, index=self.idx).astype(float)
            raise SpecError(f"Neznano ime '{node.id}'")
        if isinstance(node, ast.UnaryOp):
            v = self.ev(node.operand)
            if isinstance(node.op, ast.USub):
                return -v
            if isinstance(node.op, ast.Not):
                return ~self.s(v).astype(bool)
            raise SpecError("Nedovoljen operator")
        if isinstance(node, ast.BinOp):
            a, b = self.ev(node.left), self.ev(node.right)
            ops = {ast.Add: lambda x, y: x + y, ast.Sub: lambda x, y: x - y,
                   ast.Mult: lambda x, y: x * y, ast.Div: lambda x, y: x / y}
            for t, f in ops.items():
                if isinstance(node.op, t):
                    return f(a, b)
            raise SpecError("Dovoljeni so samo + - * /")
        if isinstance(node, ast.BoolOp):
            vals = [self.s(self.ev(v)).fillna(False).astype(bool) for v in node.values]
            out = vals[0]
            for v in vals[1:]:
                out = (out & v) if isinstance(node.op, ast.And) else (out | v)
            return out
        if isinstance(node, ast.Compare):
            left = self.s(self.ev(node.left))
            out = pd.Series(True, index=self.idx)
            for op, comp in zip(node.ops, node.comparators):
                right = self.s(self.ev(comp))
                if isinstance(op, ast.Gt):
                    r = left > right
                elif isinstance(op, ast.GtE):
                    r = left >= right
                elif isinstance(op, ast.Lt):
                    r = left < right
                elif isinstance(op, ast.LtE):
                    r = left <= right
                elif isinstance(op, ast.Eq):
                    r = left == right
                elif isinstance(op, ast.NotEq):
                    r = left != right
                else:
                    raise SpecError("Nedovoljena primerjava")
                out &= r.fillna(False)
                left = right
            return out
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise SpecError("Nedovoljen klic")
            fn = getattr(self, "f_" + node.func.id.lower(), None)
            if fn is None:
                raise SpecError(f"Neznana funkcija '{node.func.id}'")
            if node.keywords:
                raise SpecError("Uporabi argumente brez imen, npr. ema(20)")
            key = ast.dump(node)
            if key not in self.cache:
                self.cache[key] = fn(*[self.ev(a) for a in node.args])
            return self.cache[key]
        if isinstance(node, ast.Subscript):
            v = self.ev(node.value)
            k = node.slice
            if isinstance(k, ast.Constant) and isinstance(k.value, int) and k.value >= 0:
                return self.s(v).shift(k.value)
            raise SpecError("V [ ] je dovoljeno samo celo število ≥ 0, npr. close[1]")
        raise SpecError(f"Nedovoljen izraz: {type(node).__name__}")

    def run(self, text):
        t = _prep(text)
        if not t:
            return None
        try:
            tree = ast.parse(t, mode="eval")
        except SyntaxError as e:
            raise SpecError(f"Napaka v zapisu: '{text}' ({e.msg})")
        return self.ev(tree)


def _bool(env, text):
    v = env.run(text)
    if v is None:
        return pd.Series(False, index=env.idx)
    return env.s(v).fillna(False).astype(bool)


def spec_signals(df, spec):
    """Spec -> tabela vstopov v obliki, ki jo uporablja Lab."""
    env = Env(df)
    close = df["close"]
    sl_dist = env.s(env.run(spec.get("sl") or "2*atr(14)"))
    sl_pct = (sl_dist / close).values
    tp_r = float(spec.get("tp_r") or 0)
    mb = int(spec.get("max_svečk") or 0)
    rows = []
    for side, key, xkey in ((1, "long", "izhod_long"), (-1, "short", "izhod_short")):
        cond = _bool(env, spec.get(key))
        if spec.get("samo_prvi"):
            cond = cond & ~cond.shift(1, fill_value=False)
        ex = np.flatnonzero(_bool(env, spec.get(xkey)).values)
        for i in np.flatnonzero(cond.values):
            sp = sl_pct[i]
            if not (sp == sp and sp > 0):
                continue
            k = np.searchsorted(ex, i + 1)
            rows.append({"i": int(i), "side": side, "sl_pct": float(sp),
                         "tp_pct": tp_r * float(sp) if tp_r > 0 else np.nan,
                         "max_bars": mb if mb > 0 else NO_LIMIT,
                         "exit_i": float(ex[k]) if k < len(ex) else np.nan})
    if not rows:
        return pd.DataFrame(columns=["i", "side", "sl_pct", "tp_pct", "max_bars", "exit_i"])
    return pd.DataFrame(rows).sort_values("i").reset_index(drop=True)


def check_spec(df, spec):
    """Vrne (število long, število short, napaka ali None)."""
    try:
        s = spec_signals(df, spec)
        return int((s["side"] > 0).sum()), int((s["side"] < 0).sum()), None
    except SpecError as e:
        return 0, 0, str(e)
    except Exception as e:
        return 0, 0, f"Napaka: {e}"


def normalize(spec):
    out = dict(EMPTY_SPEC)
    for k in EMPTY_SPEC:
        if k in spec and spec[k] is not None:
            out[k] = spec[k]
    out["tp_r"] = float(out["tp_r"] or 0)
    out["max_svečk"] = int(out["max_svečk"] or 0)
    out["samo_prvi"] = bool(out["samo_prvi"])
    out["režimi"] = [r for r in (out["režimi"] or []) if r in EMPTY_SPEC["režimi"]] or list(EMPTY_SPEC["režimi"])
    out["trgi"] = [r for r in (out["trgi"] or []) if r in EMPTY_SPEC["trgi"]] or list(EMPTY_SPEC["trgi"])
    return out


def parse_json(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        raise SpecError("Ni najdenega JSON ({ ... })")
    return normalize(json.loads(m.group(0)))
