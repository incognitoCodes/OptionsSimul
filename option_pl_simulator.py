# file: option_pl_simulator.py
from __future__ import annotations
import argparse, datetime as dt, math, re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np, pandas as pd

# UI deps
try:
    import streamlit as st
    import plotly.graph_objects as go
    import requests
except Exception:
    st = None; go = None; requests = None

# -------- OpenAPI SDK import (Moomoo/Futu) --------
OpenQuoteContext = None
RET_OK = -1

class _SubTypeShim:  # avoids NameError if SDK missing
    QUOTE = object()

SubType = _SubTypeShim

# moomoo (some envs ship the SDK under this name)
try:
    from moomoo import OpenQuoteContext as _MQC, RET_OK as _RET, SubType as _ST
    OpenQuoteContext, RET_OK, SubType = _MQC, _RET, _ST
except Exception:
    pass

# futu-api (official pip pkg name)
if OpenQuoteContext is None:
    try:
        from futu import OpenQuoteContext as _FQC, RET_OK as _FRET, SubType as _FST
        OpenQuoteContext, RET_OK, SubType = _FQC, _FRET, _FST
    except Exception:
        pass

MULTIPLIER = 100  # contracts

# ===================== Models =====================
@dataclass
class OptionLeg:
    ticker: str
    expiry: dt.date
    kind: str        # "C" or "P"
    strike: float
    qty: int         # +long / -short
    avg_cost: float  # premium/contract
    name: str = ""
    source_id: Optional[str] = None

    def payoff_at_expiry(self, s: float) -> float:
        intrinsic = max(s - self.strike, 0.0) if self.kind == "C" else max(self.strike - s, 0.0)
        per = (intrinsic - self.avg_cost) if self.qty > 0 else (self.avg_cost - intrinsic)
        return per * abs(self.qty) * MULTIPLIER

    def label(self) -> str:
        side = "L" if self.qty > 0 else "S"
        return f"{self.ticker} {self.expiry:%Y-%m-%d} {side}{self.kind}{self.strike:g} x{abs(self.qty)} @ {self.avg_cost}"

@dataclass
class Strategy:
    name: str
    legs: List[OptionLeg] = field(default_factory=list)
    enabled: bool = True
    user_tag: str = ""  # "current" or "what-if"

    def tickers(self) -> List[str]:
        return sorted({l.ticker for l in self.legs})

    def payoff_at_expiry(self, s: float, ticker: Optional[str] = None) -> float:
        return sum(l.payoff_at_expiry(s) for l in self.legs if self.enabled and (ticker is None or l.ticker == ticker))

    def describe(self) -> str:
        return "\n".join([f"{self.name} [{'ON' if self.enabled else 'OFF'}]"] + [f"  - {l.label()}" for l in self.legs])

# ===================== CSV parsing =====================
EXPECTED_COLUMNS = {"Symbol","Name","Quantity","Current price","Average Cost"}
LEG_SYMBOL_RE = re.compile(r"^(?P<ticker>[A-Z]+?)(?P<expiry>\d{6})(?P<kind>[CP])(?P<strike>\d{3,6})(?:\.\d+)?$")
SPREAD_SYMBOL_RE = re.compile(r"^(?P<ticker>[A-Z]+?)(?P<expiry>\d{6})(?P<kind>[CP])(?P<strike1>\d{2,4})/(?P<strike2>\d{2,4})$")

def parse_expiry(s: str) -> dt.date:
    return dt.date(2000 + int(s[:2]), int(s[2:4]), int(s[4:6]))

def parse_strike(raw: str) -> float:
    s = str(raw)
    if s.isdigit():             # Moomoo style 207500 -> 207.5
        return float(s) / 1000.0
    try:
        return float(s)
    except ValueError:
        return float(re.sub(r"[^\d.]", "", s))

def parse_moomoo_positions(df: pd.DataFrame) -> Tuple[List[Strategy], List[OptionLeg]]:
    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    strategies: Dict[str, Strategy] = {}
    legs: List[OptionLeg] = []
    for _, row in df.iterrows():
        symbol = str(row["Symbol"]).strip()
        name = str(row.get("Name", "")).strip()
        qty = int(float(row["Quantity"]))
        avg = float(row.get("Average Cost", 0.0))
        if SPREAD_SYMBOL_RE.match(symbol):
            if name and name not in strategies:
                strategies[name] = Strategy(name, [], True, "current")
            continue
        m = LEG_SYMBOL_RE.match(symbol.replace(" ", "")) or (LEG_SYMBOL_RE.match(name.replace(" ", "")) if name else None)
        if not m:
            continue
        leg = OptionLeg(
            ticker=m.group("ticker"),
            expiry=parse_expiry(m.group("expiry")),
            kind=m.group("kind"),
            strike=parse_strike(m.group("strike")),
            qty=qty,
            avg_cost=abs(avg),
            name=name,
            source_id=symbol,
        )
        legs.append(leg)
        if name:
            strategies.setdefault(name, Strategy(name, [], True, "current")).legs.append(leg)
        else:
            sname = f"{leg.ticker} {leg.expiry} {leg.kind}{leg.strike:g} Single"
            strategies[sname] = Strategy(sname, [leg], True, "current")
    return list(strategies.values()), legs

# ===================== Math helpers =====================
def price_grid(center: float, pct: float = 0.5, steps: int = 201) -> np.ndarray:
    low  = max(0.0, center * (1 - pct))
    high = center * (1 + pct)
    return np.linspace(low, high, steps)

def strategy_pl_curve(s: Strategy, xs: np.ndarray, tk: str) -> np.ndarray:
    return np.array([s.payoff_at_expiry(x, ticker=tk) for x in xs])

def breakevens(xs: np.ndarray, ys: np.ndarray) -> List[float]:
    idx = np.where(np.diff(np.sign(ys)) != 0)[0]
    outs: List[float] = []
    for i in idx:
        x0, x1 = xs[i], xs[i+1]
        y0, y1 = ys[i], ys[i+1]
        if y1 != y0:
            outs.append(round(float(x0 - y0*(x1-x0)/(y1-y0)), 4))
    return outs

# ===================== Builders =====================
def build_single(t, exp, k, strike, qty, price, tag="what-if") -> Strategy:
    return Strategy(f"{t} {exp} {('Long' if qty>0 else 'Short')}{k} {strike:g}",
                    [OptionLeg(t, exp, k, strike, qty, abs(price))], True, tag)

def build_vertical(t, exp, k, short_k, long_k, qty, credit, tag="what-if") -> Strategy:
    return Strategy(f"{t} {exp} {k} Vertical {long_k:g}/{short_k:g}", [
        OptionLeg(t, exp, k, short_k, -abs(qty), abs(credit)),
        OptionLeg(t, exp, k, long_k,  +abs(qty), 0.0),
    ], True, tag)

def build_iron_condor(t, exp, sp, lp, sc, lc, qty, credit, tag="what-if") -> Strategy:
    return Strategy(f"{t} {exp} Iron Condor P:{lp:g}/{sp:g} C:{sc:g}/{lc:g}", [
        OptionLeg(t, exp, "P", sp, -abs(qty), abs(credit)/2.0),
        OptionLeg(t, exp, "P", lp, +abs(qty), 0.0),
        OptionLeg(t, exp, "C", sc, -abs(qty), abs(credit)/2.0),
        OptionLeg(t, exp, "C", lc, +abs(qty), 0.0),
    ], True, tag)

def build_strangle(t, exp, pk, ck, qty, credit, tag="what-if") -> Strategy:
    short = credit >= 0
    return Strategy(f"{t} {exp} {'Short' if short else 'Long'} Strangle {pk:g}/{ck:g}", [
        OptionLeg(t, exp, "P", pk, -abs(qty) if short else +abs(qty), abs(credit)/2.0),
        OptionLeg(t, exp, "C", ck, -abs(qty) if short else +abs(qty), abs(credit)/2.0),
    ], True, tag)

def build_iron_butterfly(t, exp, center_k, put_wing, call_wing, qty, credit, tag="what-if") -> Strategy:
    return Strategy(f"{t} {exp} Iron Butterfly {put_wing:g}-{center_k:g}-{call_wing:g}", [
        OptionLeg(t, exp, "P", center_k, -abs(qty), abs(credit)/2.0),
        OptionLeg(t, exp, "C", center_k, -abs(qty), abs(credit)/2.0),
        OptionLeg(t, exp, "P", put_wing, +abs(qty), 0.0),
        OptionLeg(t, exp, "C", call_wing, +abs(qty), 0.0),
    ], True, tag)

# ===================== Providers (Yahoo removed) =====================
class QuoteProvider:
    Moomoo = "Moomoo/Futu OpenAPI (OpenD)"
    Tradier = "Tradier"
    Polygon = "Polygon.io"

def _mm_symbol(ticker: str) -> str:
    return f"US.{ticker.strip().upper()}"

MOOMOO_DEFAULT_HOST = "127.0.0.1"
MOOMOO_DEFAULT_QUOTE_PORT = 11111

def _with_moomoo_quote_ctx(host: str, port: int):
    if OpenQuoteContext is None:
        raise RuntimeError("OpenAPI SDK missing. Install: pip install futu-api")
    return OpenQuoteContext(host=host or MOOMOO_DEFAULT_HOST, port=port or MOOMOO_DEFAULT_QUOTE_PORT)

def moomoo_fetch_spot(ticker: str, host: str, port: int) -> Optional[float]:
    code = _mm_symbol(ticker)
    ctx = _with_moomoo_quote_ctx(host, port)
    try:
        ret, _ = ctx.subscribe(code, SubType.QUOTE, push=False)
        if ret != RET_OK:
            return None
        ret, df = ctx.get_stock_quote(code)
        if ret != RET_OK or df is None or df.empty:
            return None
        for col in ("last_price", "cur_price"):
            if col in df.columns and pd.notna(df[col].iloc[0]):
                return float(df[col].iloc[0])
        return None
    finally:
        ctx.close()

def moomoo_get_available_strikes(ticker: str, expiry: dt.date, host: str, port: int) -> List[float]:
    code = _mm_symbol(ticker)
    ctx = _with_moomoo_quote_ctx(host, port)
    try:
        ret, exps = ctx.get_option_expiration_date(code=code)
        if ret != RET_OK or exps is None or exps.empty:
            return []
        exps = exps.copy()
        exps["d"] = pd.to_datetime(exps["strike_time"]).dt.date
        want = min(exps["d"], key=lambda d: abs((d - expiry).days))
        ret, chain = ctx.get_option_chain(code=code, start=want.strftime("%Y-%m-%d"), end=want.strftime("%Y-%m-%d"))
        if ret != RET_OK or chain is None or chain.empty:
            return []
        strikes = chain["strike_price"].astype(float).unique().tolist()
        return sorted(set(map(float, strikes)))
    finally:
        ctx.close()

def moomoo_fetch_option_mid(ticker: str, expiry: dt.date, kind: str, strike: float, host: str, port: int) -> Optional[float]:
    code = _mm_symbol(ticker)
    ctx = _with_moomoo_quote_ctx(host, port)
    try:
        ret, exps = ctx.get_option_expiration_date(code=code)
        if ret != RET_OK or exps is None or exps.empty:
            return None
        exps = exps.copy(); exps["d"] = pd.to_datetime(exps["strike_time"]).dt.date
        want = min(exps["d"], key=lambda d: abs((d - expiry).days))
        ret, chain = ctx.get_option_chain(code=code, start=want.strftime("%Y-%m-%d"), end=want.strftime("%Y-%m-%d"))
        if ret != RET_OK or chain is None or chain.empty:
            return None
        tbl = chain.copy()
        tbl["strike_price"] = tbl["strike_price"].astype(float)
        tbl["option_type"] = tbl["option_type"].astype(str).str.upper().str[0]
        side = "C" if kind.upper().startswith("C") else "P"
        cands = tbl[tbl["option_type"] == side]
        if cands.empty:
            return None
        row = cands.iloc[(cands["strike_price"] - float(strike)).abs().argsort()[:1]]
        opt_code = row["code"].iloc[0]
        ret, _ = ctx.subscribe(opt_code, SubType.QUOTE, push=False)
        if ret != RET_OK:
            return None
        ret, q = ctx.get_stock_quote(opt_code)
        if ret != RET_OK or q is None or q.empty:
            return None
        bid = q["bid_price"].iloc[0] if "bid_price" in q.columns else np.nan
        ask = q["ask_price"].iloc[0] if "ask_price" in q.columns else np.nan
        last = q["last_price"].iloc[0] if "last_price" in q.columns else np.nan
        if pd.notna(bid) and pd.notna(ask) and float(ask) > 0:
            return (float(bid) + float(ask)) / 2.0
        if pd.notna(last) and float(last) > 0:
            return float(last)
        return None
    finally:
        ctx.close()

# ===================== Streamlit UI =====================
def st_app():
    st.set_page_config(page_title="Options P/L-at-Expiry", layout="wide")
    st.title("📈 Options P/L-at-Expiry Simulator")
    st.caption("Default provider: Moomoo/Futu OpenAPI (OpenD). Upload Moomoo CSV • Build what-ifs.")

    if OpenQuoteContext is None:
        st.warning("OpenAPI SDK not installed. Run: `pip install futu-api`")

    uploaded = st.sidebar.file_uploader("Positions CSV", type=["csv"])
    use_sample = st.sidebar.checkbox("Load sample data", value=True)

    if uploaded is not None:
        df = pd.read_csv(uploaded)
    elif use_sample:
        df = sample_positions_df()
    else:
        st.warning("Upload a CSV or load the sample to begin."); st.stop()

    try:
        strategies, legs = parse_moomoo_positions(df)
    except Exception as e:
        st.error(f"Failed to parse CSV: {e}"); st.dataframe(df.head(50)); st.stop()

    tickers = sorted({l.ticker for l in legs}) or ["NVDA"]

    st.sidebar.header("Quotes Provider")
    provider = st.sidebar.selectbox("Provider",
        [QuoteProvider.Moomoo, QuoteProvider.Tradier, QuoteProvider.Polygon], index=0)

    mm_host = st.sidebar.text_input("OpenD Host", value="127.0.0.1") if provider == QuoteProvider.Moomoo else ""
    mm_port = st.sidebar.number_input("OpenD Quote Port", value=11111, step=1) if provider == QuoteProvider.Moomoo else 0
    tradier_token = st.sidebar.text_input("Tradier Token", type="password") if provider == QuoteProvider.Tradier else ""
    polygon_key  = st.sidebar.text_input("Polygon API Key", type="password") if provider == QuoteProvider.Polygon else ""

    st.sidebar.header("Underlying Spot Prices")
    spot_inputs: Dict[str, float] = {}
    for tk in tickers:
        if f"spot_{tk}" not in st.session_state:
            ks = [l.strike for l in legs if l.ticker == tk]
            st.session_state[f"spot_{tk}"] = float(np.median(ks)) if ks else 500.0
        spot_inputs[tk] = st.sidebar.number_input(
            f"{tk} spot", min_value=0.0, value=float(st.session_state[f"spot_{tk}"]), step=1.0, key=f"spot_{tk}"
        )
        if st.sidebar.button(f"Fetch {tk} spot", key=f"btn_spot_{tk}"):
            fetched = None
            if provider == QuoteProvider.Moomoo and OpenQuoteContext is not None:
                try:
                    fetched = moomoo_fetch_spot(tk, mm_host, int(mm_port))
                except Exception as e:
                    st.sidebar.warning(f"Moomoo error: {e}")
            elif provider == QuoteProvider.Tradier and requests:
                try:
                    r = requests.get("https://api.tradier.com/v1/markets/quotes",
                                     params={"symbols": tk},
                                     headers={"Authorization": f"Bearer {tradier_token}", "Accept": "application/json"},
                                     timeout=8); r.raise_for_status()
                    fetched = float(r.json()["quotes"]["quote"]["last"])
                except Exception as e:
                    st.sidebar.warning(f"Tradier error: {e}")
            elif provider == QuoteProvider.Polygon and requests:
                try:
                    r = requests.get(f"https://api.polygon.io/v2/last/trade/{tk}",
                                     params={"apiKey": polygon_key}, timeout=8); r.raise_for_status()
                    fetched = float(r.json()["results"]["p"])
                except Exception as e:
                    st.sidebar.warning(f"Polygon error: {e}")
            if fetched:
                st.session_state[f"spot_{tk}"] = float(fetched); st.experimental_rerun()
            else:
                st.sidebar.info(f"{tk}: live price unavailable. Using {st.session_state[f'spot_{tk}']:.2f}.")

    st.sidebar.header("Price Grid")
    pct   = st.sidebar.slider("Grid width (±%)", 5, 80, 50, 5) / 100.0
    steps = st.sidebar.slider("Grid steps", 101, 801, 401, 50)

    st.header("Current Strategies (from CSV)")
    for s in strategies:
        s.enabled = st.checkbox(s.name, True, key=f"cur_{s.name}")
    with st.expander("Show strategy details"):
        for s in strategies: st.text(s.describe())

    st.header("Add What-If Strategies")
    with st.expander("Builder"):
        tabs = st.tabs(["Single","Vertical","Iron Condor","Iron Butterfly","Strangle/Straddle"])
        what_if: List[Strategy] = st.session_state.get("what_if", [])

        with tabs[0]:
            tk   = st.selectbox("Ticker", options=tickers, key="single_tk")
            exp  = st.date_input("Expiry", value=dt.date.today()+dt.timedelta(days=7), key="single_exp")
            kind = st.selectbox("Type", options=["C","P"], key="single_kind")
            strike = st.number_input("Strike", min_value=0.0, value=float(round(spot_inputs.get(tk,500.0))), step=1.0, key="single_strike")
            qty = st.number_input("Contracts (+long / -short)", value=-1, step=1, key="single_qty")
            price_key = "single_price"
            _ = st.number_input("Premium per contract", min_value=0.0, value=1.0, step=0.05, key=price_key)
            if st.button("Auto price from API", key="single_auto"):
                prem = None
                if provider == QuoteProvider.Moomoo and OpenQuoteContext is not None:
                    try:
                        prem = moomoo_fetch_option_mid(tk, exp, kind, float(strike), mm_host, int(mm_port))
                    except Exception as e:
                        st.warning(f"Moomoo error: {e}")
                if prem is not None:
                    st.session_state[price_key] = float(prem); st.experimental_rerun()
                else:
                    st.warning("No quote found for that maturity/strike.")
            if st.button("➕ Add Single"):
                what_if.append(build_single(tk,exp,kind,float(strike),int(qty),float(st.session_state[price_key]))); st.success("Added Single")

        with tabs[1]:
            tk   = st.selectbox("Ticker", options=tickers, key="vert_tk")
            exp  = st.date_input("Expiry", value=dt.date.today()+dt.timedelta(days=7), key="vert_exp")
            kind = st.selectbox("Type", options=["C","P"], key="vert_kind")
            s1 = st.number_input("Short strike", min_value=0.0, value=float(round(spot_inputs.get(tk,500.0))), step=1.0, key="vert_short")
            s2 = st.number_input("Long strike",  min_value=0.0, value=float(max(0.0, round(spot_inputs.get(tk,500.0)-5))), step=1.0, key="vert_long")
            qty = st.number_input("Contracts (abs)", value=1, step=1, key="vert_qty")
            credit_key = "vert_credit"; _ = st.number_input("Net credit (per contract)", min_value=0.0, value=1.0, step=0.05, key=credit_key)
            if st.button("Auto price from API", key="vert_auto"):
                short_mid = long_mid = None
                if provider == QuoteProvider.Moomoo and OpenQuoteContext is not None:
                    try:
                        short_mid = moomoo_fetch_option_mid(tk, exp, kind, float(s1), mm_host, int(mm_port))
                        long_mid  = moomoo_fetch_option_mid(tk, exp, kind, float(s2), mm_host, int(mm_port))
                    except Exception as e:
                        st.warning(f"Moomoo error: {e}")
                if short_mid is not None and long_mid is not None:
                    st.session_state[credit_key] = max(0.0, float(short_mid - long_mid)); st.experimental_rerun()
                else:
                    st.warning("Could not fetch both legs.")
            if st.button("➕ Add Vertical"):
                ss, ls = float(s1), float(s2)
                if kind == "P" and ls > ss: ls, ss = ss, ls
                if kind == "C" and ss > ls: ls, ss = ss, ls
                what_if.append(build_vertical(tk,exp,kind,ss,ls,int(qty),float(st.session_state[credit_key]))); st.success("Added Vertical")

        with tabs[2]:
            tk = st.selectbox("Ticker", options=tickers, key="ic_tk")
            exp = st.date_input("Expiry", value=dt.date.today()+dt.timedelta(days=7), key="ic_exp")
            sp = st.number_input("Short put",  min_value=0.0, value=float(max(1.0, round(spot_inputs.get(tk,500.0)-10))), step=1.0, key="ic_sp")
            lp = st.number_input("Long put",   min_value=0.0, value=float(max(0.0, round(sp-5))), step=1.0, key="ic_lp")
            sc = st.number_input("Short call", min_value=0.0, value=float(round(spot_inputs.get(tk,500.0)+10)), step=1.0, key="ic_sc")
            lc = st.number_input("Long call",  min_value=0.0, value=float(round(sc+5)), step=1.0, key="ic_lc")
            qty = st.number_input("Contracts (abs)", value=1, step=1, key="ic_qty")
            credit_key = "ic_credit"; _ = st.number_input("Net credit (per contract)", min_value=0.0, value=1.0, step=0.05, key=credit_key)
            if st.button("Auto price from API", key="ic_auto"):
                pm  = plm = cm = clm = None
                if provider == QuoteProvider.Moomoo and OpenQuoteContext is not None:
                    try:
                        pm  = moomoo_fetch_option_mid(tk, exp, "P", float(sp), mm_host, int(mm_port))
                        plm = moomoo_fetch_option_mid(tk, exp, "P", float(lp), mm_host, int(mm_port))
                        cm  = moomoo_fetch_option_mid(tk, exp, "C", float(sc), mm_host, int(mm_port))
                        clm = moomoo_fetch_option_mid(tk, exp, "C", float(lc), mm_host, int(mm_port))
                    except Exception as e:
                        st.warning(f"Moomoo error: {e}")
                if all(x is not None for x in [pm, plm, cm, clm]):
                    st.session_state[credit_key] = max(0.0, float((pm - plm) + (cm - clm))); st.experimental_rerun()
                else:
                    st.warning("Missing one or more legs from API.")
            if st.button("➕ Add Iron Condor"):
                what_if.append(build_iron_condor(tk,exp,float(sp),float(lp),float(sc),float(lc),int(qty),float(st.session_state[credit_key]))); st.success("Added Iron Condor")

        with tabs[3]:
            tk = st.selectbox("Ticker", options=tickers, key="ib_tk")
            exp = st.date_input("Expiry", value=dt.date.today()+dt.timedelta(days=7), key="ib_exp")
            center = st.number_input("Center strike (short straddle)", min_value=0.0, value=float(round(spot_inputs.get(tk,500.0))), step=1.0, key="ib_center")
            width  = st.number_input("Wing width", min_value=0.0, value=5.0, step=0.5, key="ib_width")
            if st.button("🔧 Snap to market", key="ib_snap"):
                strikes = []
                if provider == QuoteProvider.Moomoo and OpenQuoteContext is not None:
                    try:
                        strikes = moomoo_get_available_strikes(tk, exp, mm_host, int(mm_port))
                    except Exception as e:
                        st.warning(f"Moomoo error: {e}")
                if strikes:
                    nearest_center = min(strikes, key=lambda s: abs(s - float(st.session_state["ib_center"])))
                    st.session_state["ib_center"] = float(nearest_center); st.experimental_rerun()
                else:
                    st.warning("Could not load market strikes to snap.")
            put_wing  = float(max(0.0, st.session_state["ib_center"] - st.session_state["ib_width"]))
            call_wing = float(st.session_state["ib_center"] + st.session_state["ib_width"])
            st.write(f"Wings ⇢ Put {put_wing:g}, Call {call_wing:g}")
            qty = st.number_input("Contracts (abs)", value=1, step=1, key="ib_qty")
            credit_key = "ib_credit"; _ = st.number_input("Net credit (per contract)", min_value=0.0, value=1.0, step=0.05, key=credit_key)
            if st.button("Auto price from API", key="ib_auto"):
                pm = cm = plm = clm = None
                if provider == QuoteProvider.Moomoo and OpenQuoteContext is not None:
                    try:
                        pm  = moomoo_fetch_option_mid(tk, exp, "P", st.session_state["ib_center"], mm_host, int(mm_port))
                        cm  = moomoo_fetch_option_mid(tk, exp, "C", st.session_state["ib_center"], mm_host, int(mm_port))
                        plm = moomoo_fetch_option_mid(tk, exp, "P", put_wing, mm_host, int(mm_port))
                        clm = moomoo_fetch_option_mid(tk, exp, "C", call_wing, mm_host, int(mm_port))
                    except Exception as e:
                        st.warning(f"Moomoo error: {e}")
                if all(x is not None for x in [pm, cm, plm, clm]):
                    st.session_state[credit_key] = max(0.0, float((pm + cm) - (plm + clm))); st.experimental_rerun()
                else:
                    st.warning("Missing one or more legs from API.")
            if st.button("➕ Add Iron Butterfly"):
                what_if.append(build_iron_butterfly(tk,exp,float(st.session_state["ib_center"]),float(put_wing),float(call_wing),int(qty),float(st.session_state[credit_key]))); st.success("Added Iron Butterfly")

        with tabs[4]:
            tk = st.selectbox("Ticker", options=tickers, key="sg_tk")
            exp = st.date_input("Expiry", value=dt.date.today()+dt.timedelta(days=7), key="sg_exp")
            put_k  = st.number_input("Put strike",  min_value=0.0, value=float(max(0.0, round(spot_inputs.get(tk,500.0)-10))), step=1.0, key="sg_pk")
            call_k = st.number_input("Call strike", min_value=0.0, value=float(round(spot_inputs.get(tk,500.0)+10)), step=1.0, key="sg_ck")
            qty = st.number_input("Contracts (abs)", value=1, step=1, key="sg_qty")
            credit_key = "sg_credit"; _ = st.number_input("Net credit (>0 short, 0 for long)", min_value=0.0, value=1.0, step=0.05, key=credit_key)
            if st.button("Auto price from API", key="sg_auto"):
                pm = cm = None
                if provider == QuoteProvider.Moomoo and OpenQuoteContext is not None:
                    try:
                        pm = moomoo_fetch_option_mid(tk, exp, "P", float(put_k),  mm_host, int(mm_port))
                        cm = moomoo_fetch_option_mid(tk, exp, "C", float(call_k), mm_host, int(mm_port))
                    except Exception as e:
                        st.warning(f"Moomoo error: {e}")
                if pm is not None and cm is not None:
                    st.session_state[credit_key] = max(0.0, float(pm + cm)); st.experimental_rerun()
                else:
                    st.warning("Could not price both legs.")
            if st.button("➕ Add Strangle/Straddle"):
                what_if.append(build_strangle(tk,exp,float(put_k),float(call_k),int(qty),float(st.session_state[credit_key]))); st.success("Added Strangle/Straddle")
        st.session_state["what_if"] = what_if

    st.header("P/L at Expiry (per Ticker)")
    for tk in tickers:
        xs = price_grid(st.session_state[f"spot_{tk}"], pct, steps)
        cur = sum(strategy_pl_curve(s, xs, tk) for s in strategies if s.enabled and tk in s.tickers())
        add = sum(strategy_pl_curve(s, xs, tk) for s in st.session_state.get("what_if", []) if s.enabled and tk in s.tickers())
        comb = cur + add
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xs, y=cur, name="Current"))
        fig.add_trace(go.Scatter(x=xs, y=comb, name="Current + What-If"))
        fig.add_hline(y=0, line_dash="dash"); fig.add_vline(x=st.session_state[f"spot_{tk}"], line_dash="dot")
        fig.update_layout(height=400, xaxis_title="Underlying at Expiry", yaxis_title="P/L (USD)")
        st.plotly_chart(fig, use_container_width=True)

    st.header("Portfolio (Combined)")
    tk0 = tickers[0]; xs = price_grid(st.session_state[f"spot_{tk0}"], pct, steps)
    def agg(strats: List[Strategy]) -> np.ndarray:
        return sum(strategy_pl_curve(s, xs, tk0) for s in strats if s.enabled and tk0 in s.tickers())
    current = agg(strategies); added = agg(st.session_state.get("what_if", [])); comb = current + added
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=xs, y=current, name="Current"))
    fig2.add_trace(go.Scatter(x=xs, y=comb, name="Current + What-If"))
    fig2.add_hline(y=0, line_dash="dash"); fig2.add_vline(x=st.session_state[f"spot_{tk0}"], line_dash="dot")
    fig2.update_layout(height=400, xaxis_title=f"{tk0} Price at Expiry", yaxis_title="P/L (USD)")
    st.plotly_chart(fig2, use_container_width=True)

    st.download_button("Download sample CSV", data=sample_positions_df().to_csv(index=False),
                       file_name="sample_moomoo_positions.csv", mime="text/csv")

# ===================== Sample + tests =====================
def sample_positions_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"Symbol":"NVDA251031P195/200","Name":"NVDA Vertical","Quantity":-29,"Current price":0.655,"Average Cost":0.995},
        {"Symbol":"NVDA251031P195000","Name":"NVDA 251031 195.00P","Quantity":29,"Current price":0.34,"Average Cost":0.54},
        {"Symbol":"NVDA251031P200000","Name":"NVDA 251031 200.00P","Quantity":-29,"Current price":1.0,"Average Cost":1.535},
    ])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tests", action="store_true")
    args, _ = parser.parse_known_args()
    if args.run_tests:
        leg = OptionLeg("ABC", dt.date(2025,1,1), "C", 100, +1, 2.0)
        assert math.isclose(leg.payoff_at_expiry(120), 1800.0)
        strategies, legs = parse_moomoo_positions(sample_positions_df())
        assert len(legs) == 2
        print("Tests OK"); return
    if st is None:
        print("Install: pip install streamlit plotly pandas numpy requests futu-api")
        print("Run: streamlit run option_pl_simulator.py"); return
    st_app()

if __name__ == "__main__":
    main()
