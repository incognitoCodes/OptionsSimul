# file: option_pl_simulator.py
"""
Options P/L-at-Expiry Simulator (with Live Quotes)
- Streamlit UI
- Upload Moomoo CSV (or sample)
- Accurate intrinsic P/L at expiry per leg/strategy/ticker
- What-if builder (Single, Vertical, Iron Condor, Strangle)
- Live quotes: Yahoo (yfinance), Tradier, Polygon (optional)
- Safe defaults & diagnostics

Install
    pip install streamlit plotly pandas numpy yfinance requests

Run
    streamlit run option_pl_simulator.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import datetime as dt
import math
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import streamlit as st
    import plotly.graph_objects as go
    import yfinance as yf
    import requests
except Exception:
    st = None
    go = None
    yf = None
    requests = None

MULTIPLIER = 100

# ============================
# Data Models
# ============================

@dataclass
class OptionLeg:
    ticker: str
    expiry: dt.date
    kind: str  # 'C' or 'P'
    strike: float
    qty: int   # +long, -short
    avg_cost: float  # per contract
    name: str = ""
    source_id: Optional[str] = None

    def payoff_at_expiry(self, underlying: float) -> float:
        intrinsic = max(underlying - self.strike, 0.0) if self.kind == "C" else max(self.strike - underlying, 0.0)
        per_contract_pl = (intrinsic - self.avg_cost) if self.qty > 0 else (self.avg_cost - intrinsic)
        return per_contract_pl * abs(self.qty) * MULTIPLIER

    def label(self) -> str:
        sign = "L" if self.qty > 0 else "S"
        return f"{self.ticker} {self.expiry.strftime('%Y-%m-%d')} {sign}{self.kind}{self.strike:g} x{abs(self.qty)} @ {self.avg_cost}"


@dataclass
class Strategy:
    name: str
    legs: List[OptionLeg] = field(default_factory=list)
    enabled: bool = True
    user_tag: str = ""  # "current" or "what-if"

    def tickers(self) -> List[str]:
        return sorted(list({leg.ticker for leg in self.legs}))

    def expiries(self) -> List[dt.date]:
        return sorted(list({leg.expiry for leg in self.legs}))

    def payoff_at_expiry(self, underlying: float, ticker: Optional[str] = None) -> float:
        total = 0.0
        for leg in self.legs:
            if not self.enabled:
                continue
            if ticker is not None and leg.ticker != ticker:
                continue
            total += leg.payoff_at_expiry(underlying)
        return total

    def describe(self) -> str:
        parts = [f"{self.name} [{'ON' if self.enabled else 'OFF'}]"]
        parts += [f"  - {leg.label()}" for leg in self.legs]
        return "\n".join(parts)


# ============================
# Parse Moomoo CSV
# ============================

EXPECTED_COLUMNS = {"Symbol","Name","Quantity","Current price","Average Cost"}
LEG_SYMBOL_RE = re.compile(r"^(?P<ticker>[A-Z]+?)(?P<expiry>\d{6})(?P<kind>[CP])(?P<strike>\d{3,6})(?:\.\d+)?$")
SPREAD_SYMBOL_RE = re.compile(r"^(?P<ticker>[A-Z]+?)(?P<expiry>\d{6})(?P<kind>[CP])(?P<strike1>\d{2,4})/(?P<strike2>\d{2,4})$")

def parse_expiry(yyMMdd: str) -> dt.date:
    y = 2000 + int(yyMMdd[:2]); m = int(yyMMdd[2:4]); d = int(yyMMdd[4:6])
    return dt.date(y, m, d)

def parse_strike(raw: str) -> float:
    # Moomoo encodes strikes as integer * 1000 (e.g., 195000 -> 195.0, 207500 -> 207.5)
    raw_str = str(raw)
    if raw_str.isdigit():
        return float(raw_str) / 1000.0
    try:
        return float(raw_str)
    except ValueError:
        return float(re.sub(r"[^\d.]", "", raw_str))

def parse_moomoo_positions(df: pd.DataFrame) -> Tuple[List[Strategy], List[OptionLeg]]:
    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    strategies: Dict[str, Strategy] = {}
    legs: List[OptionLeg] = []

    for _, row in df.iterrows():
        symbol = str(row["Symbol"]).strip()
        name = str(row.get("Name","")).strip()
        qty = int(row["Quantity"])
        avg = float(row.get("Average Cost", 0.0))

        if SPREAD_SYMBOL_RE.match(symbol):
            if name and name not in strategies:
                strategies[name] = Strategy(name=name, legs=[], enabled=True, user_tag="current")
            continue

        m = LEG_SYMBOL_RE.match(symbol.replace(" ", ""))
        if not m and name:
            m = LEG_SYMBOL_RE.match(name.replace(" ", ""))

        if not m:
            continue  # ignore non-option rows

        ticker = m.group("ticker")
        expiry = parse_expiry(m.group("expiry"))
        kind = m.group("kind")
        strike = parse_strike(m.group("strike"))

        leg = OptionLeg(ticker, expiry, kind, strike, qty, abs(avg), name=name, source_id=symbol)
        legs.append(leg)

        group_key = None
        if name and ("Vertical" in name or "Iron" in name or "Straddle" in name or "Strangle" in name):
            group_key = name
        elif name and name not in ("", "nan"):
            group_key = name

        if group_key:
            strategies.setdefault(group_key, Strategy(name=group_key, legs=[], enabled=True, user_tag="current"))
            strategies[group_key].legs.append(leg)
        else:
            single = f"{ticker} {expiry} {kind}{strike:g} Single"
            strategies[single] = Strategy(name=single, legs=[leg], enabled=True, user_tag="current")

    return list(strategies.values()), legs


# ============================
# P/L grid
# ============================

def price_grid(center_price: float, pct_width: float = 0.5, steps: int = 201) -> np.ndarray:
    low = max(0.0, center_price * (1.0 - pct_width))
    high = center_price * (1.0 + pct_width)
    return np.linspace(low, high, steps)

def strategy_pl_curve(strategy: Strategy, prices: np.ndarray, ticker: str) -> np.ndarray:
    return np.array([strategy.payoff_at_expiry(p, ticker=ticker) for p in prices])

def aggregate_pl_curves(strategies: List[Strategy], prices: np.ndarray, ticker: Optional[str] = None) -> np.ndarray:
    totals = np.zeros_like(prices, dtype=float)
    for s in strategies:
        if not s.enabled:
            continue
        if ticker:
            if ticker not in s.tickers():
                continue
            totals += strategy_pl_curve(s, prices, ticker)
        else:
            for tk in s.tickers():
                totals += np.array([sum(leg.payoff_at_expiry(p) for leg in s.legs if leg.ticker == tk) for p in prices])
    return totals

def breakevens_from_curve(prices: np.ndarray, pl: np.ndarray) -> List[float]:
    sign = np.sign(pl); z = np.where(np.diff(sign) != 0)[0]; bes = []
    for i in z:
        x0, x1 = prices[i], prices[i+1]; y0, y1 = pl[i], pl[i+1]
        if (y1 - y0) != 0:
            x = x0 - y0 * (x1 - x0) / (y1 - y0); bes.append(round(float(x), 4))
    return bes


# ============================
# Builders
# ============================

def build_single(ticker: str, expiry: dt.date, kind: str, strike: float, qty: int, price_per_contract: float, tag="what-if") -> Strategy:
    leg = OptionLeg(ticker=ticker, expiry=expiry, kind=kind, strike=strike, qty=qty, avg_cost=abs(price_per_contract))
    return Strategy(name=f"{ticker} {expiry} {('Long' if qty>0 else 'Short')}{kind} {strike:g}", legs=[leg], enabled=True, user_tag=tag)

def build_vertical(ticker: str, expiry: dt.date, kind: str, short_strike: float, long_strike: float, qty: int, net_credit: float, tag="what-if") -> Strategy:
    short_leg = OptionLeg(ticker, expiry, kind, short_strike, qty=-abs(qty), avg_cost=abs(net_credit))  # credit on short
    long_leg  = OptionLeg(ticker, expiry, kind, long_strike,  qty=+abs(qty), avg_cost=0.0)
    return Strategy(name=f"{ticker} {expiry} {kind} Vertical {long_strike:g}/{short_strike:g}", legs=[short_leg, long_leg], enabled=True, user_tag=tag)

def build_iron_condor(ticker: str, expiry: dt.date, short_put: float, long_put: float, short_call: float, long_call: float, qty: int, net_credit: float, tag="what-if") -> Strategy:
    legs = [
        OptionLeg(ticker, expiry, "P", short_put, qty=-abs(qty), avg_cost=abs(net_credit)/2.0),
        OptionLeg(ticker, expiry, "P", long_put,  qty=+abs(qty), avg_cost=0.0),
        OptionLeg(ticker, expiry, "C", short_call, qty=-abs(qty), avg_cost=abs(net_credit)/2.0),
        OptionLeg(ticker, expiry, "C", long_call,  qty=+abs(qty), avg_cost=0.0),
    ]
    return Strategy(name=f"{ticker} {expiry} Iron Condor P:{long_put:g}/{short_put:g} C:{short_call:g}/{long_call:g}", legs=legs, enabled=True, user_tag=tag)

def build_strangle(ticker: str, expiry: dt.date, put_strike: float, call_strike: float, qty: int, net_credit: float, tag="what-if") -> Strategy:
    legs = [
        OptionLeg(ticker, expiry, "P", put_strike,  qty=-abs(qty) if net_credit>=0 else +abs(qty), avg_cost=abs(net_credit)/2.0),
        OptionLeg(ticker, expiry, "C", call_strike, qty=-abs(qty) if net_credit>=0 else +abs(qty), avg_cost=abs(net_credit)/2.0),
    ]
    direction = "Short" if net_credit>=0 else "Long"
    return Strategy(name=f"{ticker} {expiry} {direction} Strangle {put_strike:g}/{call_strike:g}", legs=legs, enabled=True, user_tag=tag)


# ============================
# Quotes
# ============================

class QuoteProvider:
    Yahoo = "Yahoo Finance (yfinance)"
    Tradier = "Tradier"
    Polygon = "Polygon.io"
    Moomoo = "Moomoo (custom stub)"

def _to_yahoo_expiry(d: dt.date) -> str:
    return d.strftime("%Y-%m-%d")

if st:
    @st.cache_data(show_spinner=False, ttl=300)
    def _yf_chain(ticker: str, expiry: str):
        try:
            tkr = yf.Ticker(ticker); chain = tkr.option_chain(expiry)
            return chain.calls, chain.puts
        except Exception:
            return pd.DataFrame(), pd.DataFrame()

    @st.cache_data(show_spinner=False, ttl=120)
    def fetch_spot(provider: str, ticker: str, api_key: str = "", moomoo_cookie: str = "") -> Optional[float]:
        try:
            if provider == QuoteProvider.Yahoo and yf:
                fi = yf.Ticker(ticker).fast_info
                for k in ("last_price","lastPrice","last","regularMarketPrice"):
                    if k in fi and fi[k] is not None:
                        return float(fi[k])
                hist = yf.Ticker(ticker).history(period="1d")
                if not hist.empty:
                    return float(hist["Close"].iloc[-1])
                return None
            if provider == QuoteProvider.Tradier and requests:
                url = "https://api.tradier.com/v1/markets/quotes"
                headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
                r = requests.get(url, params={"symbols": ticker}, headers=headers, timeout=8)
                data = r.json()["quotes"]["quote"]
                return float(data["last"])
            if provider == QuoteProvider.Polygon and requests:
                base = "https://api.polygon.io"
                r = requests.get(f"{base}/v2/last/trade/{ticker}", params={"apiKey": api_key}, timeout=8)
                data = r.json()
                return float(data["results"]["p"])
            if provider == QuoteProvider.Moomoo:
                return None  # stub
        except Exception:
            return None
        return None

    @st.cache_data(show_spinner=False, ttl=300)
    def fetch_option_mid(provider: str, ticker: str, expiry: dt.date, kind: str, strike: float, api_key: str = "", moomoo_cookie: str = "") -> Optional[float]:
        try:
            if provider == QuoteProvider.Yahoo and yf:
                exp = _to_yahoo_expiry(expiry)
                calls, puts = _yf_chain(ticker, exp)
                chain = calls if kind == "C" else puts
                if chain.empty: return None
                idx = (chain["strike"] - strike).abs().sort_values().index[:1]
                row = chain.loc[idx].iloc[0]
                bid = None if pd.isna(row.get("bid", None)) else float(row["bid"])
                ask = None if pd.isna(row.get("ask", None)) else float(row["ask"])
                last = None if pd.isna(row.get("lastPrice", None)) else float(row["lastPrice"])
                if bid is not None and ask is not None and ask > 0:
                    return (bid + ask) / 2.0
                if last is not None and last > 0:
                    return last
                return None
            if provider == QuoteProvider.Tradier and requests:
                url = "https://api.tradier.com/v1/markets/options/chains"
                headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
                r = requests.get(url, params={"symbol": ticker, "expiration": expiry.strftime("%Y-%m-%d"), "greeks": "false"}, headers=headers, timeout=10)
                items = r.json().get("options", {}).get("option", [])
                if not items: return None
                filtered = [x for x in items if x.get("option_type","").upper().startswith("C" if kind=="C" else "P")]
                nearest = min(filtered, key=lambda x: abs(float(x["strike"]) - strike), default=None)
                if not nearest: return None
                bid = float(nearest.get("bid", 0) or 0); ask = float(nearest.get("ask", 0) or 0); last = float(nearest.get("last", 0) or 0)
                if bid > 0 and ask > 0: return (bid + ask) / 2.0
                if last > 0: return last
                return None
            if provider == QuoteProvider.Polygon:  # placeholder
                return None
            if provider == QuoteProvider.Moomoo:
                return None  # stub
        except Exception:
            return None
        return None


# ============================
# Streamlit App
# ============================

def st_app():
    st.set_page_config(page_title="Options P/L-at-Expiry Simulator (with Live Quotes)", layout="wide")
    st.title("📈 Options P/L-at-Expiry Simulator")
    st.caption("Upload Moomoo CSV • Build what‑ifs • Live quote auto‑pricing")

    st.sidebar.header("1) Upload Moomoo CSV")
    uploaded = st.sidebar.file_uploader("Positions CSV", type=["csv"], help="Upload your latest positions export from Moomoo.")
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

    # Provider
    st.sidebar.header("2) Quotes Provider")
    provider = st.sidebar.selectbox("Provider", [QuoteProvider.Yahoo, QuoteProvider.Tradier, QuoteProvider.Polygon, QuoteProvider.Moomoo], index=0)
    tradier_token = st.sidebar.text_input("Tradier Token", type="password") if provider == QuoteProvider.Tradier else ""
    polygon_key = st.sidebar.text_input("Polygon API Key", type="password") if provider == QuoteProvider.Polygon else ""
    moomoo_cookie = st.sidebar.text_input("Moomoo Cookie (custom)", type="password") if provider == QuoteProvider.Moomoo else ""

    # Spot inputs — initialize first; then allow API button to set & rerun
    st.sidebar.header("3) Underlying Spot Prices")
    spot_inputs: Dict[str, float] = {}
    for tk in tickers:
        if f"spot_{tk}" not in st.session_state:
            strikes = [l.strike for l in legs if l.ticker == tk]
            guess = float(np.median(strikes)) if strikes else 500.0
            st.session_state[f"spot_{tk}"] = guess
        spot_inputs[tk] = st.sidebar.number_input(f"{tk} spot", min_value=0.0, value=float(st.session_state[f"spot_{tk}"]), step=1.0, key=f"spot_{tk}")
        if st.sidebar.button(f"Fetch {tk} spot from API", key=f"btn_spot_{tk}"):
            fetched = fetch_spot(provider, tk, api_key=(tradier_token or polygon_key), moomoo_cookie=moomoo_cookie)
            if fetched:
                st.session_state[f"spot_{tk}"] = float(fetched)
                st.experimental_rerun()
            else:
                st.warning(f"{tk}: could not fetch spot from {provider}")

    st.sidebar.header("4) Price Grid")
    pct_width = st.sidebar.slider("Grid width (±%)", min_value=5, max_value=80, value=50, step=5) / 100.0
    steps = st.sidebar.slider("Grid steps", min_value=101, max_value=801, value=401, step=50)

    # Current strategies
    st.header("Current Strategies (from CSV)")
    for s in strategies:
        s.enabled = st.checkbox(s.name, value=True, key=f"cur_{s.name}")
    with st.expander("Show strategy details"):
        for s in strategies:
            st.text(s.describe())

    # Builder
    st.header("Add What-If Strategies")
    with st.expander("Builder"):
        builder_tab = st.tabs(["Single", "Vertical", "Iron Condor", "Strangle/Straddle"])
        what_if: List[Strategy] = st.session_state.get("what_if", [])

        # Single
        with builder_tab[0]:
            tk = st.selectbox("Ticker", options=tickers, key="single_tk")
            exp = st.date_input("Expiry", value=dt.date.today() + dt.timedelta(days=7), key="single_exp")
            kind = st.selectbox("Type", options=["C", "P"], key="single_kind")
            strike = st.number_input("Strike", min_value=0.0, value=float(round(spot_inputs.get(tk, 500.0))), step=1.0, key="single_strike")
            qty = st.number_input("Contracts (+long / -short)", value=-1, step=1, key="single_qty")
            price_key = "single_price"
            _ = st.number_input("Premium per contract", min_value=0.0, value=1.0, step=0.05, key=price_key)
            if st.button("Auto price from API", key="single_auto"):
                prem = fetch_option_mid(provider, tk, exp, kind, strike, api_key=(tradier_token or polygon_key), moomoo_cookie=moomoo_cookie)
                if prem is not None:
                    st.session_state[price_key] = float(prem); st.experimental_rerun()
                else:
                    st.warning("No quote found for that maturity/strike.")
            if st.button("➕ Add Single"):
                what_if.append(build_single(tk, exp, kind, float(strike), int(qty), float(st.session_state[price_key])))
                st.success("Added Single")

        # Vertical
        with builder_tab[1]:
            tk = st.selectbox("Ticker", options=tickers, key="vert_tk")
            exp = st.date_input("Expiry", value=dt.date.today() + dt.timedelta(days=7), key="vert_exp")
            kind = st.selectbox("Type", options=["C", "P"], key="vert_kind")
            s1 = st.number_input("Short strike", min_value=0.0, value=float(round(spot_inputs.get(tk, 500.0))), step=1.0, key="vert_short")
            s2 = st.number_input("Long strike",  min_value=0.0, value=float(max(0.0, round(spot_inputs.get(tk, 500.0)-5))), step=1.0, key="vert_long")
            qty = st.number_input("Contracts (abs)", value=1, step=1, key="vert_qty")
            credit_key = "vert_credit"
            _ = st.number_input("Net credit (per contract)", min_value=0.0, value=1.0, step=0.05, key=credit_key)
            if st.button("Auto price from API", key="vert_auto"):
                short_mid = fetch_option_mid(provider, tk, exp, kind, float(s1), api_key=(tradier_token or polygon_key), moomoo_cookie=moomoo_cookie)
                long_mid  = fetch_option_mid(provider, tk, exp, kind, float(s2), api_key=(tradier_token or polygon_key), moomoo_cookie=moomoo_cookie)
                if short_mid is not None and long_mid is not None:
                    st.session_state[credit_key] = max(0.0, float(short_mid - long_mid)); st.experimental_rerun()
                else:
                    st.warning("Could not fetch both legs.")
            if st.button("➕ Add Vertical"):
                # auto-swap for proper ordering on credit spreads
                ss, ls = float(s1), float(s2)
                if kind == "P" and ls > ss: ls, ss = ss, ls
                if kind == "C" and ss > ls: ls, ss = ss, ls
                what_if.append(build_vertical(tk, exp, kind, short_strike=ss, long_strike=ls, qty=int(qty), net_credit=float(st.session_state[credit_key])))
                st.success("Added Vertical")

        # Iron Condor
        with builder_tab[2]:
            tk = st.selectbox("Ticker", options=tickers, key="ic_tk")
            exp = st.date_input("Expiry", value=dt.date.today() + dt.timedelta(days=7), key="ic_exp")
            sp = st.number_input("Short put", min_value=0.0, value=float(max(1.0, round(spot_inputs.get(tk, 500.0)-10))), step=1.0, key="ic_sp")
            lp = st.number_input("Long put",  min_value=0.0, value=float(max(0.0, round(sp-5))), step=1.0, key="ic_lp")
            sc = st.number_input("Short call", min_value=0.0, value=float(round(spot_inputs.get(tk, 500.0)+10)), step=1.0, key="ic_sc")
            lc = st.number_input("Long call",  min_value=0.0, value=float(round(sc+5)), step=1.0, key="ic_lc")
            qty = st.number_input("Contracts (abs)", value=1, step=1, key="ic_qty")
            credit_key = "ic_credit"
            _ = st.number_input("Net credit (per contract)", min_value=0.0, value=1.0, step=0.05, key=credit_key)
            if st.button("Auto price from API", key="ic_auto"):
                pm = fetch_option_mid(provider, tk, exp, "P", float(sp), api_key=(tradier_token or polygon_key), moomoo_cookie=moomoo_cookie)
                plm = fetch_option_mid(provider, tk, exp, "P", float(lp), api_key=(tradier_token or polygon_key), moomoo_cookie=moomoo_cookie)
                cm = fetch_option_mid(provider, tk, exp, "C", float(sc), api_key=(tradier_token or polygon_key), moomoo_cookie=moomoo_cookie)
                clm = fetch_option_mid(provider, tk, exp, "C", float(lc), api_key=(tradier_token or polygon_key), moomoo_cookie=moomoo_cookie)
                if all(x is not None for x in [pm, plm, cm, clm]):
                    st.session_state[credit_key] = max(0.0, float((pm - plm) + (cm - clm))); st.experimental_rerun()
                else:
                    st.warning("Missing one or more legs from API.")
            if st.button("➕ Add Iron Condor"):
                what_if.append(build_iron_condor(tk, exp, float(sp), float(lp), float(sc), float(lc), int(qty), float(st.session_state[credit_key])))
                st.success("Added Iron Condor")

        # Strangle
        with builder_tab[3]:
            tk = st.selectbox("Ticker", options=tickers, key="sg_tk")
            exp = st.date_input("Expiry", value=dt.date.today() + dt.timedelta(days=7), key="sg_exp")
            put_k = st.number_input("Put strike", min_value=0.0, value=float(max(0.0, round(spot_inputs.get(tk, 500.0)-10))), step=1.0, key="sg_pk")
            call_k = st.number_input("Call strike", min_value=0.0, value=float(round(spot_inputs.get(tk, 500.0)+10)), step=1.0, key="sg_ck")
            qty = st.number_input("Contracts (abs)", value=1, step=1, key="sg_qty")
            credit_key = "sg_credit"
            _ = st.number_input("Net credit (>0 short, 0 for long)", min_value=0.0, value=1.0, step=0.05, key=credit_key)
            if st.button("Auto price from API", key="sg_auto"):
                pm = fetch_option_mid(provider, tk, exp, "P", float(put_k),  api_key=(tradier_token or polygon_key), moomoo_cookie=moomoo_cookie)
                cm = fetch_option_mid(provider, tk, exp, "C", float(call_k), api_key=(tradier_token or polygon_key), moomoo_cookie=moomoo_cookie)
                if pm is not None and cm is not None:
                    st.session_state[credit_key] = max(0.0, float(pm + cm)); st.experimental_rerun()
                else:
                    st.warning("Could not price both legs.")
            if st.button("➕ Add Strangle/Straddle"):
                what_if.append(build_strangle(tk, exp, float(put_k), float(call_k), int(qty), float(st.session_state[credit_key])))
                st.success("Added Strangle/Straddle")

        st.session_state["what_if"] = what_if

    # What-if list
    st.subheader("What-If Basket")
    what_if = st.session_state.get("what_if", [])
    if what_if:
        for i, s in enumerate(what_if):
            s.enabled = st.checkbox(s.name, value=True, key=f"wi_{i}")
            col1, col2 = st.columns([2,1])
            with col1: st.text(s.describe())
            with col2:
                if st.button("❌ Remove", key=f"rm_{i}"):
                    what_if.pop(i); st.experimental_rerun()

    st.checkbox("Show diagnostics", value=False, key="diag")

    # Per-ticker plots
    st.header("P/L at Expiry (per Ticker)")
    for tk in tickers:
        st.subheader(tk)
        prices = price_grid(spot_inputs[tk], pct_width=pct_width, steps=steps)
        cur = sum(strategy_pl_curve(s, prices, tk) for s in strategies if s.enabled and tk in s.tickers())
        add = sum(strategy_pl_curve(s, prices, tk) for s in what_if if s.enabled and tk in s.tickers())
        comb = cur + add

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=prices, y=cur, name="Current"))
        fig.add_trace(go.Scatter(x=prices, y=comb, name="Current + What-If"))
        fig.add_hline(y=0, line_dash="dash")
        fig.add_vline(x=spot_inputs[tk], line_dash="dot")
        fig.update_layout(height=400, xaxis_title="Underlying at Expiry", yaxis_title="P/L (USD)")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Current**")
        st.write({"P/L @ spot": round(float(np.interp(spot_inputs[tk], prices, cur)),2), "Max": round(float(np.max(cur)),2), "Min": round(float(np.min(cur)),2), "Breakevens": breakevens_from_curve(prices, cur)})
        st.markdown("**Current + What-If**")
        st.write({"P/L @ spot": round(float(np.interp(spot_inputs[tk], prices, comb)),2), "Max": round(float(np.max(comb)),2), "Min": round(float(np.min(comb)),2), "Breakevens": breakevens_from_curve(prices, comb)})

        if st.session_state.get("diag"):
            st.write("Legs:")
            for s in strategies:
                for lg in s.legs:
                    if lg.ticker==tk:
                        st.write(lg.label())

    # Combined approximate view
    st.header("Portfolio (Combined) – Approximate")
    tk0 = tickers[0]
    prices = price_grid(spot_inputs[tk0], pct_width=pct_width, steps=steps)
    current = aggregate_pl_curves(strategies, prices, ticker=tk0)
    added = aggregate_pl_curves(what_if, prices, ticker=tk0)
    combined = current + added
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=prices, y=current, name="Current"))
    fig2.add_trace(go.Scatter(x=prices, y=combined, name="Current + What-If"))
    fig2.add_hline(y=0, line_dash="dash")
    fig2.add_vline(x=spot_inputs[tk0], line_dash="dot")
    fig2.update_layout(height=400, xaxis_title=f"{tk0} Price at Expiry", yaxis_title="P/L (USD)")
    st.plotly_chart(fig2, use_container_width=True)

    st.info("Spot & premiums fetched via selected provider. Moomoo direct chain fetch is not public; share internal access if available.")

    st.download_button("Download sample CSV", data=sample_positions_df().to_csv(index=False), file_name="sample_moomoo_positions.csv", mime="text/csv")


# ============================
# Sample Data
# ============================

def sample_positions_df() -> pd.DataFrame:
    data = [
        {"Symbol": "NVDA251031P195/200", "Name": "NVDA Vertical", "Quantity": -29, "Current price": 0.655, "Average Cost": 0.995},
        {"Symbol": "NVDA251031P195000", "Name": "NVDA 251031 195.00P", "Quantity": 29, "Current price": 0.34, "Average Cost": 0.54},
        {"Symbol": "NVDA251031P200000", "Name": "NVDA 251031 200.00P", "Quantity": -29, "Current price": 1.0, "Average Cost": 1.535},
    ]
    return pd.DataFrame(data)


# ============================
# Tests
# ============================

def _test_payoff_basic():
    leg = OptionLeg("ABC", dt.date(2025,1,1), "C", 100, qty=+1, avg_cost=2.0)
    assert math.isclose(leg.payoff_at_expiry(120), (20-2)*100)
    assert math.isclose(leg.payoff_at_expiry(90), (0-2)*100)
    leg = OptionLeg("ABC", dt.date(2025,1,1), "P", 50, qty=-2, avg_cost=1.5)
    assert math.isclose(leg.payoff_at_expiry(60), +1.5*2*100)
    assert math.isclose(leg.payoff_at_expiry(40), (+1.5-10)*2*100)

def _test_parser_and_grouping():
    df = sample_positions_df()
    strategies, legs = parse_moomoo_positions(df)
    strikes = sorted({l.strike for l in legs})
    assert strikes == [195.0, 200.0]
    qtys = { (l.kind, l.strike): l.qty for l in legs }
    assert qtys[("P", 195.0)] == 29 and qtys[("P", 200.0)] == -29

def _run_tests():
    _test_payoff_basic()
    _test_parser_and_grouping()
    print("All tests passed.")


# ============================
# Entrypoint
# ============================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tests", action="store_true", help="Run unit tests and exit.")
    args, _ = parser.parse_known_args()
    if args.run_tests:
        _run_tests(); return
    if st is None:
        print("Streamlit not installed. Install with `pip install streamlit plotly pandas numpy yfinance requests` and run:\n  streamlit run option_pl_simulator.py")
        return
    st_app()

if __name__ == "__main__":
    main()
