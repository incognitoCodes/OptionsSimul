# file: option_pl_simulator.py
from __future__ import annotations
import argparse, datetime as dt, math, re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np, pandas as pd

# Optional UI/quotes deps
try:
    import streamlit as st
    import plotly.graph_objects as go
    import yfinance as yf
    import requests
except Exception:
    st = None; go = None; yf = None; requests = None

MULTIPLIER = 100  # contract size

# ===================== Models =====================
@dataclass
class OptionLeg:
    ticker: str
    expiry: dt.date
    kind: str        # "C" or "P"
    strike: float
    qty: int         # +long / -short
    avg_cost: float  # premium per contract
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

        # Skip spread header (legs appear as separate rows)
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

# ===================== Quotes =====================
class QuoteProvider:
    Yahoo   = "Yahoo Finance (yfinance)"
    Tradier = "Tradier"
    Polygon = "Polygon.io"
    Moomoo  = "Moomoo (custom stub)"

def normalize_symbol_for_yahoo(t: str) -> str:
    return t.strip().upper().replace(" ", "").replace(".", "-")

def _safe_float(v):
    try:
        import numpy as _np
        if v is None: return None
        if isinstance(v, float) and _np.isnan(v): return None
        return float(v)
    except Exception:
        return None

if st:
    @st.cache_data(ttl=300, show_spinner=False)
    def yahoo_available_expiries(ticker: str) -> List[str]:
        try:
            return list((yf.Ticker(normalize_symbol_for_yahoo(ticker)).options) or [])
        except Exception:
            return []

    def _nearest_expiry_str(ticker: str, want: dt.date) -> Optional[str]:
        exps = yahoo_available_expiries(ticker)
        if not exps:
            return None
        def to_date(s): y,m,d = map(int, s.split("-")); return dt.date(y,m,d)
        return min(exps, key=lambda s: abs((to_date(s) - want).days))

    @st.cache_data(ttl=300, show_spinner=False)
    def get_available_strikes(provider: str, ticker: str, expiry: dt.date,
                              api_key: str = "", moomoo_cookie: str = "") -> List[float]:
        strikes: List[float] = []
        try:
            if provider == QuoteProvider.Yahoo and yf:
                tk = normalize_symbol_for_yahoo(ticker)
                t  = yf.Ticker(tk)
                exps = t.options or []
                if not exps:
                    return strikes
                def to_date(s): y,m,d = map(int, s.split("-")); return dt.date(y,m,d)
                exp_str = min(exps, key=lambda s: abs((to_date(s) - expiry).days))
                chain = t.option_chain(exp_str)
                if chain and getattr(chain, "calls", None) is not None and not chain.calls.empty:
                    strikes.extend(chain.calls["strike"].astype(float).tolist())
                if chain and getattr(chain, "puts", None) is not None and not chain.puts.empty:
                    strikes.extend(chain.puts["strike"].astype(float).tolist())
                return sorted(set(strikes))
        except Exception:
            return strikes
        return strikes

    @st.cache_data(ttl=90, show_spinner=False)
    def fetch_spot(provider: str, ticker: str, api_key: str = "", moomoo_cookie: str = "") -> Tuple[Optional[float], str]:
        """
        Returns (price, source) or (None, reason).
        Robust fallbacks for Yahoo; simple for others.
        """
        try:
            if provider == QuoteProvider.Yahoo and yf:
                tk = normalize_symbol_for_yahoo(ticker)
                t  = yf.Ticker(tk)

                # 1) fast_info
                try:
                    fi = t.fast_info
                    for k in ("last_price", "regularMarketPrice", "last", "lastPrice"):
                        v = _safe_float(fi.get(k) if hasattr(fi, "get") else getattr(fi, k, None))
                        if v and v > 0:
                            return v, f"fast_info.{k}"
                except Exception:
                    pass

                # 2) info
                try:
                    info = t.info or {}
                    for k in ("regularMarketPrice", "currentPrice", "previousClose"):
                        v = _safe_float(info.get(k))
                        if v and v > 0:
                            return v, f"info.{k}"
                except Exception:
                    pass

                # 3) 1d history Close
                try:
                    h = t.history(period="1d")
                    if not h.empty and _safe_float(h["Close"].iloc[-1]):
                        return float(h["Close"].iloc[-1]), "history(1d).Close"
                except Exception:
                    pass

                # 4) 5d history last valid Close
                try:
                    h5 = t.history(period="5d")
                    if not h5.empty and _safe_float(h5["Close"].dropna().iloc[-1]):
                        return float(h5["Close"].dropna().iloc[-1]), "history(5d).Close(last valid)"
                except Exception:
                    pass

                # 5) download 5d/1d
                try:
                    d5 = yf.download(tk, period="5d", interval="1d", progress=False, threads=False)
                    if not d5.empty and _safe_float(d5["Close"].dropna().iloc[-1]):
                        return float(d5["Close"].dropna().iloc[-1]), "download(5d,1d)"
                except Exception:
                    pass

                # 6) download 1d/1m (last tick)
                try:
                    d1m = yf.download(tk, period="1d", interval="1m", progress=False, threads=False)
                    if not d1m.empty and _safe_float(d1m["Close"].dropna().iloc[-1]):
                        return float(d1m["Close"].dropna().iloc[-1]), "download(1d,1m)"
                except Exception:
                    pass

                return None, "no_yahoo_price"

            if provider == QuoteProvider.Tradier and requests:
                r = requests.get(
                    "https://api.tradier.com/v1/markets/quotes",
                    params={"symbols": ticker},
                    headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
                    timeout=8
                ); r.raise_for_status()
                return float(r.json()["quotes"]["quote"]["last"]), "tradier.last"

            if provider == QuoteProvider.Polygon and requests:
                r = requests.get(f"https://api.polygon.io/v2/last/trade/{ticker}",
                                 params={"apiKey": api_key}, timeout=8); r.raise_for_status()
                return float(r.json()["results"]["p"]), "polygon.last_trade"

            return None, "unsupported_provider"
        except Exception as e:
            return None, f"error:{e}"

    @st.cache_data(ttl=300, show_spinner=False)
    def fetch_option_mid(provider: str, ticker: str, expiry: dt.date, kind: str, strike: float,
                         api_key: str = "", moomoo_cookie: str = "") -> Optional[float]:
        try:
            if provider == QuoteProvider.Yahoo and yf:
                tk = normalize_symbol_for_yahoo(ticker)
                t  = yf.Ticker(tk)
                exp = _nearest_expiry_str(ticker, expiry)
                if not exp:
                    return None
                try:
                    chain = t.option_chain(exp)
                except Exception:
                    return None
                tbl = chain.calls if kind == "C" else chain.puts
                if tbl is None or tbl.empty:
                    return None
                row = tbl.iloc[(tbl["strike"] - strike).abs().argsort()[:1]]
                bid = _safe_float(row["bid"].iloc[0]); ask = _safe_float(row["ask"].iloc[0])
                last = _safe_float(row.get("lastPrice", pd.Series([np.nan])).iloc[0])
                if bid and ask and ask > 0: return (bid + ask) / 2.0
                if last and last > 0: return last
                return None

            if provider == QuoteProvider.Tradier and requests:
                r = requests.get(
                    "https://api.tradier.com/v1/markets/options/chains",
                    params={"symbol": ticker, "expiration": expiry.strftime("%Y-%m-%d"), "greeks": "false"},
                    headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
                    timeout=10
                ); r.raise_for_status()
                items = r.json().get("options", {}).get("option", [])
                if not items: return None
                filt = [x for x in items if x.get("option_type","").upper().startswith("C" if kind=="C" else "P")]
                nearest = min(filt, key=lambda x: abs(float(x["strike"]) - strike), default=None)
                if not nearest: return None
                bid = _safe_float(nearest.get("bid")); ask = _safe_float(nearest.get("ask")); last = _safe_float(nearest.get("last"))
                if bid and ask and ask > 0: return (bid + ask) / 2.0
                if last and last > 0: return last
                return None

            return None
        except Exception:
            return None

# ===================== Streamlit UI =====================
def st_app():
    st.set_page_config(page_title="Options P/L-at-Expiry", layout="wide")
    st.title("📈 Options P/L-at-Expiry Simulator")
    st.caption("Upload Moomoo CSV • Build what-ifs • Live quote auto-pricing")

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
        st.error(f"Failed to parse CSV: {e}")
        st.dataframe(df.head(50))
        st.stop()

    tickers = sorted({l.ticker for l in legs}) or ["NVDA"]

    # Provider
    st.sidebar.header("Quotes Provider")
    provider = st.sidebar.selectbox("Provider",
        [QuoteProvider.Yahoo, QuoteProvider.Tradier, QuoteProvider.Polygon, QuoteProvider.Moomoo], index=0)
    tradier_token = st.sidebar.text_input("Tradier Token", type="password") if provider==QuoteProvider.Tradier else ""
    polygon_key  = st.sidebar.text_input("Polygon API Key", type="password") if provider==QuoteProvider.Polygon else ""
    moomoo_cookie= st.sidebar.text_input("Moomoo Cookie (custom)", type="password") if provider==QuoteProvider.Moomoo else ""

    # Spots
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
            price, source = fetch_spot(provider, tk, api_key=(tradier_token or polygon_key), moomoo_cookie=moomoo_cookie)
            if price is not None:
                st.session_state[f"spot_{tk}"] = float(price)
                st.sidebar.success(f"{tk}: {price:.2f} • {source}")
                st.experimental_rerun()
            else:
                st.sidebar.info(f"{tk}: live price unavailable ({source}). Using current value {st.session_state[f'spot_{tk}']:.2f}.")

    # Grid
    st.sidebar.header("Price Grid")
    pct   = st.sidebar.slider("Grid width (±%)", 5, 80, 50, 5) / 100.0
    steps = st.sidebar.slider("Grid steps", 101, 801, 401, 50)

    # Current
    st.header("Current Strategies (from CSV)")
    for s in strategies:
        s.enabled = st.checkbox(s.name, True, key=f"cur_{s.name}")
    with st.expander("Show strategy details"):
        for s in strategies: st.text(s.describe())

    # Builders
    st.header("Add What-If Strategies")
    with st.expander("Builder"):
        tabs = st.tabs(["Single","Vertical","Iron Condor","Iron Butterfly","Strangle/Straddle"])
        what_if: List[Strategy] = st.session_state.get("what_if", [])

        # Single
        with tabs[0]:
            tk   = st.selectbox("Ticker", options=tickers, key="single_tk")
            exp  = st.date_input("Expiry", value=dt.date.today()+dt.timedelta(days=7), key="single_exp")
            kind = st.selectbox("Type", options=["C","P"], key="single_kind")
            strike = st.number_input("Strike", min_value=0.0, value=float(round(spot_inputs.get(tk,500.0))), step=1.0, key="single_strike")
            qty = st.number_input("Contracts (+long / -short)", value=-1, step=1, key="single_qty")
            price_key = "single_price"
            _ = st.number_input("Premium per contract", min_value=0.0, value=1.0, step=0.05, key=price_key)
            if st.button("Auto price from API", key="single_auto"):
                prem = fetch_option_mid(provider, tk, exp, kind, strike, api_key=(tradier_token or polygon_key), moomoo_cookie=moomoo_cookie)
                if prem is not None: st.session_state[price_key] = float(prem); st.experimental_rerun()
                else: st.warning("No quote found for that maturity/strike.")
            if st.button("➕ Add Single"):
                what_if.append(build_single(tk,exp,kind,float(strike),int(qty),float(st.session_state[price_key]))); st.success("Added Single")

        # Vertical
        with tabs[1]:
            tk   = st.selectbox("Ticker", options=tickers, key="vert_tk")
            exp  = st.date_input("Expiry", value=dt.date.today()+dt.timedelta(days=7), key="vert_exp")
            kind = st.selectbox("Type", options=["C","P"], key="vert_kind")
            s1 = st.number_input("Short strike", min_value=0.0, value=float(round(spot_inputs.get(tk,500.0))), step=1.0, key="vert_short")
            s2 = st.number_input("Long strike",  min_value=0.0, value=float(max(0.0, round(spot_inputs.get(tk,500.0)-5))), step=1.0, key="vert_long")
            qty = st.number_input("Contracts (abs)", value=1, step=1, key="vert_qty")
            credit_key = "vert_credit"; _ = st.number_input("Net credit (per contract)", min_value=0.0, value=1.0, step=0.05, key=credit_key)
            if st.button("Auto price from API", key="vert_auto"):
                short_mid = fetch_option_mid(provider, tk, exp, kind, float(s1), api_key=(tradier_token or polygon_key), moomoo_cookie=moomoo_cookie)
                long_mid  = fetch_option_mid(provider, tk, exp, kind, float(s2), api_key=(tradier_token or polygon_key), moomoo_cookie=moomoo_cookie)
                if short_mid is not None and long_mid is not None: st.session_state[credit_key] = max(0.0, float(short_mid - long_mid)); st.experimental_rerun()
                else: st.warning("Could not fetch both legs.")
            if st.button("➕ Add Vertical"):
                ss, ls = float(s1), float(s2)
                if kind == "P" and ls > ss: ls, ss = ss, ls
                if kind == "C" and ss > ls: ls, ss = ss, ls
                what_if.append(build_vertical(tk,exp,kind,ss,ls,int(qty),float(st.session_state[credit_key]))); st.success("Added Vertical")

        # Iron Condor
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
                pm  = fetch_option_mid(provider, tk, exp, "P", float(sp), api_key=(tradier_token or polygon_key), moomoo_cookie=moomoo_cookie)
                plm = fetch_option_mid(provider, tk, exp, "P", float(lp), api_key=(tradier_token or polygon_key), moomoo_cookie=moomoo_cookie)
                cm  = fetch_option_mid(provider, tk, exp, "C", float(sc), api_key=(tradier_token or polygon_key), moomoo_cookie=moomoo_cookie)
                clm = fetch_option_mid(provider, tk, exp, "C", float(lc), api_key=(tradier_token or polygon_key), moomoo_cookie=moomoo_cookie)
                if all(x is not None for x in [pm, plm, cm, clm]): st.session_state[credit_key] = max(0.0, float((pm - plm) + (cm - clm))); st.experimental_rerun()
                else: st.warning("Missing one or more legs from API.")
            if st.button("➕ Add Iron Condor"):
                what_if.append(build_iron_condor(tk,exp,float(sp),float(lp),float(sc),float(lc),int(qty),float(st.session_state[credit_key]))); st.success("Added Iron Condor")

        # Iron Butterfly (+ Snap to market)
        with tabs[3]:
            tk = st.selectbox("Ticker", options=tickers, key="ib_tk")
            exp = st.date_input("Expiry", value=dt.date.today()+dt.timedelta(days=7), key="ib_exp")
            center = st.number_input("Center strike (short straddle)", min_value=0.0, value=float(round(spot_inputs.get(tk,500.0))), step=1.0, key="ib_center")
            width  = st.number_input("Wing width", min_value=0.0, value=5.0, step=0.5, key="ib_width")

            step_choice = st.selectbox("Snap width to", options=[1.0, 2.5, 5.0, 10.0], index=2, key="ib_step")
            if st.button("🔧 Snap to market", key="ib_snap"):
                strikes = get_available_strikes(provider, tk, exp, api_key=(tradier_token or polygon_key), moomoo_cookie=moomoo_cookie)
                if strikes:
                    nearest_center = min(strikes, key=lambda s: abs(s - float(st.session_state["ib_center"])))
                    st.session_state["ib_center"] = float(nearest_center)
                    step = float(step_choice)
                    snapped_width = max(step, round(float(st.session_state["ib_width"]) / step) * step)
                    st.session_state["ib_width"] = float(snapped_width)
                    st.experimental_rerun()
                else:
                    st.warning("Could not load market strikes to snap.")

            put_wing  = float(max(0.0, st.session_state["ib_center"] - st.session_state["ib_width"]))
            call_wing = float(st.session_state["ib_center"] + st.session_state["ib_width"])
            st.write(f"Wings ⇢ Put {put_wing:g}, Call {call_wing:g}")

            qty = st.number_input("Contracts (abs)", value=1, step=1, key="ib_qty")
            credit_key = "ib_credit"; _ = st.number_input("Net credit (per contract)", min_value=0.0, value=1.0, step=0.05, key=credit_key)
            if st.button("Auto price from API", key="ib_auto"):
                pm  = fetch_option_mid(provider, tk, exp, "P", st.session_state["ib_center"], api_key=(tradier_token or polygon_key), moomoo_cookie=moomoo_cookie)
                cm  = fetch_option_mid(provider, tk, exp, "C", st.session_state["ib_center"], api_key=(tradier_token or polygon_key), moomoo_cookie=moomoo_cookie)
                plm = fetch_option_mid(provider, tk, exp, "P", put_wing, api_key=(tradier_token or polygon_key), moomoo_cookie=moomoo_cookie)
                clm = fetch_option_mid(provider, tk, exp, "C", call_wing, api_key=(tradier_token or polygon_key), moomoo_cookie=moomoo_cookie)
                if all(x is not None for x in [pm, cm, plm, clm]): st.session_state[credit_key] = max(0.0, float((pm + cm) - (plm + clm))); st.experimental_rerun()
                else: st.warning("Missing one or more legs from API.")
            if st.button("➕ Add Iron Butterfly"):
                what_if.append(build_iron_butterfly(tk,exp,float(st.session_state["ib_center"]),float(put_wing),float(call_wing),int(qty),float(st.session_state[credit_key]))); st.success("Added Iron Butterfly")

        # Strangle/Straddle
        with tabs[4]:
            tk = st.selectbox("Ticker", options=tickers, key="sg_tk")
            exp = st.date_input("Expiry", value=dt.date.today()+dt.timedelta(days=7), key="sg_exp")
            put_k  = st.number_input("Put strike",  min_value=0.0, value=float(max(0.0, round(spot_inputs.get(tk,500.0)-10))), step=1.0, key="sg_pk")
            call_k = st.number_input("Call strike", min_value=0.0, value=float(round(spot_inputs.get(tk,500.0)+10)), step=1.0, key="sg_ck")
            qty = st.number_input("Contracts (abs)", value=1, step=1, key="sg_qty")
            credit_key = "sg_credit"; _ = st.number_input("Net credit (>0 short, 0 for long)", min_value=0.0, value=1.0, step=0.05, key=credit_key)
            if st.button("Auto price from API", key="sg_auto"):
                pm = fetch_option_mid(provider, tk, exp, "P", float(put_k),  api_key=(tradier_token or polygon_key), moomoo_cookie=moomoo_cookie)
                cm = fetch_option_mid(provider, tk, exp, "C", float(call_k), api_key=(tradier_token or polygon_key), moomoo_cookie=moomoo_cookie)
                if pm is not None and cm is not None: st.session_state[credit_key] = max(0.0, float(pm + cm)); st.experimental_rerun()
                else: st.warning("Could not price both legs.")
            if st.button("➕ Add Strangle/Straddle"):
                what_if.append(build_strangle(tk,exp,float(put_k),float(call_k),int(qty),float(st.session_state[credit_key]))); st.success("Added Strangle/Straddle")

        st.session_state["what_if"] = what_if

    # Charts
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

    # Portfolio (approx per first ticker)
    st.header("Portfolio (Combined)")
    tk0 = tickers[0]
    xs = price_grid(st.session_state[f"spot_{tk0}"], pct, steps)
    def agg(strats: List[Strategy]) -> np.ndarray:
        return sum(strategy_pl_curve(s, xs, tk0) for s in strats if s.enabled and tk0 in s.tickers())
    current = agg(strategies)
    added   = agg(st.session_state.get("what_if", []))
    comb    = current + added
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
        # quick sanity
        leg = OptionLeg("ABC", dt.date(2025,1,1), "C", 100, +1, 2.0)
        assert math.isclose(leg.payoff_at_expiry(120), 1800.0)
        df = sample_positions_df()
        parse_moomoo_positions(df)
        print("Tests OK")
        return
    if st is None:
        print("Install: pip install streamlit plotly pandas numpy yfinance requests")
        print("Run: streamlit run option_pl_simulator.py"); return
    st_app()

if __name__ == "__main__":
    main()
