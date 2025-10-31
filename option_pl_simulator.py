# file: option_pl_simulator.py
from __future__ import annotations
import argparse, datetime as dt, math, re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np, pandas as pd

# UI/quotes (optional at import time)
try:
    import streamlit as st
    import plotly.graph_objects as go
    import yfinance as yf
    import requests
except Exception:
    st = None; go = None; yf = None; requests = None

MULTIPLIER = 100  # contract size

# ---------------- Models ----------------
@dataclass
class OptionLeg:
    ticker: str
    expiry: dt.date
    kind: str         # 'C' or 'P'
    strike: float
    qty: int          # +long / -short
    avg_cost: float   # premium per contract
    name: str = ""
    source_id: Optional[str] = None

    def payoff_at_expiry(self, s: float) -> float:
        intrinsic = max(s - self.strike, 0.0) if self.kind == "C" else max(self.strike - s, 0.0)
        per_contract = (intrinsic - self.avg_cost) if self.qty > 0 else (self.avg_cost - intrinsic)
        return per_contract * abs(self.qty) * MULTIPLIER

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

    def expiries(self) -> List[dt.date]:
        return sorted({l.expiry for l in self.legs})

    def payoff_at_expiry(self, s: float, ticker: Optional[str] = None) -> float:
        return sum(l.payoff_at_expiry(s) for l in self.legs if self.enabled and (ticker is None or l.ticker == ticker))

    def describe(self) -> str:
        return "\n".join([f"{self.name} [{'ON' if self.enabled else 'OFF'}]"] + [f"  - {l.label()}" for l in self.legs])

# ---------------- CSV parsing ----------------
EXPECTED_COLUMNS = {"Symbol","Name","Quantity","Current price","Average Cost"}
LEG_SYMBOL_RE = re.compile(r"^(?P<ticker>[A-Z]+?)(?P<expiry>\d{6})(?P<kind>[CP])(?P<strike>\d{3,6})(?:\.\d+)?$")
SPREAD_SYMBOL_RE = re.compile(r"^(?P<ticker>[A-Z]+
