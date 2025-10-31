# Options P/L-at-Expiry Simulator (Moomoo-friendly)

Interactive Streamlit app to model option strategies on top of current Moomoo positions.  
- Upload Moomoo CSV, see per-ticker + combined P/L at expiry
- Build what-if strategies (Singles, Verticals, Iron Condors, Strangles)
- Auto-fill premiums/credits via Yahoo (yfinance) or optional Tradier/Polygon keys

## Quickstart (local)
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run option_pl_simulator.py

