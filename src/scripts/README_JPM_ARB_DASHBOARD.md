# JPM Arbitrage Dashboard

## Run
```bash
cd /Users/xuzhenyu/Desktop/UBS_Quant
streamlit run src/scripts/jpm_arb_dashboard.py
```

## Required
- Model file available (default): `models/mlp_direct.json`
- Python packages:
```bash
pip install streamlit pandas requests
```

## Optional data sources
- Futu OpenD (real-time preferred):
  - install `futu-api`
  - set `FUTU_HOST`, `FUTU_PORT` if not default
- AlphaVantage fallback:
  - set `ALPHAVANTAGE_API_KEY` (in `.env` or sidebar)

## In dashboard
- Select source:
  - `Hybrid (Futu options + Alpha spot)` (recommended when Futu has option-only permission)
  - `Auto / Futu / AlphaVantage`
- Select arbitrage mode: `Price edge / Vol edge / Both`
- Click `Refresh`
- If spot is unavailable from APIs, use `Manual spot override`
- Use filters: `Min edge`, `Min |vol_edge|`, `Min open interest`, `Min volume`
- Download opportunity table as CSV

## Key output columns
- `mispricing = market_mid - model_fair`
- `edge = |mispricing| - est_cost`
- `iv_mkt`: implied vol solved from market mid
- `iv_fair`: implied vol solved from model fair price
- `vol_edge = iv_mkt - iv_fair`
- `vol_action`: `SELL_VOL` (vol_edge>0) / `BUY_VOL` (vol_edge<0)
