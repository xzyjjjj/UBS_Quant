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
- `gross_edge = |mispricing|`
- `net_edge = gross_edge - est_cost`
- `edge` equals `net_edge` for backward compatibility
- `iv_mkt`: implied vol solved from market mid
- `iv_fair`: implied vol solved from model fair price
- `vol_edge = iv_mkt - iv_fair`
- `vol_action`: `SELL_VOL` (vol_edge>0) / `BUY_VOL` (vol_edge<0)
- `price_signal_ok`: price-edge threshold pass
- `vol_signal_ok`: vol-edge threshold pass
- `signal_confirmed`: both views pass together (`CONFIRMED_BOTH`)

## Signal and data-quality logic
- Signal trigger:
  - Price view uses `net_edge >= min_edge`
  - Vol view uses `|vol_edge| >= min_abs_vol_edge`
  - `Both` mode requires confirmation from both views
- Cost model:
  - `est_cost = fee_per_contract + slippage_bps*mid + spread_cross_ratio*spread`
  - cost is deducted before signal trigger (net edge)
- Hybrid timestamp checks:
  - align check between Futu option quote timestamp and Alpha spot timestamp
  - freshness SLA checks for both sources
  - warnings shown in dashboard caption when threshold is breached or timestamps are missing
