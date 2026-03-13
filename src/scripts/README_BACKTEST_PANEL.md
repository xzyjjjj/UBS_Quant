# Backtest Panel

## Run
```bash
cd /Users/xuzhenyu/Desktop/UBS_Quant
streamlit run src/scripts/backtest_panel.py
```

## What it shows
- All actions (every trade day, including skipped/relaxed/executed)
- Equity curve (cumulative PnL)
- Common backtest metrics:
  - Total PnL
  - Avg PnL per trade
  - Win rate
  - Max drawdown
  - Daily Sharpe
  - Profit factor

## Engine
The panel calls:
- `src/scripts/backtest_10d_7d_near_spot.py`

Core rule defaults:
- near-spot: `|strike - spot| <= 5`
- target DTE: `7 +/- 1`
- one trade per day
- force daily trade enabled by default
