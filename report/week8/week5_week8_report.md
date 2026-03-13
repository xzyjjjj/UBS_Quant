# Week5-Week8 Integrated Report

## 1. Scope
This report summarizes the delivered work across Week5 to Week8:
- Week5: model implementation and training
- Week6: explainability + strategy diagnostics
- Week7: interactive arbitrage/backtest tooling
- Week8: integrated workflow and deployable scripts

## 2. Week5: Model Training (Time Split 70/15/15)
Data entry: `jpm_options_final.csv`.
Split: strict date-based 70/15/15 to avoid look-ahead bias.

### Model comparison (server run)
| Model | Train MAE | Train RMSE | Val MAE | Val RMSE | Test MAE | Test RMSE |
|---|---:|---:|---:|---:|---:|---:|
| MLP | 0.842995 | 1.396970 | 1.477292 | 2.353230 | 6.101606 | 9.964472 |
| BSM+MLP residual | 1.319153 | 2.138775 | 1.313159 | 2.519798 | 2.248075 | 4.030562 |
| Transformer | 15.943744 | 21.209773 | 18.029755 | 23.046355 | 28.232014 | 38.077330 |

Conclusion: `BSM+MLP residual` is the best test performer.

## 3. Week6: Explainability + Requirement Mapping
Implemented scripts:
- `src/scripts/run_week6_mlp_explainability.py`
- `src/scripts/run_week6_mlp_requirements.py`

Delivered items:
- Hyper-parameter search
- MAE / RMSE / R2
- Global explainability (permutation)
- Local explainability proxy (occlusion-style)

### Key explainability result (global)
Most influential features (delta MAE descending):
1. `log_moneyness`
2. `is_call`
3. `moneyness`
4. `sqrt_ttm`
5. `q`

Interpretation: model decisions are mainly driven by moneyness and option side, consistent with option pricing structure.

## 4. Week7: Panel + Strategy Workflow
Implemented live tools:
- `src/scripts/jpm_arb_dashboard.py`
- `src/scripts/backtest_panel.py`

Main capabilities:
- Hybrid data source mode: `Futu options + AlphaVantage spot`
- Price-edge and Vol-edge views
- Candidate filtering + downloadable action table
- Backtest panel with:
  - full action log
  - equity curve
  - metrics (PnL, win rate, max drawdown, Sharpe, PF, avg win/loss, win-loss ratio)

## 5. Week8: Strategy Integration and Evaluation
Backtest engine upgraded:
- `src/scripts/backtest_10d_7d_near_spot.py`
- supports `mlp_direct` and `bsm_mlp_residual`
- supports model-edge ranking and liquidity filters

### Tuned strategy (balanced, 1Y)
Config highlights:
- target DTE = 14, tol = 10
- max strike gap = 15
- max moneyness deviation = 0.30
- min volume/open interest = 1/1
- model: `bsm_mlp_residual`, rank by model edge
- no forced daily trade

Result:
- Trades: 41
- Win rate: 58.54%
- Total PnL: 7,452.5
- Profit factor: 4.24

## 6. Deliverables
- Model training scripts and inference scripts
- Explainability and requirement-run scripts
- Live arbitrage panel + backtest panel
- Tunable backtest engine with model-based ranking
- This integrated Week5-Week8 report
