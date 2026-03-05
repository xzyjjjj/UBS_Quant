# UBS Quant

## Week4 快速运行

### 1) 推荐（一键跑最终 pipeline）
```bash
python src/quant_research/run_week4_final_pipeline.py
```

### 2) 输入数据（默认路径）
- `output/market_panel/raw/options_history.csv`
- `output/market_panel/processed/panel.csv`
- `output/market_panel/raw/equity_daily.csv`

### 3) 三个脚本分别做什么
- `python src/quant_research/run_week4_final_pipeline.py`  
  一键串联 Week4 最终流程：先准备评估数据，再跑 baseline 评估并输出结果。
- `python src/quant_research/run_week4_prepare_eval_data.py`  
  把期权原始数据与 panel 合并，构造 `market_price/S0/sigma/r/q/T_years`，输出标准评估输入 CSV。
- `python src/quant_research/run_week4_baseline.py`  
  用 BSM 计算预测值并与真实交易价对比，输出 MAE/RMSE、分组失效场景和误差画像表。

## Contact
- `xuzhenjyu23@mails.ucas.ac.cn`
