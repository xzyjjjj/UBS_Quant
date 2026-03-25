# Week 3 — Chooser Option (BSM) 现跑通验证

本周目标改为“先跑通并形成可复现流程”，**不要求与论文参数/结果一致**。

## 依赖与数据

- 数据源：Futu OpenD + `futu-api`
- 先运行（示例）：
  - `python /Users/xuzhenyu/Desktop/UBS_Quant/src/scripts/futu_test.py --codes HK.00700 --start 2025-02-26 --end 2026-02-26 --calc-sigma`
- 会产出：
  - `output/futu/snapshot.csv`（含 `last_price` 作为 `S0`）
  - `output/futu/kline_HK_00700.csv`（含 `close` 用于估计历史波动率）
  - `output/futu/metrics.json`（含 `hist_sigma.sigma_annualized`）

## 定价模型（BSM + Monte Carlo）

- Chooser 定义：在 `T1` 时刻选择**价值更高**的 European call/put（同一 `K`、同一到期 `T2`），到 `T2` 结算对应 payoff。
- 参数（可调）：
  - `S0`：来自 `snapshot.last_price`
  - `sigma`：来自历史日收益的年化波动率（默认窗口 252）
  - `r, q`：默认先设为 0（后续可用 HKD 利率/分红率替换）
  - `T1=0.5, T2=1.0` 年（示例）
  - `K=150`（注意：港股标的为 HKD 计价，`K` 单位需保持一致；此处仅用于“跑通流程”）

## 结果输出

- Notebook：`report/week3/bsm_chooser.ipynb`
- 脚本（可直接跑 MC）：`src/scripts/chooser_bsm_mc.py`
  - 默认读取 `output/futu/snapshot.csv` 和 `output/futu/kline_HK_00700.csv`
  - 输出 `output/futu/chooser_mc.json`

