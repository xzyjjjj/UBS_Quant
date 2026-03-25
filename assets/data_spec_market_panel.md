# Market panel 数据规范（Data Spec）

时间范围（默认）：`2018-01-01` ~ `2024-12-31`

## 1) Yahoo Finance

### Equity（日频）
- 标的：默认 `JPM`（可通过脚本参数修改）
- 频率：日频（交易日）
- 字段（raw）：`Open/High/Low/Close/Adj Close/Volume`（并开启 actions 时额外包含 `Dividends/Stock Splits`）
- 落盘：
  - `output/market_panel/raw/equity_daily.csv`

## 2) FRED（美债利率：尽量“全”）

使用 FRED 的常见“常数期限国债收益率”（daily, 单位：百分比水平，例如 4.12 表示 4.12%）。

- Series IDs（默认）：
  - `DGS1MO, DGS3MO, DGS6MO, DGS1, DGS2, DGS3, DGS5, DGS7, DGS10, DGS20, DGS30`
- 频率：日频（非交易日可能缺失）
- 落盘：
  - `output/market_panel/raw/treasury_yields.csv`

### VIX（2018–2024，日频，默认推荐）
- Series ID：`VIXCLS`
- 频率：日频（交易日/发布日）
- 字段（raw）：`VIXCLS`
- 落盘：
  - `output/market_panel/raw/vol_daily.csv`

## 3) Alpha Vantage（可选备源）

仅用于 **Equity** 日线备源（需要 key 且有调用频率限制）。

## 4) 统一面板（processed）

输出一个以 `date` 为索引的外连接面板，供清洗/特征工程直接读取。

- 文件：
  - `output/market_panel/processed/panel.csv`
  - `output/market_panel/processed/panel.parquet`（可选：依赖 pyarrow/fastparquet）
- 列名约定：
  - Equity：`equity_open/equity_high/equity_low/equity_close/equity_adj_close/equity_volume`
  - Vol（FRED）：`vol_close`（由 `VIXCLS` 重命名）
  - Vol（Yahoo 可选）：`vol_open/vol_high/vol_low/vol_close/vol_adj_close/vol_volume`
  - 美债：`DGS1MO ... DGS30`（保持 FRED 原 series id）
  - 日收益率：`equity_ret_1d`，基于 `equity_adj_close`（若缺失则回退 `equity_close`）计算 simple return（`pct_change()`）
  - 滚动波动率（Realized Vol，年化）：`equity_rvol_{window}`（默认输出 `21/63/252`），基于 `equity_adj_close`（若缺失则回退 `equity_close`）计算 log return，再做 `rolling(window).std()`，并乘以 `sqrt(252)` 年化
  - 分红增长（逐次派息增长）：`equity_div_growth`，仅在分红事件上计算 `div_t / div_prev - 1`，并对非分红日做 `ffill`（无历史分红则为 NaN）
  - VIX 与 Equity 相关性：`equity_vix_corr_{window}`（默认 `21/63/252`），相关性计算基于 `equity_ret_1d` 与 `vol_close.pct_change()`
  - 利率动量（DGS10）：`rate_mom_dgs10_21/63` 及 `rate_mom_dgs10_21/63_z252`，动量为 `DGS10.diff(h)`，zscore 为滚动 252 日标准化
  - 新闻情绪分数（0--1）：`news_sent_01` 与 `news_sent_01_ewm7`（来源：`output/news/sentiment_daily.csv`，由 GDELT tone 日聚合后 sigmoid 映射得到）

## 5) 一键脚本

执行入口：`src/quant_research/data/sources/run_market_panel_pipeline.py`

可分阶段运行（避免每次加 feature 都重新下载）：
- `--mode fetch`：只下载 raw 并落盘（会触网）
- `--mode panel`：只用已落盘 raw 构建 `processed/panel.csv`（不触网）
- `--mode features`：只在已有 `processed/panel.csv` 上追加特征列（不触网）
- `--mode all`：完整流程（默认）

缓存策略（避免冲掉本地 raw）：
- 默认会优先使用 `output/market_panel/raw/` 下的缓存；除非传 `--refresh-raw` 才会强制重新下载并覆盖
