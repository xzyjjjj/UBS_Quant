# Week 1 数据规范（Data Spec）

时间范围（默认）：`2018-01-01` ~ `2024-12-31`

## 1) Yahoo Finance

### JPM（日频）
- 标的：`JPM`
- 频率：日频（交易日）
- 字段（raw）：`Open/High/Low/Close/Adj Close/Volume`
- 落盘：
  - `output/week1/raw/jpm_daily.csv`

## 2) FRED（美债利率：尽量“全”）

使用 FRED 的常见“常数期限国债收益率”（daily, 单位：百分比水平，例如 4.12 表示 4.12%）。

- Series IDs（默认）：
  - `DGS1MO, DGS3MO, DGS6MO, DGS1, DGS2, DGS3, DGS5, DGS7, DGS10, DGS20, DGS30`
- 频率：日频（非交易日可能缺失）
- 落盘：
  - `output/week1/raw/treasury_yields.csv`

### VIX（2018–2024，日频，默认推荐）
- Series ID：`VIXCLS`
- 频率：日频（交易日/发布日）
- 字段（raw）：`VIXCLS`
- 落盘：
  - `output/week1/raw/vix_daily.csv`

## 3) Alpha Vantage（可选备源）

仅用于 **JPM** 日线备源（需要 key 且有调用频率限制）。

## 4) 统一面板（processed）

输出一个以 `date` 为索引的外连接面板，供 Week2 清洗/特征工程直接读取。

- 文件：
  - `output/week1/processed/panel.csv`
  - `output/week1/processed/panel.parquet`（可选：依赖 pyarrow/fastparquet）
- 列名约定：
  - JPM：`jpm_open/jpm_high/jpm_low/jpm_close/jpm_adj_close/jpm_volume`
  - VIX（FRED）：`vix_close`（由 `VIXCLS` 重命名）
  - VIX（Yahoo 可选）：`vix_open/vix_high/vix_low/vix_close/vix_adj_close/vix_volume`
  - 美债：`DGS1MO ... DGS30`（保持 FRED 原 series id）

## 5) 一键脚本

执行入口：`src/quant_research/data/sources/fetch_wk01.py`
