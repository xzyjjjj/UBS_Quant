# Data (`quant_research.data`)

这个包提供“数据获取 → 预处理 → 输出面板”的最小可复用 pipeline。

## Pipeline 行为（以 market panel 为例）

入口脚本：`src/quant_research/data/sources/run_market_panel_pipeline.py`

1. **Fetch（sources）**：拉取 `equity / vol / treasury` 三类 raw 数据（见 `quant_research.data.sources.market_panel.fetch_raw`）
2. **Trading calendar（cleaning）**：默认用 `equity` raw 的日期索引作为 trading calendar（`quant_research.data.cleaning.get_trading_calendar`）
3. **Raw 过滤（cleaning）**：raw 数据按 trading calendar 删除非交易日行（`quant_research.data.cleaning.filter_raw_to_calendar`）
4. **Build panel（sources）**：按 `date` 外连接构建统一面板（`quant_research.data.sources.market_panel.build_panel`）
5. **Fill（pipeline）**：对最终 `panel` 做一次 `ffill`
6. **Save（sources）**：输出 `raw/` 与 `processed/` 到 `output/market_panel/`

## 输出结构

- `output/market_panel/raw/`
  - `equity_daily.csv`
  - `vol_daily.csv`
  - `treasury_yields.csv`
- `output/market_panel/processed/`
  - `panel.csv`
  - `panel.parquet`（可选，依赖 pyarrow/fastparquet）
