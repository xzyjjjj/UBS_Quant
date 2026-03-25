# Data sources (`quant_research.data.sources`)

这个目录放“外部数据源”的拉取/清洗适配器（FRED、Yahoo Finance、Alpha Vantage 等），并在 `__init__.py` 中统一导出对外 API。

## 主要模块

- `fred.py`：FRED 时间序列（含常用美债收益率 series 列表）
- `yahoo_finance.py`：Yahoo/yfinance 日频行情
- `alpha_vantage.py`：Alpha Vantage 日频（adjusted）行情
- `market_panel.py`：示例数据面板（Equity/Vol/Rates）拉取与面板构建
- `run_market_panel_pipeline.py`：可分阶段跑的 pipeline（fetch/panel/features/all）

## 用法示例

```python
from quant_research.data.sources import fetch_fred_series_many, fetch_yahoo_daily
```

需要 API Key 的数据源通常从环境变量读取（例如 `FRED_API_KEY`、`ALPHAVANTAGE_API_KEY`），也支持显式传参（见对应模块/脚本）。

示例 `market_panel` pipeline 的预处理逻辑放在 `quant_research.data.cleaning`：默认用 `equity` 的日期索引作为 trading calendar，过滤 raw 中的非交易日行；并对最终 `panel` 做一次 `ffill`。

## 添加新数据源

1. 新增一个 `*.py`（保持函数式接口，返回 `pandas.DataFrame`）
2. 在 `__init__.py` 里导入并加入 `__all__`
3. 尽量保持列名/索引约定：日期索引命名为 `date`，并去重+排序
