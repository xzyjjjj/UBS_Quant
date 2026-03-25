# Features (`quant_research.features`)

本目录放“特征构建”逻辑；数据拉取与落盘在 `quant_research.data`，避免每次加特征都重新下载。

## Market panel 已实现特征

- 滚动波动率（Realized Vol，年化）：`rolling_volatility.py`（输出 `equity_rvol_{window}`，默认 `21/63/252`；log return；×`sqrt(252)`）
- 日收益率：`daily_return.py`（输出 `equity_ret_1d`；优先 `equity_adj_close`，回退 `equity_close`；simple return）
- 分红增长（逐次派息增长 A）：`dividend_growth.py`（输出 `equity_div_growth`；分红事件日算 `div_t/div_prev - 1`；非分红日默认 `ffill`）
- VIX 与 JPM 相关性：`equity_vol_correlation.py`（输出 `equity_vix_corr_{window}`，默认 `21/63/252`；相关性计算基于 `equity_ret_1d` 与 `vol_close.pct_change()`）
- 利率动量：`rate_momentum.py`（输出 `rate_mom_dgs10_21/63` 及其 `z252` 版本；动量用 `DGS10.diff(h)`）
- 新闻情绪分数（0--1）：`news_sentiment.py`（读取 `output/news/sentiment_daily.csv` 并输出 `news_sent_01` 与 `news_sent_01_ewm7`；缺失默认 `ffill`）

## 运行方式（只做特征，不触网）

当 `output/market_panel/processed/panel.csv` 已存在时：

```bash
python src/quant_research/data/sources/run_market_panel_pipeline.py --mode features
```

## 新闻情绪数据生成（触网）

```bash
python src/quant_research/data/sources/run_news_sentiment_pipeline.py --start 2018-01-01 --end 2024-12-31 --query '(JPM OR "JPMorgan")'
```

如遇到 TLS/连接不稳定，可尝试切换到 HTTP：

```bash
python src/quant_research/data/sources/run_news_sentiment_pipeline.py --fallback-http --gdelt-base-url http://api.gdeltproject.org/api/v2/doc/doc
```

## 数据依赖与解耦约定

- 分红数据必须在拉取阶段获取：Yahoo/yfinance 以 `actions=True` 拉取 `Dividends/Stock Splits`（见 `quant_research.data.sources.market_panel.fetch_raw`）。
- raw 缓存默认不覆盖：`output/market_panel/raw/` 已存在时，`--mode fetch` 默认复用缓存；只有传 `--refresh-raw` 才会强制重下并覆盖（且不会用空结果覆盖已有文件）。
- yfinance 失败会打印是哪个数据集失败（equity/vol/treasury），便于定位 rate limit。
