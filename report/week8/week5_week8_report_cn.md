# Week5-Week8 综合报告(中文版)

## 1. 工作范围
本报告汇总 Week5 到 Week8 的主要交付：
- Week5：模型实现与训练对比
- Week6：可解释性分析与指标补齐
- Week7：套利与回测面板工具
- Week8：策略整合与评估

## 2. Week5：模型训练(70/15/15 时序切分)
数据入口：`jpm_options_final.csv`。  
切分方式：按日期严格做 70/15/15，避免 look-ahead bias。

### 时序切分节点(明确日期)
| 划分 | 日期范围 |
|---|---|
| Train | `<= 2023-03-14` |
| Val | `2023-03-15 ~ 2024-02-06` |
| Test | `>= 2024-02-07` |

说明：采用按交易日顺序切分，确保未来样本不会泄漏到过去训练阶段。

### 模型效果(服务器实跑)
| 模型 | Train MAE | Train RMSE | Val MAE | Val RMSE | Test MAE | Test RMSE |
|---|---:|---:|---:|---:|---:|---:|
| MLP | 0.842995 | 1.396970 | 1.477292 | 2.353230 | 6.101606 | 9.964472 |
| BSM+MLP residual | 1.319153 | 2.138775 | 1.313159 | 2.519798 | 2.248075 | 4.030562 |
| Transformer | 15.943744 | 21.209773 | 18.029755 | 23.046355 | 28.232014 | 38.077330 |

结论：`BSM+MLP residual` 在测试集上最优。

### Week5 补充：LSTM / RandomForest / CatBoost 等模型实测
（服务器实跑结果，统一时序切分与评估口径）

| 模型 | Train MAE | Train RMSE | Train R² | Val MAE | Val RMSE | Val R² | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RandomForest | 0.7384 | 1.1223 | 0.9975 | 1.9977 | 3.2796 | 0.9808 | 7.4761 | 12.1837 | 0.8923 |
| CatBoost | 0.5665 | 0.9229 | 0.9983 | 1.5119 | 2.5136 | 0.9887 | 6.5774 | 11.2676 | 0.9079 |
| LSTM | 0.6965 | 1.2512 | 0.9969 | 1.1625 | 1.9720 | 0.9931 | 5.4313 | 9.3939 | 0.9360 |
| GBRT | 2.8450 | 3.6446 | 0.9737 | 3.9176 | 4.8980 | 0.9572 | 9.6679 | 13.7945 | 0.8619 |
| MLP(sklearn) | 0.5377 | 0.9584 | 0.9982 | 1.4070 | 2.2736 | 0.9908 | 5.8325 | 9.4033 | 0.9358 |

补充结论：
- 在该组 Week5 模型实验中，`LSTM` 的测试集指标最佳（Test RMSE 最低、Test R² 最高）。
- `MLP(sklearn)` 与 `CatBoost` 次之，`RandomForest/GBRT` 相对较弱。

## 3. Week6：可解释性与要求映射
已实现脚本：
- `src/scripts/run_week6_mlp_explainability.py`
- `src/scripts/run_week6_mlp_requirements.py`

已覆盖项目要求：
- 超参数搜索
- MAE / RMSE / R²
- 全局可解释性(Permutation)
- 局部可解释性近似(Occlusion 风格)

### 全局可解释性结果(按 delta MAE 由高到低)
| 排名 | 特征 | delta MAE |
|---:|---|---:|
| 1 | `log_moneyness` | 25.423673 |
| 2 | `is_call` | 20.874212 |
| 3 | `moneyness` | 2.603320 |
| 4 | `sqrt_ttm` | 1.953568 |
| 5 | `q` | 0.150669 |

### 五个关键参数含义(大白话)
| 参数 | 含义 | 在定价中的作用 |
|---|---|---|
| `log_moneyness` | `ln(S/K)`，现价与执行价的相对位置 | 决定期权价内/价外程度，是最核心信息之一 |
| `is_call` | 是否看涨期权(call=1, put=0) | 决定使用看涨还是看跌的定价结构 |
| `moneyness` | `S/K`，现价/执行价比值 | 与 `log_moneyness` 同源，补充刻画价内外状态 |
| `sqrt_ttm` | `sqrt(T)`，到期时间开根号 | 对应时间价值扩散项，影响 `d1/d2` 尺度 |
| `q` | 连续分红率 | 影响 `S*e^{-qT}` 和漂移项，改变 call/put 相对价格 |

### Q3 补充：训练/验证/测试分布偏移检查(实证)
针对“是否存在分布偏移导致泛化问题”的问题，已新增独立报告：  
`output/week8_distribution_check/q3_distribution_shift_supplement.md`

核心结果(PSI, train vs test)：
- 高偏移：`r=14.5259`、`q=14.2387`、`sigma=0.2636`、`market_price=0.2589`
- 中等偏移：`moneyness=0.2305`、`log_moneyness=0.2305`、`T_years=0.1204`

结论：存在显著分布偏移风险，尤其是利率/分红率口径与波动率分布变化；后续应采用分层重加权、分桶建模或按 regime 划分验证来提升 OOS 泛化稳定性。

### Q3 补充：PSI 与敏感度联合判读
为避免“只看偏移或只看重要性”的单维误判，已新增联合分析：  
`output/week8_distribution_check/psi_sensitivity_joint.csv`

| 特征 | PSI | delta MAE | 联合结论 |
|---|---:|---:|---|
| `q` | 14.2387 | 2.4669 | 偏移高且影响不低，优先治理(critical) |
| `log_moneyness` | 0.2305 | 22.0687 | 影响极高且有中等偏移，需持续监控(watch) |
| `moneyness` | 0.2305 | 5.1774 | 影响较高且有中等偏移，建议分层建模(watch) |
| `r` | 14.5259 | 0.8641 | 偏移极高但当前敏感度较低，建议口径统一(low) |
| `sigma` | 0.2636 | 0.2627 | 偏移较高但当前敏感度低，建议稳健化(low) |

一句话：**优先处理既“在漂移”又“高敏感”的变量（如 `q`、`moneyness` 相关特征），其次再处理仅偏移高但敏感度低的变量。**

## 4. Week7：工具面板
已实现：
- `src/scripts/jpm_arb_dashboard.py`
- `src/scripts/backtest_panel.py`
- `src/scripts/run_week7_extreme_tests.py`

主要能力：
- 数据源混合模式：`Futu 期权 + AlphaVantage 现货`
- Price-edge 与 Vol-edge 双视角(含双信号确认)
- 回测动作全记录、收益曲线、常用绩效指标展示

### Week7 极端测试(已实跑)
测试场景(已扩展)：
- 波动上冲：`sigma +50%`
- 利率上行：`r +2%`
- 组合冲击：`sigma +50% + r +2%`

样本口径：从 `jpm_options_final.csv` 分层抽样 30,000 条可交易样本(深度 ITM/ATM/深度 OTM 各 10,000)，基于 `BSM+MLP residual` 重估价格。

分层定义（`m = S/K`，且样本走廊 `0.60 <= m <= 1.40`）：
- Call：深度 ITM `m>=1.15`；ATM `0.85<m<1.15`；深度 OTM `m<=0.85`
- Put：深度 ITM `m<=0.85`；ATM `0.85<m<1.15`；深度 OTM `m>=1.15`

| 分层 | 场景 | 样本数 | 均值价格 | p50 | p90 | 相对基准均值变化 | 相对基准中位变化 |
|---|---|---:|---:|---:|---:|---:|---:|
| deep_itm | base | 10000 | 43.1876 | 38.7790 | 69.5109 | 0.0000% | 0.0000% |
| deep_itm | sigma +50% | 10000 | 44.7761 | 40.3176 | 71.2958 | +3.6781% | +2.7416% |
| deep_itm | r +2% | 10000 | 43.0026 | 38.8666 | 69.0020 | -0.4283% | -0.1539% |
| deep_itm | sigma+rate | 10000 | 44.5941 | 40.4776 | 70.6908 | +3.2568% | +2.6758% |
| atm | base | 10000 | 8.3685 | 6.1017 | 20.1081 | 0.0000% | 0.0000% |
| atm | sigma +50% | 10000 | 10.4203 | 8.0171 | 23.5338 | +24.5180% | +32.1788% |
| atm | r +2% | 10000 | 8.3481 | 6.1088 | 20.1117 | -0.2437% | -0.6488% |
| atm | sigma+rate | 10000 | 10.3842 | 8.0127 | 23.7108 | +24.0869% | +31.0206% |
| deep_otm | base | 10000 | 1.2141 | 0.5544 | 5.2176 | 0.0000% | 0.0000% |
| deep_otm | sigma +50% | 10000 | 2.8425 | 1.0821 | 9.3581 | +134.1166% | +81.4440% |
| deep_otm | r +2% | 10000 | 1.1491 | 0.4884 | 5.4120 | -5.3549% | -3.5980% |
| deep_otm | sigma+rate | 10000 | 2.7925 | 1.0255 | 9.5725 | +129.9972% | +72.2296% |

#### Sigma 偏移与均值偏移关系图（按分层）
![Sigma偏移与均值偏移关系图](../../output/week7_extreme_tests/week7_sigma_mean_shift_curve.png)

#### r 单因子冲击与均值偏移关系图（按分层）
![r单因子冲击与均值偏移关系图](../../output/week7_extreme_tests/week7_r_mean_shift_curve.png)

#### sigma+r 联合冲击与均值偏移关系图（按分层）
![sigma+r联合冲击与均值偏移热力图（按分层）](../../output/week7_extreme_tests/week7_sigma_r_joint_mean_shift_heatmap.png)

结论(Week7)：
- 深度 OTM 对波动冲击弹性显著高于 ATM/深度 ITM，尾部敏感性差异明确。  
- 组合冲击(`sigma + rate`)结果与单因子并非简单线性加总，存在非线性耦合。  
- 已补齐“分层极端测试 + 组合冲击 + 工具原型(Streamlit)”。

结果文件：
- `output/week7_extreme_tests/week7_extreme_summary.csv`
- `output/week7_extreme_tests/week7_extreme_summary.md`

## 5. Week8：策略整合与评估
回测引擎升级：
- `src/scripts/backtest_10d_7d_near_spot.py`
- 支持 `mlp_direct` 与 `bsm_mlp_residual`
- 支持 model-edge 排序与流动性过滤

### 调参后策略(平衡版，1 年窗口)
参数摘要：
- `target_dte=14`, `dte_tol=10`
- `max_strike_gap=15`
- `max_moneyness_dev=0.30`
- `min_volume=1`, `min_open_interest=1`
- `model=bsm_mlp_residual`, `use_model_rank=True`
- `force_daily_trade=False`

结果：
- Trades: 41
- Win rate: 58.54%
- Total PnL: 7,452.5
- Profit factor: 4.24

### 策略可视化与交易案例
下图展示了策略参数、回测指标、收益曲线，以及单笔交易动作与开平仓区间验证。

#### 5.1 回测面板总览(参数 + 指标)
![回测面板总览](./fig/backtest_panel_overview.png)

#### 5.2 收益曲线(Equity Curve)
![收益曲线](./fig/backtest_equity_curve.png)

#### 5.3 交易动作样例(All Actions)
![交易动作样例](./fig/backtest_all_actions_case.png)

#### 5.4 开仓与结算日期的标的走势截图
开仓日(示例)：`2024-10-03`  
<img src="./fig/spot_open_day_2024-10-03.png" alt="开仓日标的走势" width="33%" />

结算日(示例)：`2024-10-11`  
<img src="./fig/spot_expiry_day_2024-10-11.png" alt="结算日标的走势" width="33%" />

说明：样例交易在开仓日被模型选中后，标的在到期前上涨，最终对应 call 合约内在价值提升，形成正向 PnL。

#### 5.5 补充：市场行情覆盖与分时段收益统计
基于同一回测文件 `output/week6_backtest_1y_tuned_balanced.csv`（`2024-01-02 ~ 2024-12-31`），已补充专项报告：  
`report/week8/week8_supplement_report_cn.md`

多标的扩展验证（新增）：  
`report/week8/week8_multisymbol_q1_report_cn.md`（5 股票：`JPM/AAPL/MSFT/NVDA/TSLA`，区间 `2024-01-01 ~ 2024-03-31`）

关键结论（简版）：
- 行情覆盖：已覆盖 `bull / bear / sideways` 以及 `high_vol / low_vol`，不只是单一市场状态。
- 分时段统计：已按季度、月度、周内交易日拆分收益，收益主要来自 `Q1` 与 `Q4`，其中周四、周五贡献较高。
- 归因分析：已补充“周五更高、周二偏弱”与“3 月高、1 月低”的原因拆解（DTE/价内价外结构、信号强度、异常值稳健性）；并在 5 股票多区间扩展中验证其稳定性边界。
- 当前限制：本回测为日频日志，尚不包含盘中时间戳，因此本次“分时段”不包含开盘/午盘等盘中切片。

## 6. 交付清单
- 模型训练与推理脚本
- Week6 可解释性与要求对齐脚本
- 套利面板与回测面板
- 可调参回测引擎
- 本 Week5-Week8 中文综合报告
- Week8 补充报告（市场行情覆盖与分时段收益统计）
