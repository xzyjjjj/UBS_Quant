# 四个问题专项回复报告

## 分层定义与绘图口径
本报告中的 moneyness 定义为 `m = S/K`，并采用如下规则：

- 样本走廊：`0.60 <= m <= 1.40`
- 对 `Call`：
  - 深度 ITM：`m >= 1.15`
  - ATM：`0.85 < m < 1.15`
  - 深度 OTM：`m <= 0.85`
- 对 `Put`（方向相反）：
  - 深度 ITM：`m <= 0.85`
  - ATM：`0.85 < m < 1.15`
  - 深度 OTM：`m >= 1.15`

说明：ITM/OTM 按期权方向定义，因此 deep ITM 组中会同时出现 `m>1` 的 call 与 `m<1` 的 put。

## 1) 极端测试是否覆盖深度 ITM / 深度 OTM
已补齐。`src/scripts/run_week7_extreme_tests.py` 已从“近 ATM 样本”改为分层抽样：
- 深度 ITM：`10000`
- ATM：`10000`
- 深度 OTM：`10000`

分层规则是按期权方向识别 ITM/OTM（call 与 put 使用相反 moneyness 方向），避免把 put 的深度 ITM 误分到 OTM。

最新实跑结果（`output/week7_extreme_tests/week7_extreme_summary.csv`）显示：
- 深度 OTM 在 `sigma +50%` 下，均值变动 `+134.1166%`
- ATM 在 `sigma +50%` 下，均值变动 `+24.5180%`
- 深度 ITM 在 `sigma +50%` 下，均值变动 `+3.6781%`

结论：深度价外对波动冲击最敏感，ATM 次之，深度价内最缓，这一差异已被明确量化。

## 2) 是否仅做了单变量冲击，是否有 vol+rate 组合冲击
Call：
- 深度 ITM：`m >= 1.15`
- ATM：`0.85 < m < 1.15`
- 深度 OTM：`m <= 0.85`

Put（方向相反）：
- 深度 ITM：`m <= 0.85`
- ATM：`0.85 < m < 1.15`
- 深度 OTM：`m >= 1.15`

### sigma 单因子冲击：平均价格偏移曲线
图文件：`output/week7_extreme_tests/week7_sigma_mean_shift_curve.png`  
数据文件：`output/week7_extreme_tests/week7_sigma_mean_shift_curve.csv`

![sigma单因子冲击-平均价格偏移曲线](../../output/week7_extreme_tests/week7_sigma_mean_shift_curve.png)

### r 单因子冲击：平均价格偏移曲线
图文件：`output/week7_extreme_tests/week7_r_mean_shift_curve.png`  
数据文件：`output/week7_extreme_tests/week7_r_mean_shift_curve.csv`

![r单因子冲击-平均价格偏移曲线](../../output/week7_extreme_tests/week7_r_mean_shift_curve.png)

### sigma+r 联合冲击：平均价格偏移曲线
图文件：`output/week7_extreme_tests/week7_sigma_r_joint_mean_shift_curve.png`  
数据文件：`output/week7_extreme_tests/week7_sigma_r_joint_mean_shift_curve.csv`

![sigma+r联合冲击-平均价格偏移热力图](../../output/week7_extreme_tests/week7_sigma_r_joint_mean_shift_heatmap.png)

## 3) Futu 期权 + AlphaVantage 现货混合时，是否做了时间戳对齐和实时性校验
已补齐到面板逻辑。`src/scripts/jpm_arb_dashboard.py` 增加了：
- 时间戳抽取与标准化（UTC）
- Hybrid 模式下源间对齐校验：
  - `|Futu option ts - Alpha spot ts| <= 300s` 视为通过
- 数据新鲜度校验：
  - Futu 期权与 Alpha 现货分别按 `<= 900s` SLA 监控
- 未通过时在面板 `Data quality checks` 区域告警显示

这保证了混源时不会“无感错配”，可以直接看到对齐状态和时效状态。

## 4) Price-edge / Vol-edge 是否有明确信号逻辑，是否计入手续费和滑点
已明确且前置到交易决策：
- `gross_edge = |mispricing|`
- `est_cost = fee + slippage + spread_cross`
- `net_edge = gross_edge - est_cost`
- `edge` 保留为 `net_edge`（兼容旧字段）

信号逻辑新增显式字段：
- `price_signal_ok`: `net_edge >= min_edge`
- `vol_signal_ok`: `|vol_edge| >= min_abs_vol_edge`
- `signal_confirmed`: 双视角同时通过
- `signal_logic`: `CONFIRMED_BOTH / UNCONFIRMED`

即：成本不是复盘后扣减，而是先扣成本再判定可交易信号。

## 相关变更文件
- `src/scripts/run_week7_extreme_tests.py`
- `src/scripts/jpm_arb_dashboard.py`
- `src/scripts/README_JPM_ARB_DASHBOARD.md`
- `output/week7_extreme_tests/week7_extreme_summary.csv`
- `report/week8/week5_week8_report_cn.md`
