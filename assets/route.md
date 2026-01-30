很好，这一步你已经**站在“系统设计者”的位置**了。
下面我会给你一份 **「可执行 + 可落地 + 可演进」的技术路线**，重点放在：

1. **清晰的代码架构（可视化）**
2. **时间轴上：每一阶段你“该改哪些文件、为什么改”**
3. **预测 / 时间序列 / RL 决策 三者在架构中的正确位置**

这不是“学习清单”，而是 **一条工程化研究路线图**。

---

# 一、最终目标架构（先给你“终局图”）

你最终要收敛到的，是一个 **Predict–Decide–Backtest** 的研究系统：

```
┌──────────────────────────────────────────┐
│              run_experiment.py            │
│        (环境初始化 / 配置 / 调度)           │
└───────────────┬──────────────────────────┘
                │
        ┌───────▼────────┐
        │   Data Layer   │
        │ 数据准备 / 清洗 │
        └───────┬────────┘
                │
        ┌───────▼────────┐
        │ Feature Layer  │
        │ 时间序列 / Alpha│
        └───────┬────────┘
                │
        ┌───────▼────────┐
        │ Label Layer    │
        │ Future Return  │
        └───────┬────────┘
                │
        ┌───────▼────────┐
        │ Predictor      │  ← TS / ML / DL
        │ (只预测，不决策) │
        └───────┬────────┘
                │  预测分布 / score
        ┌───────▼────────┐
        │ Decision Layer │  ← RL / Rule
        │ (仓位 / 权重)   │
        └───────┬────────┘
                │  trades / weights
        ┌───────▼────────┐
        │ Backtest       │
        │ 回测 / 指标     │
        └───────┬────────┘
                │
        ┌───────▼────────┐
        │ Portfolio      │
        │ 组合 / 风控     │
        └────────────────┘
```

> **关键原则**
>
> * Predictor 永远不碰仓位
> * RL 永远不预测
> * Backtest 永远是被动执行

---

# 二、目录结构（你可以直接照这个建）

```text
quant_research/
├── run_experiment.py          # ★总入口（你现在的 main）
│
├── config/
│   └── default.yaml           # 回测 / 模型 / RL 超参
│
├── data/
│   ├── loader.py              # 数据准备
│   └── preprocess.py          # 清洗 / 对齐 / 示例
│
├── features/
│   ├── base.py                # Feature 抽象
│   ├── ts_features.py         # 时间序列特征
│   └── alpha101.py            # Alpha101
│
├── labels/
│   └── return_label.py        # 未来收益 / rank
│
├── models/
│   ├── base.py                # Predictor 抽象
│   ├── linear.py              # OLS / Ridge
│   ├── tree.py                # LightGBM
│   └── ts_model.py            # AR / LSTM
│
├── decision/
│   ├── base.py                # 决策接口
│   ├── rule_policy.py         # 规则策略
│   └── rl_policy.py           # 强化学习
│
├── backtest/
│   ├── engine.py              # 回测引擎
│   └── metrics.py             # Sharpe / IC
│
├── portfolio/
│   └── allocator.py           # 权重 / 风控
│
└── utils/
    ├── logger.py
    └── seed.py
```

---

# 三、技术路线（时间轴 + 你该改哪些文件）

下面是**真正重要的部分**。

---

## Phase 0（第 0–1 周）

### 🎯 目标：**只重构，不加新模型**

### 你要做什么

把你现在那份脚本代码 **拆散、归位**。

### 你应该修改 / 新建的文件

| 文件                       | 动作                        |
| ------------------------ | ------------------------- |
| `run_experiment.py`      | 把 main 流程搬过来              |
| `data/loader.py`         | 把 `data_preparation` 放进去  |
| `features/base.py`       | 抽象 Feature 接口             |
| `backtest/engine.py`     | 搬 `_evaluate_return_t_p1` |
| `backtest/metrics.py`    | 搬 Sharpe 计算               |
| `portfolio/allocator.py` | 搬 equity curve            |

❗ **这一阶段禁止**：

* 加 ML
* 加 RL
* 加新 alpha

> 做完这一阶段，你的代码就已经 **比 90% 实习生专业**

---

## Phase 1（第 2–3 周）

### 🎯 目标：**系统掌握时间序列（TS）**

### 学什么（只学这些）

* rolling mean / std
* momentum / reversal
* autocorr
* regime（高低波动）

### 你该改哪些文件

| 文件                        | 为什么        |
| ------------------------- | ---------- |
| `features/ts_features.py` | 所有 TS 特征   |
| `features/base.py`        | Feature 接口 |
| `labels/return_label.py`  | 明确预测目标     |

#### 示例（你应该能写出来）

```python
class RollingMomentum(TimeSeriesFeature):
    def compute(self, series):
        return series.pct_change().rolling(20).mean()
```

📌 到这一步：
你已经**不是“写 alpha 的人”，而是“建 TS 特征的人”**

---

## Phase 2（第 4–5 周）

### 🎯 目标：**Predictor 专职预测**

### 学什么

* Linear regression（baseline）
* LightGBM（主力）

### 你该改哪些文件

| 文件                  | 为什么          |
| ------------------- | ------------ |
| `models/base.py`    | Predictor 抽象 |
| `models/linear.py`  | baseline     |
| `models/tree.py`    | 主模型          |
| `run_experiment.py` | 选择模型         |

❗ 注意：

* **模型输出是 score，不是 signal**
* 不许出现仓位逻辑

---

## Phase 3（第 6–7 周）

### 🎯 目标：**RL 只负责决策**

### RL 在你系统里的位置

```text
Predictor → 预测分布
        ↓
DecisionPolicy (RL)
        ↓
target_position
```

### 你该改哪些文件

| 文件                        | 内容            |
| ------------------------- | ------------- |
| `decision/base.py`        | 决策接口          |
| `decision/rule_policy.py` | 非 RL baseline |
| `decision/rl_policy.py`   | DQN / PPO     |

#### RL State 示例（你必须明确）

```python
state = {
    "alpha": predicted_return,
    "position": current_position,
    "vol": volatility,
    "drawdown": drawdown,
}
```

📌 这一步你已经在 **用 RL 做对的事**

---

## Phase 4（第 8–9 周）

### 🎯 目标：**回测 & 研究规范**

### 你该补的不是模型，而是：

* IC / RankIC
* Turnover
* Cost
* Walk-forward

### 你该改的文件

| 文件                    | 为什么   |
| --------------------- | ----- |
| `backtest/metrics.py` | 研究指标  |
| `config/default.yaml` | 实验可复现 |

---

## Phase 5（第 10+ 周）

### 🎯 目标：**对齐 Qlib 思维**

不是“换成 Qlib”，而是：

* Feature Handler
* Dataset
* Model
* Strategy
* Backtest

你会发现：
👉 **你的框架已经“像 Qlib 了”**

---

# 四、一个非常重要的提醒（经验之谈）

> **不要同时改架构 + 学 RL + 学 DL**

正确顺序永远是：

```
架构稳定
→ TS / ML
→ 决策（RL）
→ 加速
```

---

# 五、我接下来可以直接帮你做什么（更具体）

你现在已经非常适合进入 **“一起改代码”阶段**。
我可以直接：

1. 🔧 **帮你把当前这份代码，拆成 Phase 0 的完整目录**
2. 🧠 **给你一套 TS Feature + Label 的标准实现**
3. 🤖 **给你一个最小可用 RL 决策模块（可直接接回测）**

你选一个，我下一条消息直接进入「实操级别」。
