下面是一份**可直接当讲义使用**的 Markdown（包含你图里的 workflow、3 种 Alpha 输出形式的“具体实现”，并且在每种实现里都给出对应的 **Predictor + Decision** 示例）。
我尽量把代码写成**最小可运行/可扩展**的“Qlib 风格分层接口”，你之后可以很自然地拆到文件里。

---

# 讲义：Predict–Alpha–Decide 工作流与实现模板

## 0. 目标

你希望构建一个“预测与决策分离”的量化研究/交易管线：

* Predictor（TS/ML/DL）只负责**从市场数据中提取可预测信息**
* AlphaOutput 是 Predictor 输出的**信息接口**（可以是 score、(μ,σ)、概率分布）
* Decision（Rule/RL）只负责**把 Alpha + 状态（仓位/风险/成本）映射为行动（仓位/交易）**

---

## 1. 工作流（与图一致）

```
Market Data
  ↓
Predictor (TS / ML / DL)
  ↓
Alpha Output (score / (μ,σ) / dist)
  ↓
Decision Model (Rule / RL)
```

### 1.1 每一层的职责边界（强约束）

* **Market Data**：原始行情与衍生特征（价格、成交量、波动率、因子等）
* **Predictor**：输出“对未来收益有用的信息”（AlphaOutput），不产生仓位、不考虑交易成本
* **AlphaOutput**：信息载体，形式可弱可强（score → (μ,σ) → dist）
* **Decision Model**：消费 AlphaOutput + 状态（position, risk, cost, constraints），输出 action（仓位/权重/交易）

---

## 2. 统一接口：AlphaOutput / Predictor / Decision

先给一个统一的“接口层”，后面三种 Alpha 形式都复用它。

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, Protocol
import numpy as np


# -------- Alpha Output: 三种形态的统一承载 --------
@dataclass(frozen=True)
class AlphaOutput:
    # 1) score: 最常见（一个连续分数）
    score: Optional[np.ndarray] = None  # shape: [N_assets] or [N_samples]

    # 2) parametric uncertainty: (mu, sigma)
    mu: Optional[np.ndarray] = None     # E[r]
    sigma: Optional[np.ndarray] = None  # Std[r]

    # 3) distribution-like: 用 quantiles/样本近似分布（工程上很常用）
    # 例如 q = {0.1: q10, 0.5: q50, 0.9: q90}
    quantiles: Optional[Dict[float, np.ndarray]] = None

    # 额外元信息（可选）：模型版本、训练窗口等
    meta: Optional[Dict[str, Any]] = None


# -------- Predictor 接口：只输出 AlphaOutput --------
class Predictor(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> None: ...
    def predict_alpha(self, X: np.ndarray) -> AlphaOutput: ...


# -------- Decision 接口：只消费 AlphaOutput + state，输出 action --------
# action 在最小例子里用 target_position 表示（-1~+1 或 0~1 权重）
class DecisionModel(Protocol):
    def act(self, alpha: AlphaOutput, state: Dict[str, Any]) -> np.ndarray: ...
```

> 说明：
>
> * `X` 可以是 TS 特征（rolling return、vol、alpha101 等），也可以是截面特征
> * `state` 建议至少包含：`position`（当前仓位）、`risk`（波动/回撤）、`cost`（交易成本参数）

---

## 3. 三种 Alpha 形式：实现 + Predictor + Decision

下面分 3 节：

* (A) **score alpha**（最基础）
* (B) **(μ,σ) alpha**（推荐作为你系统默认接口）
* (C) **dist alpha**（用 quantiles 表达“分布”，更适合风控/RL）

为便于讲义演示，下面用 numpy 直接写最小实现；你落地时可替换为 LightGBM / PyTorch / statsmodels。

---

# A. Alpha 形式 1：score（一个连续分数）

## A.1 Predictor：线性回归输出 score

```python
class LinearScorePredictor:
    """
    最小实现：线性回归 (OLS) -> 输出 score = Xw
    注：这只是讲义示例，真实可替换为 LGBM / NN。
    """
    def __init__(self):
        self.w = None  # shape [D]

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # OLS: w = (X'X)^-1 X'y
        X_ = np.asarray(X)
        y_ = np.asarray(y)
        self.w = np.linalg.pinv(X_.T @ X_) @ (X_.T @ y_)

    def predict_alpha(self, X: np.ndarray) -> AlphaOutput:
        score = np.asarray(X) @ self.w
        return AlphaOutput(score=score, meta={"type": "score", "model": "ols"})
```

## A.2 Decision：规则策略（基于 score 的阈值/分位数）

```python
class ThresholdDecision:
    """
    action = target_position ∈ {-1, 0, +1}
    """
    def __init__(self, buy_th: float = 0.0, sell_th: float = 0.0):
        self.buy_th = buy_th
        self.sell_th = sell_th

    def act(self, alpha: AlphaOutput, state: Dict[str, Any]) -> np.ndarray:
        score = alpha.score
        assert score is not None, "ThresholdDecision requires alpha.score"
        pos = np.zeros_like(score)
        pos[score > self.buy_th] = 1.0
        pos[score < -self.sell_th] = -1.0
        return pos
```

### A.3 什么时候用 score？

* 你在做横截面排序、Top-K/Bottom-K、或简单 long/short
* 你先把系统跑通、指标（IC/RankIC/Sharpe/Turnover）齐全
* 成本/风险由决策层粗粒度处理

---

# B. Alpha 形式 2：（μ, σ）——强烈推荐作为默认接口

这相当于你给决策层传递：

> “我预测未来收益的**期望**是多少，**不确定性（风险）**是多少”

## B.1 Predictor：Bootstrap/Ensemble 估计 (μ, σ)

讲义里用“多次重采样训练线性模型”来近似 ensemble。真实工程可用：

* LightGBM ensemble
* MC dropout
* Deep ensemble
* Bayesian regression

```python
class EnsembleMuSigmaPredictor:
    """
    通过 B 次 bootstrap 拟合得到一组预测 {y_hat^b}
    mu = mean(y_hat^b), sigma = std(y_hat^b)
    """
    def __init__(self, B: int = 20, seed: int = 42):
        self.B = B
        self.seed = seed
        self.ws = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        rng = np.random.default_rng(self.seed)
        X_ = np.asarray(X); y_ = np.asarray(y)
        n = X_.shape[0]
        self.ws = []
        for _ in range(self.B):
            idx = rng.integers(0, n, size=n)  # bootstrap
            Xb, yb = X_[idx], y_[idx]
            w = np.linalg.pinv(Xb.T @ Xb) @ (Xb.T @ yb)
            self.ws.append(w)

    def predict_alpha(self, X: np.ndarray) -> AlphaOutput:
        X_ = np.asarray(X)
        preds = np.stack([X_ @ w for w in self.ws], axis=0)  # [B, N]
        mu = preds.mean(axis=0)
        sigma = preds.std(axis=0, ddof=1) + 1e-12
        return AlphaOutput(mu=mu, sigma=sigma, meta={"type": "mu_sigma", "B": self.B})
```

## B.2 Decision：风险调整后的仓位（“信息比率”风格）

一个非常经典、工程上很实用的策略：

> position ∝ μ / σ（类似 Sharpe/信息比率），再做截断与成本控制

```python
class RiskAdjustedDecision:
    """
    target_position = clip( k * mu/sigma, -max_pos, +max_pos )
    可加入 position smoothing / turnover penalty
    """
    def __init__(self, k: float = 1.0, max_pos: float = 1.0):
        self.k = k
        self.max_pos = max_pos

    def act(self, alpha: AlphaOutput, state: Dict[str, Any]) -> np.ndarray:
        mu, sigma = alpha.mu, alpha.sigma
        assert mu is not None and sigma is not None, "RiskAdjustedDecision requires (mu, sigma)"
        raw = self.k * (mu / sigma)
        pos = np.clip(raw, -self.max_pos, self.max_pos)

        # 可选：简单“换手抑制”（turnover penalty）
        prev = state.get("position", np.zeros_like(pos))
        turnover_limit = state.get("turnover_limit", None)
        if turnover_limit is not None:
            delta = np.clip(pos - prev, -turnover_limit, turnover_limit)
            pos = prev + delta

        return pos
```

### B.3 为什么（μ,σ）是默认最佳？

* 决策层天然需要风险信息（仓位大小、风控、成本）
* 不用真的建完整分布，也能显著提升“可决策性”
* 与 RL 结合非常顺：state 里直接放 `mu, sigma`

---

# C. Alpha 形式 3：dist（用 quantiles 表达“分布”）

完整概率分布很难估计、也很容易估错。工程里常用替代是：

> **用分位数（quantiles）作为“分布近似”**
> 如：q10 / q50 / q90 或 q05 / q50 / q95

## C.1 Predictor：输出分位数（用 ensemble 的预测分布近似）

复用前面的 ensemble 预测集合 `preds[B, N]`，直接取分位数。

```python
class QuantileDistPredictor:
    def __init__(self, B: int = 50, quantile_levels=(0.1, 0.5, 0.9), seed: int = 7):
        self.B = B
        self.levels = quantile_levels
        self.seed = seed
        self.ws = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        rng = np.random.default_rng(self.seed)
        X_ = np.asarray(X); y_ = np.asarray(y)
        n = X_.shape[0]
        self.ws = []
        for _ in range(self.B):
            idx = rng.integers(0, n, size=n)
            Xb, yb = X_[idx], y_[idx]
            w = np.linalg.pinv(Xb.T @ Xb) @ (Xb.T @ yb)
            self.ws.append(w)

    def predict_alpha(self, X: np.ndarray) -> AlphaOutput:
        X_ = np.asarray(X)
        preds = np.stack([X_ @ w for w in self.ws], axis=0)  # [B, N]
        q = {}
        for lvl in self.levels:
            q[lvl] = np.quantile(preds, lvl, axis=0)
        return AlphaOutput(quantiles=q, meta={"type": "quantile_dist", "levels": list(self.levels)})
```

## C.2 Decision：基于尾部风险（CVaR 风格/下行风险）做仓位

一个常见决策逻辑：

* 以 q50 作为“中位收益”
* 用下行分位（q10）衡量 tail risk
* reward-risk = q50 / |q10|，再做仓位

```python
class TailRiskDecision:
    """
    使用 quantiles: q10/q50/q90
    做一个简单的下行风险调整：pos ∝ q50 / |q10|
    """
    def __init__(self, max_pos: float = 1.0, eps: float = 1e-12):
        self.max_pos = max_pos
        self.eps = eps

    def act(self, alpha: AlphaOutput, state: Dict[str, Any]) -> np.ndarray:
        qs = alpha.quantiles
        assert qs is not None and 0.1 in qs and 0.5 in qs, "TailRiskDecision requires quantiles 0.1 and 0.5"
        q10 = qs[0.1]
        q50 = qs[0.5]

        raw = q50 / (np.abs(q10) + self.eps)
        pos = np.clip(raw, -self.max_pos, self.max_pos)

        # 可加入成本：若 cost 高则提高开仓门槛
        cost = state.get("cost", 0.0)
        if cost > 0:
            pos[np.abs(q50) < cost] = 0.0
        return pos
```

### C.3 dist 形式什么时候上？

* 你要让决策层更“风险敏感”（尤其是尾部风险）
* 你希望 RL 的 state 里有更丰富的不确定性描述（quantiles 比 (μ,σ) 更 robust）

---

## 4. 把三种形式串起来：一个最小可运行的 Demo

下面给一个“讲义级 demo”：你只需换 predictor/decision，就能跑不同 Alpha 形式。
（这里用随机数据模拟，真实你把 X/y 替换为你 minutes/daily 的特征与未来收益标签即可。）

```python
def demo_run(predictor: Predictor, decision: DecisionModel, X_train, y_train, X_test):
    predictor.fit(X_train, y_train)
    alpha = predictor.predict_alpha(X_test)

    # state: 你可以塞 position/risk/cost/turnover_limit 等
    state = {"position": np.zeros(X_test.shape[0]), "cost": 0.0005, "turnover_limit": 0.2}
    action = decision.act(alpha, state)
    return alpha, action


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N, D = 2000, 10
    X = rng.normal(size=(N, D))
    true_w = rng.normal(size=(D,))
    y = X @ true_w + 0.1 * rng.normal(size=(N,))  # 伪造“可预测”的 future return

    X_train, y_train = X[:1500], y[:1500]
    X_test = X[1500:]

    # A) score
    alpha_a, action_a = demo_run(
        LinearScorePredictor(),
        ThresholdDecision(buy_th=0.0, sell_th=0.0),
        X_train, y_train, X_test
    )
    print("A score action stats:", action_a.mean(), action_a.std())

    # B) (mu, sigma)
    alpha_b, action_b = demo_run(
        EnsembleMuSigmaPredictor(B=30),
        RiskAdjustedDecision(k=1.0, max_pos=1.0),
        X_train, y_train, X_test
    )
    print("B mu/sigma action stats:", action_b.mean(), action_b.std())

    # C) dist (quantiles)
    alpha_c, action_c = demo_run(
        QuantileDistPredictor(B=60, quantile_levels=(0.1, 0.5, 0.9)),
        TailRiskDecision(max_pos=1.0),
        X_train, y_train, X_test
    )
    print("C dist action stats:", action_c.mean(), action_c.std())
```

---

## 5. 你接下来把它落到“Qlib 风格工程”该怎么拆文件？

你之前要的 10 步流程，映射到代码文件最清晰的方式如下：

### 5.1 文件映射表（建议你照抄）

| 讲义模块  | 建议文件                                  | 内容                          |
| ----- | ------------------------------------- | --------------------------- |
| 数据准备  | `data/loader.py`                      | 读 HDF / parquet / qlib 数据   |
| 环境初始化 | `run_experiment.py` + `utils/seed.py` | seed、日志、配置加载                |
| 模型构建  | `models/*`                            | Predictor 实现（TS/ML/DL）      |
| 数据集构建 | `features/*` + `labels/*`             | 特征与标签                       |
| 回测配置  | `config/default.yaml`                 | 成本、滑点、持有期、频率                |
| 数据示例  | `data/preprocess.py`                  | dump sample / sanity check  |
| 训练记录  | `utils/logger.py`                     | 指标记录、曲线、参数                  |
| 信号生成  | `decision/*`                          | Decision 输出 action/position |
| 信号分析  | `backtest/metrics.py`                 | IC/RankIC/Turnover/Drawdown |
| 组合回测  | `backtest/engine.py` + `portfolio/*`  | PnL、权重、风控                   |

---

## 6.（可选）RL Decision 在这个讲义体系里怎么接？

你这条讲义已经把“接口”定义好了：RL 只需实现 `DecisionModel.act()`。

最小 RL 版本可以从**Contextual Bandit**开始（比 PPO/DQN 更稳更好调）。
如果你要我补充，我可以在下一条里给你一个**可运行的最小 bandit/policy gradient**，直接消费 `mu, sigma` 或 quantiles。

---

# 你现在可以怎么用这份讲义（建议执行顺序）

1. 先用 **B (μ,σ)** 跑通整个 pipeline（推荐默认）
2. 再实现 **A score** 作为 baseline
3. 最后上 **C quantile dist**，用于风险敏感决策 / RL state 丰富化

---

如果你愿意，我下一步可以把这份讲义**升级为“你的项目骨架”**：

* 直接按目录生成所有文件的模板（每个文件写好类与 TODO），并把你现有的分钟级示例代码迁移进去（对应你那 10 个步骤）。



下面是**在上一版讲义基础上的“微调增强版完整讲义”**。
我**不改变整体逻辑结构、不推翻代码示例、不重写章节顺序**，只做你要求的两点优化：

1. **明确并抬高 Feature（特征）在工作流中的“底层地位”**
2. **对已出现的每一种 Predictor / Decision 算法，补充“概括性说明（算法语义）”**

你可以把它当成：

> **“同一份讲义，但从‘会用’升级为‘讲得清’版本”**

---

# 讲义：Predict–Alpha–Decide 工作流与实现模板（优化版）

---

## 0. 讲义目标（不变，但语义更清晰）

本讲义展示一种**严格分层、可扩展、符合现代量化研究共识**的系统设计方式：

* **Feature（特征）** 是整个系统的**信息基础**
* **Predictor（预测）** 只负责从 Feature 中提取“对未来收益有用的信息”
* **AlphaOutput** 是 Predictor 输出的信息接口（可以有多种表达）
* **Decision（决策）** 消费 Alpha + 状态，输出可执行的行动

核心原则仍然是：

> **预测与决策解耦，信息与行动分离**

---

## 1. 工作流（强调 Feature 的“底层地位”）

### 1.1 完整工作流（语义增强版）

```
Market Data
  ↓
Feature Engineering (TS / Alpha / Cross-Section)
  ↓
Predictor (TS / ML / DL)
  ↓
Alpha Output (score / (μ,σ) / dist)
  ↓
Decision Model (Rule / RL)
```

> ⚠️ **关键强调（新增）**
>
> * **Feature 是 Predictor 的“唯一信息来源”**
> * Predictor 不直接“看市场”，它只“看 Feature”
> * Feature 的质量，决定了 Alpha 的上限

---

### 1.2 各层职责重新精确定义

#### Market Data（原始数据层）

* 行情数据（price, volume, order book）
* 公司数据、宏观数据、另类数据
* **不直接进入模型**

---

#### Feature Engineering（特征层｜系统的地基）

> **Feature 是对市场的“可学习表示”**

* 时间序列特征（returns, volatility, momentum）
* 横截面特征（rank, zscore）
* Alpha101 / 手工因子
* 统计/结构性变换（rolling, decay）

📌 **非常重要的一点：**

> **Predictor 的能力上限 = Feature 的信息上限**

换句话说：

* 没有 Feature → 没有 Alpha
* Feature 决定“能不能预测”
* Model 只决定“怎么预测”

---

#### Predictor（预测层）

> **从 Feature 中估计未来收益的“信息结构”**

* 不关心仓位
* 不关心交易成本
* 不关心 reward

它的**唯一输出**是：**AlphaOutput**

---

#### Alpha Output（信息接口层）

> **Alpha 是 Predictor 与 Decision 之间的“信息契约”**

* 可以是弱形式（score）
* 可以是带不确定性形式（μ,σ）
* 可以是分布近似（quantiles）

---

#### Decision Model（决策层）

> **在约束、风险、成本下，把 Alpha 转化为行动**

* 可以是规则
* 可以是强化学习
* 输出是：仓位 / 权重 / 交易指令

---

## 2. 统一接口（保持不变，语义增强）

（接口代码与上一版完全一致，这里不再修改）

```python
@dataclass(frozen=True)
class AlphaOutput:
    score: Optional[np.ndarray] = None
    mu: Optional[np.ndarray] = None
    sigma: Optional[np.ndarray] = None
    quantiles: Optional[Dict[float, np.ndarray]] = None
    meta: Optional[Dict[str, Any]] = None
```

> **语义补充**
>
> * `score`：只表达“相对好坏”
> * `(mu, sigma)`：表达“期望 + 不确定性”
> * `quantiles`：表达“分布结构（尤其是尾部）”

---

## 3. 三种 Alpha 形式（实现 + 算法语义说明）

下面仍然是 **A / B / C 三种形式**，
但每一种我都会补充一个 **“算法在系统中的意义说明”**。

---

# A. Alpha 形式 1：Score Alpha

## A.0 这一类 Alpha 在做什么？（新增概括）

> **Score Alpha 的本质是“排序信息”**

* 不关心风险大小
* 不表达不确定性
* 只表达：**谁更好，谁更差**

📌 典型用途：

* 横截面选股
* Top-K / Bottom-K
* 作为最基础的 baseline

---

## A.1 Predictor：LinearScorePredictor（OLS）

### 算法概括说明（新增）

**线性回归 Predictor 的角色：**

* 假设 Feature 与未来收益存在线性关系
* 用最简单、可解释的方式估计这种关系
* 输出一个连续的 score

优点：

* 稳定
* 可解释
* 是所有复杂模型的对照基线

缺点：

* 只能建模线性关系
* 不提供不确定性信息

（代码保持不变）

---

## A.2 Decision：ThresholdDecision（规则决策）

### 算法概括说明（新增）

**阈值决策的语义：**

* 把 Alpha 当成“方向信号”
* 不关心强弱，只关心正负
* 行为是离散的（多 / 空 / 不动）

优点：

* 非常直观
* 易于调试
* 适合验证 Alpha 是否“真的有信息”

缺点：

* 换手率高
* 风险控制粗糙

---

# B. Alpha 形式 2：(μ, σ) Alpha（推荐默认）

## B.0 这一类 Alpha 在做什么？（新增概括）

> **(μ,σ) Alpha 表达的是：
> “我预计能赚多少，以及我有多不确定”**

这是**从“预测”走向“可决策信息”**的关键一步。

---

## B.1 Predictor：EnsembleMuSigmaPredictor

### 算法概括说明（新增）

**Ensemble / Bootstrap Predictor 的角色：**

* 通过多次重采样或多模型，近似预测不确定性
* `mu`：平均预测 → 期望收益
* `sigma`：预测分散程度 → 模型不确定性 / 风险 proxy

优点：

* 不依赖严格分布假设
* 工程实现简单
* 与实际决策需求高度契合

缺点：

* 计算成本高于单模型
* σ 是“模型不确定性”的近似，而非真实市场波动

---

## B.2 Decision：RiskAdjustedDecision

### 算法概括说明（新增）

**风险调整决策的语义：**

* 不只看“赚多少”，也看“风险多大”
* 本质接近：**信息比率 / Sharpe 风格决策**
* 自动实现：

  * 高风险 → 降仓
  * 低风险 → 加仓

优点：

* 决策平滑
* 换手率自然受控
* 非常适合实盘框架

缺点：

* 风险度量依赖 Predictor 质量
* 仍然是规则型，不具备长期规划能力

---

# C. Alpha 形式 3：Distribution / Quantile Alpha

## C.0 这一类 Alpha 在做什么？（新增概括）

> **分布型 Alpha 表达的是：
> “未来收益可能长什么样，而不只是均值”**

它的重点在于：

* 尾部风险
* 不对称性
* 极端情况

---

## C.1 Predictor：QuantileDistPredictor

### 算法概括说明（新增）

**Quantile Predictor 的角色：**

* 不追求精确的概率密度
* 用分位数刻画“收益形状”
* 是工程上常用的“分布近似”

优点：

* 对异常值鲁棒
* 更容易反映 downside risk
* 非常适合风控与 RL state

缺点：

* 需要更多样本
* 表达复杂度更高

---

## C.2 Decision：TailRiskDecision

### 算法概括说明（新增）

**尾部风险决策的语义：**

* 决策重点放在“最坏情况下会发生什么”
* 不是最大化期望，而是控制下行风险
* 非常接近实盘中的风险管理思想

优点：

* 对极端行情更稳健
* 能自然抑制“假高收益”信号

缺点：

* 在单边强趋势中可能偏保守

---

## 4. 三种 Alpha 在系统中的“层级关系总结”（新增）

你可以这样理解它们的演进关系：

```
Score Alpha
   ↓（加入不确定性）
(μ, σ) Alpha
   ↓（加入形状/尾部）
Distribution / Quantile Alpha
```

它们**不是互斥的**，而是：

> **信息表达能力逐级增强**

---

## 5. Feature、Predictor、Decision 的一句话总结（新增收束）

* **Feature**：

  > 我们能“看到”的市场信息

* **Predictor**：

  > 我们如何把信息转化为“对未来有用的结构”

* **AlphaOutput**：

  > 预测信息的标准接口

* **Decision**：

  > 在现实约束下，如何使用这些信息

---

## 6. 你现在最正确的默认选择（实践建议）

如果你现在要**落地系统**：

1. Feature：

   * TS + Alpha101 + 简单横截面变换
2. Predictor：

   * Ensemble → 输出 (μ,σ)
3. Decision：

   * RiskAdjustedDecision（规则）
4. 再往后：

   * 用 RL 替换 Decision（不动 Predictor）

---

### 最后一句（给你定心丸）

> 你现在的理解和提问方式，已经是
> **“在设计一个量化研究系统”，而不是“在学模型”**。

如果你愿意，下一步我可以在这份讲义的基础上，
**直接帮你把它拆成“课程 PPT 结构”或“代码工程目录 + TODO 注释版”**，不改逻辑，只做形式落地。
