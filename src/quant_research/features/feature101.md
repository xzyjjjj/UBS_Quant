## 算子速查表（Operators Cheat Sheet）

> 说明：下表以 WorldQuant 101 Alpha 的常见约定来解释；不同实现对 `rank/scale/Ts_ArgMax` 的归一化区间与索引起点可能略有差异，以本项目代码实现为准。

| 算子/字段 | 含义 | 维度 | 备注 |
|---|---|---|---|
| `open/high/low/close` | OHLC 价格 | 时间序列（个股） | 通常为复权价或原始价（取决于数据源） |
| `vwap` | 成交量加权均价 | 时间序列（个股） | 常用于衡量价格偏离“交易重心” |
| `volume` | 成交量 | 时间序列（个股） | 可能是股数或手数（取决于数据源） |
| `returns` | 日收益 | 时间序列（个股） | 常见为 `close/prev_close - 1` 或对数收益（取决于数据源） |
| `cap` | 市值/流通市值 | 时间序列（个股） | 依数据源字段定义（部分 alpha 会用到） |
| `advN`（如 `adv20/adv40/adv81`） | N 日平均成交量 | 时间序列（个股） | 常见为 `sum(volume, N) / N` |
| `rank(x)` | 横截面排名/分位 | 横截面（每日） | 常见做成 `[0,1]` 分位或标准化排名 |
| `scale(x)` | 横截面缩放 | 横截面（每日） | 常见为 `x / sum(abs(x))` 使总绝对权重为 1 |
| `ts_rank(x, n)` / `Ts_Rank(x, n)` | 时间序列排名 | 时间序列（个股） | 当前值在最近 n 日窗口内的分位/名次 |
| `correlation(x, y, n)` | n 日滚动相关系数 | 时间序列（个股） | 若先做 `rank(...)`，表示先横截面标准化再算相关 |
| `covariance(x, y, n)` | n 日滚动协方差 | 时间序列（个股） | 同上，输入可为原值或排名序列 |
| `stddev(x, n)` | n 日滚动标准差 | 时间序列（个股） | 波动率刻画 |
| `sum(x, n)` | n 日滚动求和 | 时间序列（个股） | |
| `product(x, n)` | n 日滚动连乘 | 时间序列（个股） | 常用于复合增长/复合因子 |
| `delay(x, d)` | 滞后 d 期 | 时间序列（个股） | 返回 `x[t-d]` |
| `delta(x, d)` | 差分 | 时间序列（个股） | 常见为 `x[t] - x[t-d]` |
| `ts_min(x, n)` / `ts_max(x, n)` | n 日滚动最小/最大 | 时间序列（个股） | |
| `Ts_ArgMax(x, n)` / `Ts_ArgMin(x, n)` | 极值位置 | 时间序列（个股） | 返回最近 n 日内最大/最小值出现的相对位置（索引起点依实现而定） |
| `ts_argmax(x, n)` / `ts_argmin(x, n)` | 极值位置（别名） | 时间序列（个股） | 常见与 `Ts_ArgMax/Ts_ArgMin` 等价 |
| `SignedPower(x, p)` | 带符号幂 | 通用 | 常见为 `sign(x) * abs(x)^p` |
| `abs(x)` / `sign(x)` / `Sign(x)` | 绝对值 / 符号函数 | 通用 | `sign/Sign` 常返回 `-1/0/+1` |
| `log(x)` / `Log(x)` | 对数 | 通用 | 通常为自然对数 `ln` |
| `decay_linear(x, n)` | 线性衰减加权 | 时间序列（个股） | 最近数据权重更大，常用于“记忆”机制 |
| `IndNeutralize(x, IndClass.xxx)` | 行业/板块中性化 | 横截面（每日） | 常见为按行业分组去均值或回归剔除行业暴露 |
| `(cond) ? a : b` | 三元条件表达式 | 通用 | 条件为真取 `a`，否则取 `b` |
| `&&` / `||` | 逻辑与 / 或 | 通用 | 用于组合条件 |
| `min(a,b)` / `max(a,b)` / `^` | 最小/最大/幂运算 | 通用 | `^` 表示幂（实现可能映射到 `pow`） |

---

## Alpha 1–10

### Alpha#1

```text
rank(
  Ts_ArgMax(
    SignedPower(
      (returns < 0) ? stddev(returns, 20) : close,
      2.0
    ),
    5
  )
) - 0.5
```

解释：当日若收益为负则用过去 20 日收益波动率（更像“风险/恐慌”刻画），否则用收盘价；对该序列做平方放大极端值，再在 5 日窗口里取最大值出现的位置（ArgMax），最后做横截面排名并居中（减 0.5）。整体偏向“极端/压力”时的相对强弱信号。

### Alpha#2

```text
-1 * correlation(
  rank(delta(log(volume), 2)),
  rank((close - open) / open),
  6
)
```

解释：比较“成交量对数的 2 日变化”（近似量能加速）与“日内涨跌幅（收盘-开盘）/开盘”的 6 日相关性，并取负号。若某票近期量能变化与日内涨跌高度同向相关，则给出更负的打分（更偏反向/回撤思路）。

### Alpha#3

```text
-1 * correlation(
  rank(open),
  rank(volume),
  10
)
```

解释：在每一天先对开盘价、成交量做横截面排名，再看两者在 10 日窗口内的相关性，并取负号。直觉上是在惩罚“高开同时高量/低开同时低量”这种稳定同向关系，偏向对价格-量能耦合做反向暴露。

### Alpha#4

```text
-1 * Ts_Rank(
  rank(low),
  9
)
```

解释：每天对最低价做横截面排名，然后对该排名序列做 9 日时间序列排名（看它在自身近 9 日中的相对位置），最后取负号。通常可理解为：更偏向押注“近期 low 的相对位置”出现均值回归/反转。

### Alpha#5

```text
rank(open - (sum(vwap, 10) / 10)) *
(-1 * abs(rank(close - vwap)))
```

解释：第一部分衡量“开盘价相对过去 10 日 VWAP 均值”的强弱并做横截面排名；第二部分衡量“收盘相对当日 VWAP 的偏离”并取其横截面排名的绝对值，再乘以 -1。组合起来更像是在：在开盘偏离中寻找机会，但对当日收盘偏离 VWAP 越极端的标的施加惩罚（整体更负）。

### Alpha#6

```text
-1 * correlation(open, volume, 10)
```

解释：直接计算开盘价与成交量在 10 日窗口内的相关性并取负号（未做 rank）。如果某标的“高开=高量”关系越稳定，则该 alpha 越负；反之则越正，偏向对价格-量能线性关系做反向配置。

### Alpha#7

```text
(adv20 < volume)
  ? (
      (-1 * ts_rank(abs(delta(close, 7)), 60)) *
      sign(delta(close, 7))
    )
  : (-1 * 1)
```

解释：当日成交量高于 20 日均量（adv20）时，取 7 日价格变动的方向（sign），并用“7 日变动幅度在过去 60 日的时间序列排名”来刻画这次波动是否极端，然后整体取负号；否则返回常数 -1。直觉上是对“放量+较大 7 日波动”的方向进行反向暴露，未放量时直接给一个固定偏空打分。

### Alpha#8

```text
-1 * rank(
  (sum(open, 5) * sum(returns, 5)) -
  delay((sum(open, 5) * sum(returns, 5)), 10)
)
```

解释：构造一个 5 日复合量：5 日开盘价之和 × 5 日收益之和，然后与其 10 日前的值做差并做横截面排名，最后取负号。更像是对“近期复合趋势相对过去的变化”做反向排序，偏短周期均值回归。

### Alpha#9

```text
(0 < ts_min(delta(close, 1), 5))
  ? delta(close, 1)
  : (
      (ts_max(delta(close, 1), 5) < 0)
        ? delta(close, 1)
        : (-1 * delta(close, 1))
    )
```

解释：看过去 5 天的日度涨跌（delta(close,1)）。如果 5 天里每天都上涨（最小值>0）或每天都下跌（最大值<0），则保持当日涨跌方向；否则（涨跌夹杂）就把当日涨跌取反。直觉上是“趋势一致时跟随、震荡时反转”的开关型规则。

### Alpha#10

```text
rank(
  (0 < ts_min(delta(close, 1), 4))
    ? delta(close, 1)
    : (
        (ts_max(delta(close, 1), 4) < 0)
          ? delta(close, 1)
          : (-1 * delta(close, 1))
      )
)
```

解释：与 Alpha#9 类似，但窗口为 4 日，并对最终结果做横截面 rank。相当于把“趋势/震荡开关”的输出标准化到横截面相对强弱，便于与其它 alpha 组合或做多空截面。

---

## Alpha 11–20

### Alpha#11

```text
(
  rank(ts_max(vwap - close, 3)) +
  rank(ts_min(vwap - close, 3))
) * rank(delta(volume, 3))
```

解释：用 3 日内 (vwap-close) 的最大/最小值衡量“价格相对 VWAP 的极端偏离”，分别做横截面 rank 后相加；再乘以 3 日成交量变化的横截面 rank。整体强调“偏离程度 × 量能变化”的联动，常用于捕捉短期偏离在量能配合下的反转或延续。

### Alpha#12

```text
sign(delta(volume, 1)) * (-1 * delta(close, 1))
```

解释：先看成交量较昨日是增是减（符号），再乘以“价格日变动的反向”。当量能放大（sign 为正）时更偏向做与当日涨跌相反的暴露；量能萎缩时则反过来，属于很直接的“量变条件下的反转/顺势”开关。

### Alpha#13

```text
-1 * rank(
  covariance(
    rank(close),
    rank(volume),
    5
  )
)
```

解释：对 close 与 volume 先做横截面 rank，再在 5 日窗口内算两者协方差，最后再做横截面 rank 并取负号。可理解为惩罚“价格排名与成交量排名在短期内一起上/下一起下”的一致性（偏反向）。

### Alpha#14

```text
(-1 * rank(delta(returns, 3))) *
correlation(open, volume, 10)
```

解释：用 returns 的 3 日变化做横截面 rank 并取负号（对“收益加速”偏反向），再乘以开盘价与成交量 10 日相关性（未 rank）。等于是用“价量相关强弱”去放大/缩小对收益变化的反向押注。

### Alpha#15

```text
-1 * sum(
  rank(
    correlation(
      rank(high),
      rank(volume),
      3
    )
  ),
  3
)
```

解释：先计算“high 的横截面 rank”与“volume 的横截面 rank”在 3 日窗口内的相关性，并对相关性做横截面 rank；再把这个序列在 3 日内求和，最后取负号。整体是在惩罚“高价位与高量能短期强耦合”这种模式的持续性。

### Alpha#16

```text
-1 * rank(
  covariance(
    rank(high),
    rank(volume),
    5
  )
)
```

解释：与 Alpha#13 类似，但把 close 换成 high。强调“高点排名”和“量能排名”在 5 日内的协方差，并取负号，偏向对“冲高放量/不放量”的结构做反向配置。

### Alpha#17

```text
(
  (-1 * rank(ts_rank(close, 10))) *
  rank(delta(delta(close, 1), 1)) *
  rank(ts_rank(volume / adv20, 5))
)
```

解释：三部分相乘：① close 在自身 10 日窗口内的时间序列排名（越靠近近期低位则 rank 越小，乘 -1 后偏正）；② 价格一阶差分的再差分（近似“加速度/反转力度”）并做横截面 rank；③ 相对均量的成交量（volume/adv20）的 5 日时间序列排名并做横截面 rank。整体更像是“低位 + 加速度变化 + 放量”共同出现时更显著的复合信号。

### Alpha#18

```text
-1 * rank(
  stddev(abs(close - open), 5) +
  (close - open) +
  correlation(close, open, 10)
)
```

解释：把三个成分加总后再做横截面 rank 并取负号：① 日内振幅 abs(close-open) 的 5 日波动率（不稳定/噪声刻画）；② 日内方向 close-open；③ close 与 open 的 10 日相关性。整体是在用“日内波动结构 + 日内方向 + 价序列一致性”做综合打分，并反向排序。

### Alpha#19

```text
(-1 * sign((close - delay(close, 7)) + delta(close, 7))) *
(1 + rank(1 + sum(returns, 250)))
```

解释：第一项本质上取 7 日价格动量的符号并取负（对短期动量做反向），第二项用近 250 日累计收益（再 +1）做横截面 rank 后加 1，作为幅度放大因子。直觉上：对短期涨/跌做反向押注，但对长期表现更“强势/弱势”的标的给予更大权重。

### Alpha#20

```text
(
  (-1 * rank(open - delay(high, 1))) *
  rank(open - delay(close, 1)) *
  rank(open - delay(low, 1))
)
```

解释：比较当日开盘价相对昨日高/收/低的“跳空位置”，分别做横截面 rank，并通过对 (open - yesterday high) 的 rank 取负来引入方向性。整体更像是在刻画开盘相对昨日区间的位置结构（偏上沿/中部/下沿），并用乘积让“多维一致的跳空形态”更突出。

---

## Alpha 21–40（中段核心区）

### Alpha#21

```text
(
  ((sum(close, 8) / 8) + stddev(close, 8)) < (sum(close, 2) / 2)
)
  ? (-1 * 1)
  : (
      (sum(close, 2) / 2) < ((sum(close, 8) / 8) - stddev(close, 8))
    )
      ? 1
      : (
          (1 < (volume / adv20)) || ((volume / adv20) == 1)
        )
          ? 1
          : (-1 * 1)
```

解释：用 8 日均值±8 日波动构造一个“带宽”，与 2 日均值比较做分段判断；若价格相对短期均值偏弱则给 -1，若偏强则给 +1，否则再用量能相对均量（volume/adv20）是否≥1 来决定输出。整体是“价格偏离带宽 + 放量确认”的离散信号。

### Alpha#22

```text
-1 * (
  delta(correlation(high, volume, 5), 5) *
  rank(stddev(close, 20))
)
```

解释：看“高价与成交量 5 日相关性”的 5 日变化（相关性是否在上升/下降），再乘以收盘价 20 日波动的横截面 rank，并取负号。更像是：在波动更高的标的上，对价量相关性变化进行反向暴露。

### Alpha#23

```text
((sum(high, 20) / 20) < high)
  ? (-1 * delta(high, 2))
  : 0
```

解释：当日 high 高于其 20 日均值（偏“冲高”）时，输出 2 日 high 变动的反向；否则输出 0。可理解为：只在“高点偏热”时启用的短期反转信号。

### Alpha#24

```text
(
  delta((sum(close, 100) / 100), 100) / delay(close, 100)
  <= 0.05
)
  ? (-1 * (close - ts_min(close, 100)))
  : (-1 * delta(close, 3))
```

解释：先用 100 日均价的 100 日变化（再除以 100 日前的 close）判断长期走势是否“温和”（≤5%）；若温和，则用“当前 close 距离 100 日最低点的距离”取负（越靠近低点越不负/更可能偏正）；否则用 3 日变动取负。属于“长期状态切换：温和期看位置，非温和期看短期反转”。

### Alpha#25

```text
rank(
  (((-1 * returns) * adv20) * vwap) *
  (high - close)
)
```

解释：把当日收益取负（偏向反转），乘以均量 adv20 与 vwap（偏向放大量能/高价区间的影响），再乘以 (high-close)（收盘距离当日高点的“回落幅度”），最后做横截面 rank。直觉上更偏好“涨得越少/跌得越多 + 放量 + 从高点回落”的标的。

### Alpha#26

```text
-1 * ts_max(
  correlation(
    ts_rank(volume, 5),
    ts_rank(high, 5),
    5
  ),
  3
)
```

解释：对 volume、high 各自做 5 日时间序列排名，再算两者 5 日相关性；在此基础上取过去 3 天的最大值，并取负号。等于是惩罚“短期内量与高点同步性最强”的时刻，偏反向。

### Alpha#27

```text
(0.5 < rank(sum(correlation(rank(volume), rank(vwap), 6), 2) / 2.0))
  ? (-1 * 1)
  : 1
```

解释：计算“volume 与 vwap 的横截面 rank”在 6 日窗口的相关性，再对相关性做 2 日均值；若该值在横截面上处于较高分位（>0.5），则输出 -1，否则输出 +1。属于一个非常离散的“价量耦合强则反向，弱则正向”的开关。

### Alpha#28

```text
scale(
  correlation(adv20, low, 5) +
  ((high + low) / 2) -
  close
)
```

解释：把三个量相加后做横截面 scale：① adv20 与 low 的 5 日相关性（量能与低点的联动）；② 当日中间价 (high+low)/2；③ 减去 close（收盘相对中位的偏离）。直觉上在刻画“收盘相对当日区间位置 + 量能与低点关系”的综合偏离，并用 scale 做权重归一。

### Alpha#29

```text
min(
  product(
    rank(
      rank(
        scale(
          log(
            sum(
              ts_min(
                rank(
                  rank(
                    (-1 * rank(delta((close - 1), 5)))
                  )
                ),
                2
              ),
              1
            )
          )
        )
      )
    ),
    1
  ),
  5
) + ts_rank(delay((-1 * returns), 6), 5)
```

解释：一个非常“工程化”的复合项：对若干层 rank/ts_min/sum/log/scale 再做 product 与截断（`min(..., 5)`），然后加上“负收益滞后 6 期的 5 日时间序列排名”。整体倾向于把若干非线性变换后的“反转/压力”特征压到可控区间，再叠加一个滞后反转项。

### Alpha#30

```text
(
  (1.0 - rank(
    sign(close - delay(close, 1)) +
    sign(delay(close, 1) - delay(close, 2)) +
    sign(delay(close, 2) - delay(close, 3))
  )) *
  sum(volume, 5)
) / sum(volume, 20)
```

解释：用过去 3 段日涨跌的符号相加，代表短期“连续性/单边性”，再做横截面 rank 并用 (1-rank) 反向；乘以 5 日成交量，再除以 20 日成交量（量能占比）。直觉上是：更偏好“短期不那么单边（更像震荡）但近期量能占比更高”的标的。

### Alpha#31

```text
rank(
  rank(
    rank(
      decay_linear(
        -1 * rank(rank(delta(close, 10))),
        10
      )
    )
  )
) +
rank(-1 * delta(close, 3)) +
sign(scale(correlation(adv20, low, 12)))
```

解释：三部分相加：① 对 close 的 10 日变化做多层 rank + 衰减，强调变化的相对位置；② 对 3 日变化取负后再 rank，偏短期反转；③ adv20 与 low 的相关性做 scale 后取符号，作为一个离散方向项。整体是“变动结构 + 反转项 + 价量/低点联动方向”的组合。

### Alpha#32

```text
scale((sum(close, 7) / 7) - close) +
20 * scale(correlation(vwap, delay(close, 5), 230))
```

解释：第一项是收盘价相对 7 日均值的偏离并做 scale（偏均值回归）；第二项是 VWAP 与 5 日滞后 close 的超长窗口相关性并做 scale，再乘 20 放大。整体把“短期偏离”与“长期价序列关系”叠加。

### Alpha#33

```text
rank(-1 * ((1 - (open / close)) ^ 1))
```

解释：本质上是对 `1 - open/close` 取负并做横截面 rank（`^1` 不改变数值）。直观等价于对“开盘相对收盘的贴水/溢价”做排序，方向为反向。

### Alpha#34

```text
rank(
  (1 - rank(stddev(returns, 2) / stddev(returns, 5))) +
  (1 - rank(delta(close, 1)))
)
```

解释：把两个“越小越好”的量加总后再 rank：① 短期/中期波动比（2 日/5 日）越小越偏好；② 当日涨跌（delta）越小越偏好（更偏向回撤/反转）。整体偏向“短期波动收敛 + 价格走弱”的组合。

### Alpha#35

```text
Ts_Rank(volume, 32) *
(1 - Ts_Rank((close + high) - low, 16)) *
(1 - Ts_Rank(returns, 32))
```

解释：用三个时间序列分位相乘：① 近期量能在自身 32 日里的位置；② 日内区间位置量 `(close+high-low)` 的 16 日分位取反；③ 收益在 32 日分位取反。强调“量能相对更高，但价格/收益分位更靠低端”的状态（偏反转/防守）。

### Alpha#36

```text
(2.21 * rank(correlation(close - open, delay(volume, 1), 15))) +
(0.7 * rank(open - close)) +
(0.73 * rank(Ts_Rank(delay(-1 * returns, 6), 5))) +
rank(abs(correlation(vwap, adv20, 6))) +
(0.6 * rank(((sum(close, 200) / 200) - open) * (close - open)))
```

解释：一个加权线性组合：包含“日内涨跌与滞后量能的相关性”“日内方向”“滞后反转项的时间序列分位”“VWAP 与均量相关性的强度”“长期均价与开盘、日内方向的交互项”。属于典型的多特征集成信号。

### Alpha#37

```text
rank(correlation(delay(open - close, 1), close, 200)) +
rank(open - close)
```

解释：把“昨日日内方向 (open-close) 与 close 的长期相关性”做 rank，再加上当日 (open-close) 的 rank。整体在用一个长期关系去刻画当日形态的相对意义。

### Alpha#38

```text
(-1 * rank(Ts_Rank(close, 10))) *
rank(close / open)
```

解释：第一项偏向“价格处于自身近 10 日较低分位”（取负后偏正），第二项刻画当日从开到收的相对强弱（close/open）。乘积强调“低位背景下的日内走强”。

### Alpha#39

```text
(-1 * rank(
  delta(close, 7) * (1 - rank(decay_linear(volume / adv20, 9)))
)) * (1 + rank(sum(returns, 250)))
```

解释：用 7 日价格变化与“相对均量的衰减 rank（取反）”做交互，再整体取负并 rank；同时用 250 日累计收益 rank 做幅度放大。直觉上是：在长期强弱分层下，对“近期变动×量能结构”的反向排序。

### Alpha#40

```text
(-1 * rank(stddev(high, 10))) *
correlation(high, volume, 10)
```

解释：把 high 的 10 日波动（stddev）做 rank 后取负，再乘以 high 与 volume 的 10 日相关性。更偏向在“高点波动更小”的标的上，根据价量相关性进行加权暴露。

---

## Alpha 41–60（含经典 Alpha#42）

### Alpha#41

```text
((high * low) ^ 0.5) - vwap
```

解释：用几何均值 `sqrt(high*low)` 作为当日“价格中枢”的近似，与 vwap 做差。若 vwap 高于几何均值则结果偏负，反之偏正，刻画“成交重心 vs 当日区间中枢”的偏离。

### Alpha#42

```text
rank(vwap - close) / rank(vwap + close)
```

解释：分子衡量 close 相对 vwap 的偏离（vwap-close），分母用 (vwap+close) 做一个尺度归一（均做横截面 rank）。直觉上是一个“相对偏离 / 相对价格水平”的比值，常被视为经典的“收盘相对成交重心偏离”信号。

### Alpha#43

```text
ts_rank(volume / adv20, 20) *
ts_rank(-1 * delta(close, 7), 8)
```

解释：第一项是相对均量的 20 日时间序列排名（近期是否放量）；第二项是 7 日价格变动取负后的 8 日时间序列排名（近期 7 日回撤/反转的相对位置）。乘积强调“放量背景下的短期回撤/反转”。

### Alpha#44

```text
-1 * correlation(
  high,
  rank(volume),
  5
)
```

解释：计算 high 与成交量横截面 rank 在 5 日窗口的相关性并取负号。若某标的“高点抬升伴随量能更强”的关系越稳定，则 alpha 越负（偏反向配置）。

### Alpha#45

```text
-1 * (
  (
    rank(sum(delay(close, 5), 20) / 20) *
    correlation(close, volume, 2)
  ) *
  rank(correlation(sum(close, 5), sum(close, 20), 2))
)
```

解释：三块相乘后取负：① 20 日均值的“5 日滞后 close”并做横截面 rank（偏慢变量/位置）；② close 与 volume 的 2 日相关性（很短的价量联动）；③ 5 日与 20 日累计 close 的 2 日相关性再做 rank（短长周期联动）。整体是一个复合的“价量短期耦合 × 价格短长周期同步”的反向暴露。

### Alpha#46

```text
(
  0.25 <
  (
    ((delay(close, 20) - delay(close, 10)) / 10) -
    ((delay(close, 10) - close) / 10)
  )
)
  ? (-1 * 1)
  : (
      (
        ((delay(close, 20) - delay(close, 10)) / 10) -
        ((delay(close, 10) - close) / 10)
      ) < 0
    )
      ? 1
      : ((-1 * 1) * (close - delay(close, 1)))
```

解释：比较两个“10 日斜率”：20→10 的斜率减去 10→当前的斜率（类似动量的加速度）。若加速度显著为正（>0.25）则输出 -1；若加速度为负则输出 +1；否则输出“当日涨跌”的反向。属于一个用趋势加速度做状态切换的离散/半离散规则。

### Alpha#47

```text
(
  (
    (rank(1 / close) * volume) / adv20
  ) *
  (
    (high * rank(high - close)) /
    (sum(high, 5) / 5)
  )
) - rank(vwap - delay(vwap, 5))
```

解释：第一大项把“低价（1/close）排名 × 成交量 / 均量”作为放量低价权重；第二大项衡量 high 相对其 5 日均值，并乘以 (high-close) 的横截面 rank（收盘离高点的回落）。最后减去 vwap 的 5 日变化（vwap - delay(vwap,5)）的横截面 rank。整体是“放量低价 + 当日从高点回落结构”与“VWAP 趋势”之间的差值型信号。

### Alpha#48

```text
indneutralize(
  (
    correlation(
      delta(close, 1),
      delta(delay(close, 1), 1),
      250
    ) * delta(close, 1)
  ) / close,
  IndClass.subindustry
) / sum(((delta(close, 1) / delay(close, 1)) ^ 2), 250)
```

解释：分子是“日变动与其 1 日滞后变动”的 250 日相关性 × 当日变动，再除以 close 做尺度归一，并对次行业做中性化；分母是 250 日内（近似）平方收益的累计（波动规模）。整体可理解为：在剔除行业后，用“变化的自相关结构（类似惯性/反转）”按波动归一，偏向风险调整后的结构性信号。

### Alpha#49

```text
(
  (
    ((delay(close, 20) - delay(close, 10)) / 10) -
    ((delay(close, 10) - close) / 10)
  ) < (-0.1)
)
  ? 1
  : ((-1 * 1) * (close - delay(close, 1)))
```

解释：仍是“两个 10 日斜率之差”（趋势加速度），若显著为负（<-0.1）则输出 +1；否则输出当日涨跌的反向。相比 Alpha#46，这是一个更简化的“加速度阈值 + 否则反转”规则。

### Alpha#50

```text
-1 * ts_max(
  rank(correlation(rank(volume), rank(vwap), 5)),
  5
)
```

解释：先计算 volume 与 vwap 的横截面 rank 序列在 5 日窗口的相关性，并对该相关性做横截面 rank；再取过去 5 天的最大值并取负号。等于惩罚“近期出现过最强的价量耦合”的标的（偏反向）。

### Alpha#51

```text
(
  (
    ((delay(close, 20) - delay(close, 10)) / 10) -
    ((delay(close, 10) - close) / 10)
  ) < (-1 * 0.05)
)
  ? 1
  : ((-1 * 1) * (close - delay(close, 1)))
```

解释：与 Alpha#49 类似的“趋势加速度”阈值规则，但阈值改为 -0.05：当加速度显著为负时直接输出 +1，否则输出当日涨跌的反向。属于“下行加速时偏多，否则做日内反转”的离散规则。

### Alpha#52

```text
(
  (
    (-1 * ts_min(low, 5)) +
    delay(ts_min(low, 5), 5)
  ) *
  rank((sum(returns, 240) - sum(returns, 20)) / 220)
) * ts_rank(volume, 5)
```

解释：用“低点（5 日最低）相对 5 日前的变化”（并取负）作为价格结构项，乘以“长期（240）与短期（20）累计收益差”的横截面 rank（近似长期趋势强弱），再乘以成交量的 5 日时间序列分位。整体强调“低点结构变化 × 长短期趋势分层 × 量能状态”。

### Alpha#53

```text
-1 * delta(
  (((close - low) - (high - close)) / (close - low)),
  9
)
```

解释：括号内是把 close 在当日区间中的相对位置做成一个“偏上/偏下”指标（分母用 close-low 缩放），再取 9 日差分并取负。直觉上是在对“位置指标的变化”做反向（偏均值回归）。

### Alpha#54

```text
(-1 * ((low - close) * (open ^ 5))) /
((low - high) * (close ^ 5))
```

解释：一个强非线性比例项：把 (low-close) 与 open^5、close^5 等组合起来，并整体取负。通常用于放大“收盘相对低点的位置”在不同价格水平下的差异（幂次会显著放大高价/低价差异）。

### Alpha#55

```text
-1 * correlation(
  rank((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12))),
  rank(volume),
  6
)
```

解释：先把 close 映射到近 12 日高低区间的归一化位置（再做横截面 rank），与成交量的横截面 rank 做 6 日相关并取负。若“区间位置与量能”短期高度同向，则给更负分（偏反向）。

### Alpha#56

```text
0 - (
  1 * (
    rank(sum(returns, 10) / sum(sum(returns, 2), 3)) *
    rank(returns * cap)
  )
)
```

解释：整体取负。第一项是 10 日累计收益相对“2 日收益的 3 期累计”的比值（一种收益聚合/加速刻画），第二项是 returns 与市值 cap 的乘积再 rank（将收益与规模耦合）。乘积后取负，偏向对“收益结构×规模耦合”进行反向排序。注：需要数据里有 `cap` 字段。

### Alpha#57

```text
0 - (
  1 * (
    (close - vwap) /
    decay_linear(rank(ts_argmax(close, 30)), 2)
  )
)
```

解释：用 close-vwap（收盘相对 VWAP 的偏离）除以“close 在 30 日窗口内最大值出现位置”的衰减权重（再 rank），整体取负。直觉上：偏离在“近期极值更靠近现在/更久远”的情况下被不同程度放大/缩小，并做反向。

### Alpha#58

```text
-1 * Ts_Rank(
  decay_linear(
    correlation(IndNeutralize(vwap, IndClass.sector), volume, 3.92795),
    7.89291
  ),
  5.50322
)
```

解释：对（行业中性化后的）vwap 与 volume 的短窗口相关性做线性衰减，再做时间序列排名，并取负。强调“行业剔除后价量相关结构”的相对位置（偏反向）。

### Alpha#59

```text
-1 * Ts_Rank(
  decay_linear(
    correlation(
      IndNeutralize((vwap * 0.728317) + (vwap * (1 - 0.728317)), IndClass.industry),
      volume,
      4.25197
    ),
    16.2289
  ),
  8.19648
)
```

解释：与 Alpha#58 类似，但改为行业维度中性化，窗口与衰减/排名参数不同。整体仍是对“行业中性价量相关”的时间序列位置取负，偏反向。

### Alpha#60

```text
0 - (
  1 * (
    (2 * scale(rank((((close - low) - (high - close)) / (high - low)) * volume))) -
    scale(rank(ts_argmax(close, 10)))
  )
)
```

解释：整体取负。第一项把“收盘在当日区间的位置”乘以 volume 后做 rank 再 scale，并乘 2 放大；第二项是 close 在 10 日窗口内最大值出现位置（ArgMax）做 rank 再 scale。二者做差后取负，属于“位置×量能”与“极值时序结构”的对比型信号。

---

## Alpha 61–80（高阶 + 行业中性）

### Alpha#61

```text
rank(vwap - ts_min(vwap, 16.1219)) <
rank(correlation(vwap, adv180, 17.9282))
```

解释：把“VWAP 相对其近 16 天最低值的偏离”做横截面 rank，与“VWAP 和长期均量 adv180 的滚动相关性”做横截面 rank 进行比较。结果是一个布尔条件（真/假，通常映射为 1/0），用来判断“价格相对低位的程度”是否小于“价量长期相关强度”。

### Alpha#62

```text
(
  rank(correlation(vwap, sum(adv20, 22.4101), 9.91009)) <
  rank(
    (rank(open) + rank(open)) <
    (rank((high + low) / 2) + rank(high))
  )
) * -1
```

解释：比较两个横截面 rank：左边是 VWAP 与一段窗口内 adv20 累计的相关性；右边是一个由价格位置关系构造出来的布尔条件（“开盘 rank 的两倍”是否小于“中价 rank + 最高价 rank”），再对该布尔结果做 rank。若左边小于右边，则输出 -1，否则输出 0（典型实现）；属于离散的“条件触发型”因子。

### Alpha#63

```text
(
  rank(
    decay_linear(
      delta(IndNeutralize(close, IndClass.industry), 2.25164),
      8.22237
    )
  ) -
  rank(
    decay_linear(
      correlation(
        (vwap * 0.318108) + (open * (1 - 0.318108)),
        sum(adv180, 37.2467),
        13.557
      ),
      12.2883
    )
  )
) * -1
```

解释：两部分做差后取负：① 行业中性化后的 close 做一段差分并线性衰减加权，再做横截面 rank；② 将 vwap/open 做加权混合后，与长期量能累计做相关，再衰减加权并做 rank。整体是在对“行业中性价格变化”与“价量相关结构”之间的相对强弱做反向排序。

### Alpha#64

```text
(
  rank(
    correlation(
      sum((open * 0.178404) + (low * (1 - 0.178404)), 12.7054),
      sum(adv120, 12.7054),
      16.6208
    )
  ) <
  rank(
    delta(
      ((high + low) / 2 * 0.178404) + (vwap * (1 - 0.178404)),
      3.69741
    )
  )
) * -1
```

解释：左边是“加权(open/low) 的滚动和”与“adv120 的滚动和”之间的相关性并做 rank；右边是“加权(mid/vwap) 的差分”并做 rank。若左边小于右边则输出 -1，否则 0。直觉上是在用“量能与偏低价位的联动弱/强”去对比“价格变化强弱”，并做离散触发。

### Alpha#65

```text
(
  rank(
    correlation(
      (open * 0.00817205) + (vwap * (1 - 0.00817205)),
      sum(adv60, 8.6911),
      6.40374
    )
  ) <
  rank(open - ts_min(open, 13.635))
) * -1
```

解释：比较“(几乎等于 VWAP 的) 加权价格与 adv60 累计的相关性”与“开盘价相对近 13 天最低开盘的偏离”，两者均横截面 rank。相关性 rank 更小则输出 -1，否则 0，属于条件式因子。

### Alpha#66

```text
(
  rank(decay_linear(delta(vwap, 3.51013), 7.23052)) +
  Ts_Rank(
    decay_linear(
      (
        ((low * 0.96633) + (low * (1 - 0.96633)) - vwap) /
        (open - ((high + low) / 2))
      ),
      11.4157
    ),
    6.72611
  )
) * -1
```

解释：两项相加后取负：第一项是 VWAP 的差分再线性衰减并做横截面 rank；第二项构造一个“(low-vwap) 相对 (open-mid) 的比值”，先线性衰减再做时间序列排名。整体更像是在同时刻画“VWAP 的短期变化”与“价格在当日区间中的结构性偏离”，并做反向暴露。

### Alpha#67

```text
(
  rank(high - ts_min(high, 2.14593)) ^
  rank(
    correlation(
      IndNeutralize(vwap, IndClass.sector),
      IndNeutralize(adv20, IndClass.subindustry),
      6.02936
    )
  )
) * -1
```

解释：把“high 相对近期最低 high 的偏离”做 rank，与“行业/子行业中性化后的 vwap 与 adv20 的相关性”做 rank，再做幂运算并取负。幂运算会放大两个 rank 同时较大/较小的情形，属于非线性增强的反向信号。

### Alpha#68

```text
(
  Ts_Rank(
    correlation(rank(high), rank(adv15), 8.91644),
    13.9333
  ) <
  rank(delta((close * 0.518371) + (low * (1 - 0.518371)), 1.06157))
) * -1
```

解释：将“high 的横截面 rank”与“adv15 的横截面 rank”做滚动相关，并对相关性做时间序列排名；再与“加权(close/low) 的差分”做横截面 rank 比较。若前者小于后者输出 -1，否则 0。直觉上是用“价-量能短期耦合的相对位置”去对比“价格变化强弱”的条件触发。

### Alpha#69

```text
(
  rank(ts_max(delta(IndNeutralize(vwap, IndClass.industry), 2.72412), 4.79344)) ^
  Ts_Rank(
    correlation(
      (close * 0.490655) + (vwap * (1 - 0.490655)),
      adv20,
      4.92416
    ),
    9.0615
  )
) * -1
```

解释：第一项取行业中性化 VWAP 的差分，再在短窗口取最大值并做 rank；第二项是加权(close/vwap) 与 adv20 的相关性，再做时间序列排名。两者做幂运算后取负，强调“VWAP 的极端变化”与“价量相关性处于某种相对位置”同时出现时的非线性放大，并偏反向。

### Alpha#70

```text
(
  rank(delta(vwap, 1.29456)) ^
  Ts_Rank(
    correlation(IndNeutralize(close, IndClass.industry), adv50, 17.8256),
    17.9171
  )
) * -1
```

解释：把 VWAP 的差分做横截面 rank，与“行业中性化 close 与 adv50 的相关性”的时间序列排名做幂运算并取负。属于“价格变化强弱 × 长周期价量相关结构”的非线性组合，方向为反向。

### Alpha#71

```text
max(
  Ts_Rank(
    decay_linear(
      correlation(
        Ts_Rank(close, 3.43976),
        Ts_Rank(adv180, 12.0647),
        18.0175
      ),
      4.20501
    ),
    15.6948
  ),
  Ts_Rank(
    decay_linear(
      (rank((low + open) - (vwap + vwap)) ^ 2),
      16.4662
    ),
    4.4388
  )
)
```

解释：取两条子信号的较大者：① close 分位与长期均量分位的相关结构，经衰减后做时间序列排名；② (low+open-2*vwap) 的横截面 rank 取平方后衰减，再做时间序列排名。`max` 强调“任一条强就触发”的聚合方式。

### Alpha#72

```text
rank(
  decay_linear(
    correlation((high + low) / 2, adv40, 8.93345),
    10.1519
  )
) / rank(
  decay_linear(
    correlation(
      Ts_Rank(vwap, 3.72469),
      Ts_Rank(volume, 18.5188),
      6.86671
    ),
    2.95011
  )
)
```

解释：分子是“中价与 adv40 的相关性”经衰减后做横截面 rank；分母是“vwap 分位与 volume 分位的相关性”经衰减后做 rank。比值结构相当于用两类价量关系的相对强弱做归一化对比。

### Alpha#73

```text
max(
  rank(decay_linear(delta(vwap, 4.72775), 2.91864)),
  Ts_Rank(
    decay_linear(
      -1 * (
        delta((open * 0.147155) + (low * (1 - 0.147155)), 2.03608) /
        ((open * 0.147155) + (low * (1 - 0.147155)))
      ),
      3.33829
    ),
    16.7411
  )
) * -1
```

解释：两条子信号取 `max` 后整体取负：① VWAP 差分的衰减 rank；② 加权(open/low) 的相对变化取负后衰减，再做时间序列排名。强调“VWAP 变化”与“偏低价位变化”的任一强信号，并以反向输出。

### Alpha#74

```text
(
  rank(correlation(close, sum(adv30, 37.4843), 15.1365)) <
  rank(
    correlation(
      rank((high * 0.0261661) + (vwap * (1 - 0.0261661))),
      rank(volume),
      11.4791
    )
  )
) * -1
```

解释：比较两种相关结构的横截面 rank：close 与 adv30 累计的相关性 vs（加权(high/vwap) 的 rank）与 volume rank 的相关性。若前者更小则输出 -1，否则 0。属于离散触发型比较。

### Alpha#75

```text
rank(correlation(vwap, volume, 4.24304)) <
rank(correlation(rank(low), rank(adv50), 12.4413))
```

解释：比较“vwap 与 volume 的短期相关性”的 rank 与 “low rank 与 adv50 rank 的相关性”的 rank，输出布尔条件（真/假）。用来筛选两类价量关系哪一类更占优。

### Alpha#76

```text
max(
  rank(decay_linear(delta(vwap, 1.24383), 11.8259)),
  Ts_Rank(
    decay_linear(
      Ts_Rank(
        correlation(IndNeutralize(low, IndClass.sector), adv81, 8.14941),
        19.569
      ),
      17.1543
    ),
    19.383
  )
) * -1
```

解释：两条子信号取 `max` 后整体取负：① VWAP 差分的衰减 rank；② 板块中性化 low 与 adv81 的相关性，先做时间序列排名，再衰减后再做时间序列排名。强调“价格变化”或“中性化价量联动”任一强信号，并反向输出。

### Alpha#77

```text
min(
  rank(decay_linear((((high + low) / 2) + high) - (vwap + high), 20.0451)),
  rank(decay_linear(correlation((high + low) / 2, adv40, 3.1614), 5.64125))
)
```

解释：`min` 取两个信号中更“弱”的那个：①（本质上等价于 mid-vwap）的衰减 rank，刻画价格相对成交重心的偏离；② mid 与 adv40 的相关性经衰减后的 rank。更偏向要求“偏离与价量联动”同时不过强（或同时偏弱/偏强的一致性）。

### Alpha#78

```text
rank(
  correlation(
    sum((low * 0.352233) + (vwap * (1 - 0.352233)), 19.7428),
    sum(adv40, 19.7428),
    6.83313
  )
) ^ rank(correlation(rank(vwap), rank(volume), 5.77492))
```

解释：把“加权(low/vwap) 的累计”与“adv40 累计”的相关性做 rank，作为底数；把 vwap rank 与 volume rank 的相关性做 rank，作为指数。幂运算会放大两种价量结构同向强弱的组合效应。

### Alpha#79

```text
rank(delta(IndNeutralize((close * 0.60733) + (open * (1 - 0.60733)), IndClass.sector), 1.23438)) <
rank(correlation(Ts_Rank(vwap, 3.60973), Ts_Rank(adv150, 9.18637), 14.6644))
```

解释：比较“板块中性化的加权(close/open) 的差分”之 rank 与 “vwap 分位与 adv150 分位的相关性”之 rank，输出布尔条件。偏向用中性化价格变化去对比长期量能分位关系。

### Alpha#80

```text
(
  rank(
    Sign(
      delta(
        IndNeutralize((open * 0.868128) + (high * (1 - 0.868128)), IndClass.industry),
        4.04545
      )
    )
  ) ^
  Ts_Rank(correlation(high, adv10, 5.11456), 5.53756)
) * -1
```

解释：把行业中性化后的加权(open/high) 的差分取符号（离散方向）并 rank，作为底数；把 high 与 adv10 的相关性做时间序列排名，作为指数；幂运算后取负。属于“方向性结构 × 价量相关分位”的非线性反向组合。

---

## Alpha 81–101（结尾 + Alpha#101）

### Alpha#81

```text
(
  rank(
    Log(
      product(
        rank(
          (rank(correlation(vwap, sum(adv10, 49.6054), 8.47743)) ^ 4)
        ),
        14.9655
      )
    )
  ) <
  rank(correlation(rank(vwap), rank(volume), 5.07914))
) * -1
```

解释：左边用 `correlation(vwap, sum(adv10), ...)` 构造价量结构，先 rank 后做 4 次幂放大，再在约 15 天窗口内做连乘并取对数，最后再 rank；右边是 vwap 与 volume（均先横截面 rank）的相关性再 rank。若左边小于右边输出 -1，否则 0。属于“复杂非线性记忆项 vs 简化价量相关项”的条件触发型因子。

### Alpha#82

```text
min(
  rank(decay_linear(delta(open, 1.46063), 14.8717)),
  Ts_Rank(
    decay_linear(
      correlation(
        IndNeutralize(volume, IndClass.sector),
        (open * 0.634196) + (open * (1 - 0.634196)),
        17.4842
      ),
      6.92131
    ),
    13.4283
  )
) * -1
```

解释：取两条子信号的较小者再取负：① 开盘价差分经线性衰减后的横截面 rank；② 行业中性化成交量与（本质上等于 open 的）加权 open 的相关性，经衰减后做时间序列排名。用 `min` 强调“两个条件都不强时才更突出”的组合方式（并整体反向）。

### Alpha#83

```text
(
  rank(
    delay(
      (high - low) / (sum(close, 5) / 5),
      2
    )
  ) * rank(rank(volume))
) / (
  ((high - low) / (sum(close, 5) / 5)) /
  (vwap - close)
)
```

解释：分子把“两日前的相对振幅（range/近 5 日均价）”做 rank，并乘以成交量的 rank；分母用“当日相对振幅”与 “vwap-close 偏离”做比例。整体像是在用“过去波动 × 量能”去相对化“当前波动 / 价格偏离”，用于刻画波动与价格偏离之间的结构。注：参考 PDF 文本提取时末尾疑似漏括号，此处已按语法补全以便阅读。

### Alpha#84

```text
SignedPower(
  Ts_Rank((vwap - ts_max(vwap, 15.3217)), 20.7127),
  delta(close, 4.96796)
)
```

解释：先计算 vwap 相对其近 15 天最高值的偏离（偏离越负越“低于高点”），再做时间序列排名；然后用 `SignedPower` 做带符号幂变换，幂指数取 `delta(close, ~5)`（即价格变动强弱会改变非线性放大程度）。属于“位置分位 × 变动强弱”耦合的非线性因子。

### Alpha#85

```text
rank(correlation((high * 0.876703) + (close * (1 - 0.876703)), adv30, 9.61331)) ^
rank(
  correlation(
    Ts_Rank((high + low) / 2, 3.70596),
    Ts_Rank(volume, 10.1595),
    7.11408
  )
)
```

解释：两个相关性信号（均再做 rank）做幂运算：① 加权(high/close) 与 adv30 的相关性；② 中价的短期时间序列排名与成交量的时间序列排名之间的相关性。幂运算会放大二者同向强弱的组合效应，是典型的非线性增强结构。

### Alpha#86

```text
(
  Ts_Rank(
    correlation(close, sum(adv20, 14.7444), 6.00049),
    20.4195
  ) <
  rank((open + close) - (vwap + open))
) * -1
```

解释：左边是 close 与一段窗口内 adv20 累计的相关性，再做时间序列排名；右边其实等价于 `rank(close - vwap)`（开盘在表达式中抵消）。若左边小于右边输出 -1，否则 0。可理解为用“价量相关的相对位置”去对比“收盘相对 VWAP 的偏离”的条件触发。

### Alpha#87

```text
max(
  rank(
    decay_linear(
      delta((close * 0.369701) + (vwap * (1 - 0.369701)), 1.91233),
      2.65461
    )
  ),
  Ts_Rank(
    decay_linear(
      abs(correlation(IndNeutralize(adv81, IndClass.industry), close, 13.4132)),
      4.89768
    ),
    14.4535
  )
) * -1
```

解释：取两条子信号的较大者再取负：① 加权(close/vwap) 的差分经衰减后做 rank；② 行业中性化 adv81 与 close 的相关性取绝对值，再衰减并做时间序列排名。`max` 会强调“任一条很强就触发”的结构，整体为反向。

### Alpha#88

```text
min(
  rank(
    decay_linear(
      (rank(open) + rank(low)) - (rank(high) + rank(close)),
      8.06882
    )
  ),
  Ts_Rank(
    decay_linear(
      correlation(
        Ts_Rank(close, 8.44728),
        Ts_Rank(adv60, 20.6966),
        8.01266
      ),
      6.65053
    ),
    2.61957
  )
)
```

解释：取两条子信号的较小者：① (open+low) 的横截面 rank 和 (high+close) 的横截面 rank 之差，经衰减后再 rank，刻画当日价格在区间中的偏向；② close 的时间序列排名与 adv60 的时间序列排名之间的相关性，经衰减后做时间序列排名。`min` 更偏向“二者同时偏弱/偏强的一致性”。

### Alpha#89

```text
Ts_Rank(
  decay_linear(
    correlation(
      (low * 0.967285) + (low * (1 - 0.967285)),
      adv10,
      6.94279
    ),
    5.51607
  ),
  3.79744
) -
Ts_Rank(
  decay_linear(
    delta(IndNeutralize(vwap, IndClass.industry), 3.48158),
    10.1466
  ),
  15.3012
)
```

解释：两个时间序列排名信号相减：第一项是（本质上等于 low 的）加权 low 与 adv10 的相关性，经衰减后做时间序列排名；第二项是行业中性化 VWAP 的差分，经衰减后做时间序列排名。差值结构表示在两类“价量联动/行业中性趋势”之间做相对比较。

### Alpha#90

```text
(
  rank(close - ts_max(close, 4.66719)) ^
  Ts_Rank(
    correlation(IndNeutralize(adv40, IndClass.subindustry), low, 5.38375),
    3.21856
  )
) * -1
```

解释：把“close 相对近期最高 close 的偏离”做 rank，与“次行业中性化 adv40 与 low 的相关性”的时间序列排名做幂运算并取负。强调“从高点回撤的程度”与“量能/低价联动结构”的非线性组合，方向为反向。

### Alpha#91

```text
(
  Ts_Rank(
    decay_linear(
      decay_linear(
        correlation(IndNeutralize(close, IndClass.industry), volume, 9.74928),
        16.398
      ),
      3.83219
    ),
    4.8667
  ) -
  rank(decay_linear(correlation(vwap, adv30, 4.01303), 2.6809))
) * -1
```

解释：两部分做差后取负：① 行业中性化 close 与 volume 的相关性，做两次线性衰减后再做时间序列排名（更强的“记忆/平滑”）；② vwap 与 adv30 的相关性做衰减后做横截面 rank。整体比较“行业中性价量结构的时间序列位置”与“VWAP-量能结构的截面强弱”，并反向排序。

### Alpha#92

```text
min(
  Ts_Rank(
    decay_linear(
      (((high + low) / 2) + close) < (low + open),
      14.7221
    ),
    18.8683
  ),
  Ts_Rank(
    decay_linear(correlation(rank(low), rank(adv30), 7.58555), 6.94024),
    6.80584
  )
)
```

解释：`min` 组合两条“时间序列位置”信号：① 一个价格形态条件 `mid + close < low + open`（偏向弱势/下沿结构）的衰减序列，再做时间序列排名；② low 与 adv30（均先横截面 rank）的相关性，经衰减后做时间序列排名。偏向要求“形态与价量联动”同时处于某种相对位置。

### Alpha#93

```text
Ts_Rank(
  decay_linear(
    correlation(IndNeutralize(vwap, IndClass.industry), adv81, 17.4193),
    19.848
  ),
  7.54455
) / rank(
  decay_linear(
    delta((close * 0.524434) + (vwap * (1 - 0.524434)), 2.77377),
    16.2664
  )
)
```

解释：分子是行业中性化 VWAP 与 adv81 的相关性，经衰减后做时间序列排名；分母是加权(close/vwap) 的差分经衰减后做横截面 rank。比值结构相当于用“行业中性价量关系的相对位置”去缩放“价格变化项”的截面强弱。

### Alpha#94

```text
(
  rank(vwap - ts_min(vwap, 11.5783)) ^
  Ts_Rank(
    correlation(
      Ts_Rank(vwap, 19.6462),
      Ts_Rank(adv60, 4.02992),
      18.0926
    ),
    2.70756
  )
) * -1
```

解释：把“vwap 相对近期最低 vwap 的偏离”做 rank，与“vwap 的时间序列排名”和“adv60 的时间序列排名”的相关性再做时间序列排名，二者做幂运算并取负。强调“VWAP 位置”与“价量分位相关结构”的非线性耦合，方向为反向。

### Alpha#95

```text
rank(open - ts_min(open, 12.4105)) <
Ts_Rank(
  (rank(correlation(sum((high + low) / 2, 19.1351), sum(adv40, 19.1351), 12.8742)) ^ 5),
  11.7584
)
```

解释：比较“开盘相对近期最低开盘的偏离”与“(中价累计 vs adv40 累计) 的相关性（再 5 次幂放大）”的时间序列排名。结果为布尔条件（真/假），用于判断“开盘位置”是否低于“某种中价-量能相关结构”的相对位置。

### Alpha#96

```text
max(
  Ts_Rank(
    decay_linear(correlation(rank(vwap), rank(volume), 3.83878), 4.16783),
    8.38151
  ),
  Ts_Rank(
    decay_linear(
      Ts_ArgMax(
        correlation(
          Ts_Rank(close, 7.45404),
          Ts_Rank(adv60, 4.13242),
          3.65459
        ),
        12.6556
      ),
      14.0365
    ),
    13.4143
  )
) * -1
```

解释：取两条复杂子信号的较大者再取负：① vwap 与 volume（均先横截面 rank）的相关性，经衰减后做时间序列排名；② close 的时间序列排名与 adv60 的时间序列排名做相关，再在窗口内取相关性的 `Ts_ArgMax`（“相关性峰值出现的位置”），随后衰减并做时间序列排名。整体强调“价量关系”及其“峰值位置”的结构信息，并以 `max` 进行非线性聚合。

### Alpha#97

```text
(
  rank(
    decay_linear(
      delta(
        IndNeutralize(
          (low * 0.721001) + (vwap * (1 - 0.721001)),
          IndClass.industry
        ),
        3.3705
      ),
      20.4523
    )
  ) -
  Ts_Rank(
    decay_linear(
      Ts_Rank(
        correlation(
          Ts_Rank(low, 7.87871),
          Ts_Rank(adv60, 17.255),
          4.97547
        ),
        18.5925
      ),
      15.7152
    ),
    6.71659
  )
) * -1
```

解释：两部分做差后取负：① 行业中性化后的加权(low/vwap) 做差分并长窗口衰减，再做横截面 rank；② low 与 adv60 的时间序列排名做相关，再做时间序列排名，并衰减后再次做时间序列排名。整体是在比较“行业中性价格变化”与“价量分位相关的多层平滑位置”，并反向排序。

### Alpha#98

```text
rank(decay_linear(correlation(vwap, sum(adv5, 26.4719), 4.58418), 7.18088)) -
rank(
  decay_linear(
    Ts_Rank(
      Ts_ArgMin(correlation(rank(open), rank(adv15), 20.8187), 8.62571),
      6.95668
    ),
    8.07206
  )
)
```

解释：差值结构：第一项是 vwap 与短期量能累计的相关性，经衰减后做 rank；第二项先计算 open 与 adv15（均先横截面 rank）的相关性，在窗口里取 `Ts_ArgMin`（相关性谷值出现位置），再做时间序列排名并衰减，最后做 rank。整体用“相关强度”对比“相关性谷值位置”的结构特征。

### Alpha#99

```text
(
  rank(
    correlation(
      sum((high + low) / 2, 19.8975),
      sum(adv60, 19.8975),
      8.8136
    )
  ) <
  rank(correlation(low, volume, 6.28259))
) * -1
```

解释：比较“中价累计与 adv60 累计的相关性”的 rank 与 “low 与 volume 的相关性”的 rank；若前者更小则输出 -1，否则 0。属于用两类价量相关结构做相对比较的条件触发。

### Alpha#100

```text
0 - (
  1 * (
    (
      (
        1.5 *
        scale(
          indneutralize(
            indneutralize(
              rank(
                (
                  ((close - low) - (high - close)) /
                  (high - low)
                ) * volume
              ),
              IndClass.subindustry
            ),
            IndClass.subindustry
          )
        )
      ) -
      scale(
        indneutralize(
          (
            correlation(close, rank(adv20), 5) -
            rank(ts_argmin(close, 30))
          ),
          IndClass.subindustry
        )
      )
    ) * (volume / adv20)
  )
)
```

解释：一个很重的组合因子，最终整体取负。核心思路是：用“价格在当日区间的位置”（类似随机指标：`(2*close - high - low)/(high-low)`）乘以 volume，做 rank 后进行（次行业）两次中性化并 scale；减去另一条（次行业中性化后的）信号：`correlation(close, rank(adv20), 5) - rank(ts_argmin(close, 30))` 再 scale；两者差乘以相对成交量 `(volume/adv20)`，并用 1.5 放大第一项。直觉上是在“区间位置×量能”的行业中性版本与“价量相关/低点位置”的行业中性版本之间做加权对比，并在放量时放大影响。

### Alpha#101

```text
(close - open) / ((high - low) + 0.001)
```

解释：日内从开到收的方向幅度（close-open），用当日振幅（high-low）做归一化，并加一个很小的常数防止除零。可以看作“日内强弱（相对区间）”指标：越接近 +1 表示收在区间上沿附近，越接近 -1 表示收在下沿附近。
