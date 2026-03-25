IC：
( $IC = \text{Corr}(f_t,\ r_{t+1\rightarrow t+2}) $)

ICIR（滚动相关的均值/波动）：
先算滚动相关：
( $IC_t = \text{Corr}(f_{t-w+1:t},\ r_{t-w+1:t})$ )
再做：
( $ICIR = \frac{\text{mean}(IC_t)}{\text{std}(IC_t)} $)

Sharpe（收益序列的均值/波动，年化）：
( $Sharpe = \sqrt{250} \cdot \frac{\text{mean}(R)}{\text{std}(R)} $)