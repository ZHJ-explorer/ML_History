# GBDT（梯度提升决策树）

## 历史背景

Friedman于2001年提出梯度提升机（GBM），将梯度下降的思想应用于树模型。XGBoost和LightGBM是其后来的高效实现。

## 数学原理

1. 初始化：$f_0(x) = \arg\min_\gamma \sum L(y_i, \gamma)$
2. 迭代：$f_m(x) = f_{m-1}(x) + \alpha \cdot h_m(x)$
3. 其中$h_m(x)$拟合负梯度（伪残差）

## 思考题

1. GBDT和随机森林有什么区别？
2. 为什么GBDT通常比随机森林效果更好？