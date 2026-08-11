# AdaBoost

## 历史背景

1995年，Schapire和Freund提出了AdaBoost算法，这是第一个实用的boosting算法。它通过迭代地训练弱分类器并调整样本权重，将多个弱分类器组合成一个强分类器。

**核心突破**：
- 自适应调整样本权重
- 理论保证：指数衰减的误差上界

## 数学原理

### 弱分类器加权

$$G_m(x) = \text{sign}(\sum_{i=1}^m \alpha_m G_m(x))$$

### 系数计算

$$\alpha_m = \frac{1}{2}\log\frac{1-e_m}{e_m}$$

### 样本权重更新

$$w_{i,m+1} = w_{i,m}\exp(-\alpha_m y_i G_m(x_i))$$

## 思考题

1. AdaBoost和Bagging有什么区别？
2. 为什么AdaBoost对异常值敏感？