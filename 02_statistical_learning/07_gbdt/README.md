# GBDT（梯度提升决策树）

## 历史背景

Jerome Friedman于2001年正式提出梯度提升机（Gradient Boosting Machine, GBM），将梯度下降优化思想应用于决策树集成。其核心创新是用负梯度（伪残差）来拟合每棵新树。后续XGBoost（2016）和LightGBM（2017）在其基础上进一步引入了正则化和直方图算法等优化。

## 数学原理

1. **初始化**：$f_0(x) = \arg\min_\gamma \sum_i L(y_i, \gamma)$
2. **迭代优化**（$m = 1, 2, ..., M$）：
   - 计算负梯度（伪残差）：$r_{im} = -\left[\frac{\partial L(y_i, f(x_i))}{\partial f(x_i)}\right]_{f=f_{m-1}}$
   - 用基学习器拟合：$h_m = \arg\min \sum_i (y_i - h_m(x_i))^2$
   - 更新：$f_m(x) = f_{m-1}(x) + \nu \cdot h_m(x)$（$\nu$为学习率）

## 代码实现

本目录实现了基于决策树的GBDT分类器，每棵树拟合前序模型的负梯度（伪残差）。

## 思考题

1. GBDT和随机森林有什么区别？
2. 为什么GBDT通常比随机森林效果更好？
3. 学习率（shrinkage）在GBDT中的作用是什么？