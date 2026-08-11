# XGBoost

## 概述

XGBoost（eXtreme Gradient Boosting）是由 Tianqi Chen 于 2016 年提出的梯度提升树框架，以其高效的计算性能和优秀的预测精度成为机器学习竞赛中的利器。

## 核心创新

### 1. 二阶泰勒展开优化

XGBoost 在传统 GBDT 的一阶梯度基础上，引入了二阶导数（Hessian）信息：

$$Obj \approx \sum_i [l(y_i, \hat{y}_i^{(t)}) + g_i f_t(x_i) + \frac{1}{2}h_i f_t^2(x_i)] + \Omega(f_t)$$

其中 $g_i$ 是一阶梯度，$h_i$ 是二阶导数。

### 2. 正则化项

XGBoost 在目标函数中显式加入了正则化项：

$$\Omega(f) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2$$

- $\gamma$（Gamma）：控制树的复杂度，惩罚叶子节点数量
- $\lambda$（Lambda）：L2 正则化系数，惩罚叶子权重

### 3. 分裂增益计算

$$Gain = \frac{1}{2}\left[\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{G_{Total}^2}{H_{Total} + \lambda}\right] - \gamma$$

只有当 Gain > 0 时才进行分裂。

## 参数说明

| 参数 | 说明 | 典型值 |
|------|------|--------|
| n_estimators | 树的数量 | 50-500 |
| max_depth | 树的最大深度 | 3-8 |
| learning_rate | 学习率（shrinkage） | 0.01-0.3 |
| gamma | 叶子节点惩罚 | 0-5 |
| lambda | L2正则化系数 | 0-10 |

## 优势

1. **精度高**：二阶优化使收敛更快更准
2. **防过拟合**：内置正则化机制
3. **高效**：支持并行处理和缺失值处理
4. **灵活**：支持自定义损失函数

## 应用场景

- 结构化数据竞赛（Kaggle）
- 回归和分类任务
- 特征选择
- 异常检测
