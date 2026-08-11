# Bagging（Bootstrap Aggregating）

## 历史背景

Bagging（Bootstrap Aggregating）由Leo Breiman于1996年提出，核心思想是通过构建多个独立的基学习器并集成它们的预测结果（回归取平均，分类取投票）来提高模型的稳定性和泛化能力。它是集成学习中最经典的方法之一。

## 数学原理

1. **Bootstrap抽样**：从训练集有放回地抽样B次，得到B个子数据集
2. **独立训练**：对每个子数据集训练一个基学习器 $h_i$
3. **集成预测**：
   - 分类：$\hat{y} = \text{mode}(h_1(x), h_2(x), ..., h_B(x))$
   - 回归：$\hat{y} = \frac{1}{B}\sum_{i=1}^{B} h_i(x)$

## 代码实现

本目录使用决策树作为基学习器，通过Bootstrap抽样训练多个决策树分类器。

## 思考题

1. Bagging和Boosting有什么区别？
2. 为什么Bagging可以减少方差而不影响偏差？
3. OOB（Out-of-Bag）错误估计的原理是什么？