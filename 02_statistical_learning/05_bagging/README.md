# Bagging

## 历史背景

Bagging（Bootstrap Aggregating）由Breiman于1996年提出，通过构建多个独立的基学习器并取平均（回归）或投票（分类）来提高模型稳定性。

## 数学原理

1. 从训练集有放回抽样B次得到B个子集
2. 对每个子集训练一个基学习器
3. 预测时取所有基学习器的平均或投票

## 思考题

1. Bagging和Boosting有什么区别？
2. 为什么Bagging可以减少方差？