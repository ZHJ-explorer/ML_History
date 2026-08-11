# 随机森林（Random Forest）

## 简介

随机森林（Random Forest）是由 Leo Breiman 于 2001 年提出的一种集成学习算法，是对 Bagging（Bootstrap Aggregating）的重要改进。它在传统 Bagging 的基础上引入了**随机特征选择机制**——在每个节点分裂时，不再考虑所有特征，而是从 m 个特征中随机选取 k 个（通常 k = √m），只在这 k 个特征中寻找最优分裂点。这一微小却关键的改动，进一步降低了树之间的相关性，从而显著提升了模型的泛化能力。

随机森林可以同时处理分类和回归任务，在分类任务中通过多数投票决定最终结果，在回归任务中则取所有树预测的平均值。由于其出色的性能和鲁棒性，随机森林已成为工业界和学术界最广泛使用的机器学习算法之一。

## 核心创新

随机森林的核心创新在于两个随机性的引入：

1. **数据随机性（Bootstrap 采样）**：每棵树使用有放回抽样构建独立的训练子集，确保每棵树看到的数据分布略有不同。
2. **特征随机性（Random Feature Subspace）**：在每个节点分裂时，从全部 m 个特征中随机选取 k = √m 个候选特征，只在这些特征中寻找最优分裂点。这迫使不同树关注不同的特征维度，进一步降低树间相关性。

这两个随机机制的组合，使得随机森林在保持较低方差的同时，不会像单棵决策树那样容易过拟合。即使每棵树都对训练数据有较高的拟合程度（甚至过拟合），它们的集成结果依然稳健。

## 数学原理

随机森林的预测过程可以用以下公式描述：

**Bootstrap 采样：**
$$D_b = \{(x_i^{(b)}, y_i^{(b)})\}_{i=1}^{n} , \quad (x_i^{(b)}, y_i^{(b)}) \sim \text{Uniform}(D), \quad b = 1, \ldots, B$$

**随机特征选择（节点分裂）：**
$$S_t = \text{RandomSubset}(\mathcal{F}, k), \quad k = \lfloor\sqrt{m}\rfloor$$
$$t^* = \arg\max_{j \in S_t} \text{ImpurityReduction}(t, j)$$

**投票集成（分类）：**
$$\hat{y} = \underset{c}{\arg\max} \sum_{b=1}^{B} \mathbb{1}(h_b(x) = c)$$

**平均值集成（回归）：**
$$\hat{y} = \frac{1}{B} \sum_{b=1}^{B} h_b(x)$$

其中 $B$ 是树的数量，$h_b$ 是第 $b$ 棵决策树，$\mathbb{1}(\cdot)$ 是指示函数。

## 影响与后续发展

随机森林的提出标志着集成学习从理论走向大规模应用。其影响体现在多个方面：

- **基准模型**：随机森林长期以来被视为分类任务的强基准（baseline），在许多公开数据集上都能取得优异结果。
- **特征重要性**：随机森林提供了两种特征重要性度量——Gini 重要性和置换重要性（Permutation Importance），成为可解释机器学习的重要工具。
- **处理高维数据**：随机森林天然支持高维特征空间，无需特征缩放，对缺失值也有较好的容忍度。
- **后续变体**：在随机森林基础上发展出了 Extra-Trees（极端随机树）、Half-Random-Forest 等变体，以及针对大数据场景的并行化实现（如 Hoeffding Trees）。

Breiman 的原始论文《Random Forests》发表在 *Machine Learning* 期刊（2001, 45(1): 5–32），是该领域被引用次数最多的论文之一。

## 参考

- Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
- Breiman, L. (1996). Bagging Predictors. *Machine Learning*, 24(2), 123–140.
- scikit-learn documentation: [Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#random-forests)
- 本项目实现：`02_statistical_learning/06_random_forest/model.py`
