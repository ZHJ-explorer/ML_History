"""GBDT实现。"""
import numpy as np
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, '02_statistical_learning', '01_decision_tree'))

from decision_tree import DecisionTree

class GBDT:
    """梯度提升决策树分类器。每棵树拟合前序模型的负梯度（伪残差）。"""
    def __init__(self, n_estimators=10, learning_rate=0.1, max_depth=3):
        """初始化GBDT分类器。

        Args:
            n_estimators: 树的数量，默认为10。
            learning_rate: 学习率（shrinkage），默认为0.1。
            max_depth: 每棵树的最大深度，默认为3。
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []

    def _sigmoid(self, z):
        """Sigmoid激活函数。

        Args:
            z: 输入张量或数组。

        Returns:
            Sigmoid输出值，范围(0, 1)。
        """
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y):
        """训练GBDT分类器。

        Args:
            X: 训练特征矩阵，形状为(n_samples, n_features)。
            y: 训练标签（0或1），形状为(n_samples,)。

        Returns:
            self: 训练后的分类器实例。
        """
        n_samples = X.shape[0]
        self.trees = []
        F = np.zeros(n_samples)
        for _ in range(self.n_estimators):
            p = self._sigmoid(F)
            residual = y - p
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X, residual)
            predictions = tree.predict(X)
            F += self.learning_rate * predictions
            self.trees.append(tree)
        return self

    def predict(self, X):
        """对输入数据进行预测。

        Args:
            X: 输入特征矩阵，形状为(n_samples, n_features)。

        Returns:
            预测标签数组（0或1），形状为(n_samples,)。
        """
        F = np.zeros(X.shape[0])
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)
        return (self._sigmoid(F) >= 0.5).astype(int)

    def score(self, X, y):
        """计算分类准确率。

        Args:
            X: 输入特征矩阵，形状为(n_samples, n_features)。
            y: 真实标签，形状为(n_samples,)。

        Returns:
            分类准确率（0到1之间的浮点数）。
        """
        return np.mean(self.predict(X) == y)