"""随机森林实现。"""
import numpy as np
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, '02_statistical_learning', '01_decision_tree'))

from decision_tree import DecisionTree

class RandomForest:
    """随机森林分类器。在Bagging基础上引入随机特征选择。"""
    def __init__(self, n_estimators=10, max_features=None):
        """初始化随机森林分类器。

        Args:
            n_estimators: 树的数量，默认为10。
            max_features: 每个节点考虑的最大特征数，默认为sqrt(n_features)。
        """
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        """训练随机森林分类器。

        Args:
            X: 训练特征矩阵，形状为(n_samples, n_features)。
            y: 训练标签，形状为(n_samples,)。

        Returns:
            self: 训练后的分类器实例。
        """
        n_samples, n_features = X.shape
        self.max_features = self.max_features or int(np.sqrt(n_features))
        self.trees = []
        for _ in range(self.n_estimators):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_sub = X[indices]
            y_sub = y[indices]
            clf = DecisionTree(max_depth=10)
            clf.fit(X_sub, y_sub)
            self.trees.append(clf)
        return self

    def predict(self, X):
        """对输入数据进行预测。

        Args:
            X: 输入特征矩阵，形状为(n_samples, n_features)。

        Returns:
            预测标签数组，形状为(n_samples,)。
        """
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.array([np.bincount(p).argmax() for p in predictions.T])

    def score(self, X, y):
        """计算分类准确率。

        Args:
            X: 输入特征矩阵，形状为(n_samples, n_features)。
            y: 真实标签，形状为(n_samples,)。

        Returns:
            分类准确率（0到1之间的浮点数）。
        """
        return np.mean(self.predict(X) == y)