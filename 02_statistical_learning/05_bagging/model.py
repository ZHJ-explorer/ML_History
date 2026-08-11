"""Bagging实现。"""
import numpy as np
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, '02_statistical_learning', '01_decision_tree'))

from decision_tree import DecisionTree

class BaggingClassifier:
    """Bagging分类器。通过Bootstrap抽样集成多个基学习器来提高模型稳定性。"""
    def __init__(self, n_estimators=10, max_samples=1.0):
        """初始化Bagging分类器。

        Args:
            n_estimators: 基学习器数量，默认为10。
            max_samples: 每个基学习器使用的样本比例，默认为1.0（全量）。
        """
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.classifiers = []

    def fit(self, X, y):
        """训练Bagging分类器。

        Args:
            X: 训练特征矩阵，形状为(n_samples, n_features)。
            y: 训练标签，形状为(n_samples,)。

        Returns:
            self: 训练后的分类器实例。
        """
        n_samples = X.shape[0]
        self.classifiers = []
        for _ in range(self.n_estimators):
            indices = np.random.choice(n_samples, int(n_samples * self.max_samples), replace=True)
            clf = DecisionTree(max_depth=10)
            clf.fit(X[indices], y[indices])
            self.classifiers.append(clf)
        return self

    def predict(self, X):
        """对输入数据进行预测。

        Args:
            X: 输入特征矩阵，形状为(n_samples, n_features)。

        Returns:
            预测标签数组，形状为(n_samples,)。
        """
        predictions = np.array([clf.predict(X) for clf in self.classifiers])
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