"""AdaBoost实现。"""
import numpy as np
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, '02_statistical_learning', '01_decision_tree'))

from decision_tree import DecisionTree

class AdaBoost:
    """AdaBoost自适应增强算法。"""
    def __init__(self, n_estimators=10, learning_rate=1.0):
        """初始化AdaBoost分类器。

        Args:
            n_estimators: 弱学习器数量，默认为10。
            learning_rate: 学习率（shrinkage），默认为1.0。
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.classifiers = []
        self.alphas = []

    def fit(self, X, y):
        """训练AdaBoost分类器。

        Args:
            X: 训练特征矩阵，形状为(n_samples, n_features)。
            y: 训练标签，形状为(n_samples,)。

        Returns:
            self: 训练后的分类器实例。
        """
        n_samples = X.shape[0]
        weights = np.ones(n_samples) / n_samples
        for _ in range(self.n_estimators):
            clf = DecisionTree(max_depth=1)
            clf.fit(X, y)
            predictions = clf.predict(X)
            err = np.sum(weights * (predictions != y))
            if err >= 0.5:
                break
            alpha = self.learning_rate * 0.5 * np.log((1 - err) / err)
            weights *= np.exp(-alpha * y * predictions)
            weights /= np.sum(weights)
            self.classifiers.append(clf)
            self.alphas.append(alpha)
        return self

    def predict(self, X):
        """对输入数据进行预测。

        Args:
            X: 输入特征矩阵，形状为(n_samples, n_features)。

        Returns:
            预测标签数组，形状为(n_samples,)。
        """
        predictions = np.zeros(X.shape[0])
        for clf, alpha in zip(self.classifiers, self.alphas):
            predictions += alpha * clf.predict(X)
        return np.sign(predictions)

    def score(self, X, y):
        """计算分类准确率。

        Args:
            X: 输入特征矩阵，形状为(n_samples, n_features)。
            y: 真实标签，形状为(n_samples,)。

        Returns:
            分类准确率（0到1之间的浮点数）。
        """
        return np.mean(self.predict(X) == y)