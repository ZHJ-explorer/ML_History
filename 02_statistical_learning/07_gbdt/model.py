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
    """梯度提升决策树分类器。"""
    def __init__(self, n_estimators=10, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y):
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
        F = np.zeros(X.shape[0])
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)
        return (self._sigmoid(F) >= 0.5).astype(int)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)