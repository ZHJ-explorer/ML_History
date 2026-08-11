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
    """随机森林分类器。"""
    def __init__(self, n_estimators=10, max_features=None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
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
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.array([np.bincount(p).argmax() for p in predictions.T])

    def score(self, X, y):
        return np.mean(self.predict(X) == y)