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
    """Bagging分类器。"""
    def __init__(self, n_estimators=10, max_samples=1.0):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.classifiers = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        self.classifiers = []
        for _ in range(self.n_estimators):
            indices = np.random.choice(n_samples, int(n_samples * self.max_samples), replace=True)
            clf = DecisionTree(max_depth=10)
            clf.fit(X[indices], y[indices])
            self.classifiers.append(clf)
        return self

    def predict(self, X):
        predictions = np.array([clf.predict(X) for clf in self.classifiers])
        return np.array([np.bincount(p).argmax() for p in predictions.T])

    def score(self, X, y):
        return np.mean(self.predict(X) == y)