"""逻辑回归模型测试。"""
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import LogisticRegression

class TestLogisticRegression:
    def test_binary_classification(self):
        np.random.seed(42)
        X = np.random.rand(100, 2)
        y = (X[:, 0] + X[:, 1] > 1).astype(int)
        model = LogisticRegression(learning_rate=0.1, max_iter=1000)
        model.fit(X, y)
        assert model.score(X, y) > 0.8

    def test_predict_proba(self):
        model = LogisticRegression()
        model.weights = np.array([1.0, 1.0])
        model.bias = 0.0
        X = np.array([[0.5, 0.5]])
        proba = model.predict_proba(X)
        assert 0 < proba[0] < 1

if __name__ == "__main__":
    pytest.main([__file__, "-v"])