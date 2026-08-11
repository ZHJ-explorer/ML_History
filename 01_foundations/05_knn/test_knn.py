"""k近邻模型测试。"""
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import KNN

class TestKNN:
    def test_basic(self):
        X = np.array([[0, 0], [1, 1], [2, 2], [10, 10]])
        y = np.array([0, 0, 0, 1])
        model = KNN(k=3)
        model.fit(X, y)
        assert model.predict([[1, 1]])[0] == 0

    def test_different_k(self):
        X = np.array([[0, 0], [1, 1], [2, 2]])
        y = np.array([0, 0, 1])
        model = KNN(k=1)
        model.fit(X, y)
        assert model.score(X, y) == 1.0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])