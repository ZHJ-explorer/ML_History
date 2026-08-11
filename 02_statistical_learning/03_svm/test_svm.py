"""SVM模型测试。"""
import pytest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import SVM

class TestSVM:
    def test_linear_separable(self):
        np.random.seed(42)
        X = np.random.rand(50, 2)
        y = np.where(X[:, 0] + X[:, 1] > 1, 1, -1)
        model = SVM(C=1.0, kernel='linear')
        model.fit(X, y)
        assert model.score(X, y) > 0.8

    def test_rbf_kernel(self):
        np.random.seed(42)
        X = np.random.rand(50, 2)
        y = np.where(X[:, 0] * X[:, 1] > 0.25, 1, -1)
        model = SVM(C=1.0, kernel='rbf')
        model.fit(X, y)
        assert model.score(X, y) > 0.7

if __name__ == "__main__":
    pytest.main([__file__, "-v"])