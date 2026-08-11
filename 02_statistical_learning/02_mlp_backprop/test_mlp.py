"""MLP模型测试。"""
import pytest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import MLP

class TestMLP:
    def test_xor(self):
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0], [1], [1], [0]])
        model = MLP(layer_sizes=[2, 4, 1], learning_rate=0.1, max_iter=10000)
        model.fit(X, y)
        pred = (model.predict(X) > 0.5).astype(int)
        assert np.array_equal(pred, y)

    def test_simple(self):
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0], [1], [1], [0]])
        model = MLP(layer_sizes=[2, 3, 1], learning_rate=0.1, max_iter=1000)
        model.fit(X, y)
        assert model.score(X, y) > 0.8

if __name__ == "__main__":
    pytest.main([__file__, "-v"])