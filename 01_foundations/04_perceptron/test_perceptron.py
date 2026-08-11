"""感知机模型测试。"""
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import Perceptron

class TestPerceptron:
    def test_and_gate(self):
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 0, 0, 1])
        model = Perceptron()
        model.fit(X, y)
        assert model.score(X, y) == 1.0

    def test_or_gate(self):
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 1, 1, 1])
        model = Perceptron()
        model.fit(X, y)
        assert model.score(X, y) == 1.0

    def test_xor_fail(self):
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 1, 1, 0])
        model = Perceptron()
        model.fit(X, y)
        assert model.score(X, y) < 1.0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])