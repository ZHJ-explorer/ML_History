"""决策树模型测试。"""
import pytest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decision_tree import DecisionTree

class TestDecisionTree:
    def test_basic(self):
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 1, 1, 0])
        model = DecisionTree(max_depth=2)
        model.fit(X, y)
        assert model.score(X, y) == 1.0

    def test_iris(self):
        from sklearn.datasets import load_iris
        X, y = load_iris().data, load_iris().target
        model = DecisionTree(max_depth=5)
        model.fit(X, y)
        assert model.score(X, y) > 0.9

if __name__ == "__main__":
    pytest.main([__file__, "-v"])