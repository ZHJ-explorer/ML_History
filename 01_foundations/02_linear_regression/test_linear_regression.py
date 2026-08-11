"""线性回归模型测试。"""
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import LinearRegression

class TestLinearRegression:
    def test_basic_relationship(self):
        np.random.seed(42)
        X = np.random.rand(100, 1) * 10
        y = 3 * X.squeeze() + 2
        model = LinearRegression()
        model.fit(X, y)
        assert abs(model.weights[0] - 3.0) < 0.1
        assert abs(model.bias - 2.0) < 0.1

    def test_r2_score(self):
        np.random.seed(42)
        X = np.random.rand(100, 1) * 10
        y = 3 * X.squeeze() + 2
        model = LinearRegression()
        model.fit(X, y)
        assert model.score(X, y) > 0.99

    def test_unfitted(self):
        model = LinearRegression()
        with pytest.raises(ValueError):
            model.predict(np.array([[1, 2]]))

if __name__ == "__main__":
    pytest.main([__file__, "-v"])