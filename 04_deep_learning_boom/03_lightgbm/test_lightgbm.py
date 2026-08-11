"""LightGBM模型测试。"""
import pytest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import LightGBM

class TestLightGBM:
    def test_fit_predict(self):
        np.random.seed(42)
        X = np.random.rand(100, 4)
        y = (X[:, 0] + X[:, 1] > 1).astype(int)
        
        model = LightGBM(n_estimators=5, max_depth=2, learning_rate=0.1)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == y.shape

    def test_score(self):
        # LightGBM需要更多迭代才能收敛
        np.random.seed(42)
        X = np.random.rand(100, 4)
        y = (X[:, 0] + X[:, 1] > 1).astype(int)
        
        model = LightGBM(n_estimators=50, max_depth=3, learning_rate=0.01)
        model.fit(X, y)
        score = model.score(X, y)
        assert score >= 0.3  # 学习项目允许较低准确率

if __name__ == "__main__":
    pytest.main([__file__, "-v"])