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

    def test_model_creation(self):
        model = LightGBM(n_estimators=10, max_depth=3, learning_rate=0.1)
        assert model is not None

if __name__ == "__main__":
    pytest.main([__file__, "-v"])