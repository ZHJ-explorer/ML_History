"""k-Means模型测试。"""
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import KMeans

class TestKMeans:
    def test_fit(self):
        np.random.seed(42)
        X = np.random.rand(100, 2)
        model = KMeans(k=3, random_state=42)
        model.fit(X)
        assert model.centroids.shape == (3, 2)
        assert len(model.labels) == 100

    def test_predict(self):
        np.random.seed(42)
        X = np.random.rand(100, 2)
        model = KMeans(k=3, random_state=42)
        model.fit(X)
        labels = model.predict(X)
        assert len(labels) == 100
        assert all(l in [0, 1, 2] for l in labels)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])