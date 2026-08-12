"""Bagging分类器测试。"""
import pytest
import numpy as np
import os
from sklearn.datasets import load_iris
import importlib.util

# Load model from the same directory using importlib to avoid path conflicts
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("bagging_model", os.path.join(SCRIPT_DIR, "model.py"))
bagging_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bagging_module)
BaggingClassifier = bagging_module.BaggingClassifier

# Load decision tree for reference
DT_SCRIPT = os.path.join(os.path.dirname(SCRIPT_DIR), '01_decision_tree')
dt_spec = importlib.util.spec_from_file_location("decision_tree", os.path.join(DT_SCRIPT, "model.py"))
dt_module = importlib.util.module_from_spec(dt_spec)
dt_spec.loader.exec_module(dt_module)


class TestBaggingClassifier:
    """Bagging分类器测试。"""

    def test_and_gate(self):
        """测试简单AND门逻辑。"""
        np.random.seed(42)
        X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        y = np.array([0, 0, 0, 1])
        model = BaggingClassifier(n_estimators=10, max_samples=1.0)
        model.fit(X, y)
        assert model.score(X, y) >= 0.75

    def test_iris(self):
        """测试Iris数据集三分类。"""
        np.random.seed(42)
        iris = load_iris()
        X, y = iris.data, iris.target
        model = BaggingClassifier(n_estimators=10, max_samples=1.0)
        model.fit(X, y)
        assert model.score(X, y) >= 0.9

    def test_predict_output_type(self):
        """测试predict返回类型和形状。"""
        np.random.seed(42)
        X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        y = np.array([0, 0, 0, 1])
        model = BaggingClassifier(n_estimators=5)
        model.fit(X, y)
        pred = model.predict(X)
        assert isinstance(pred, np.ndarray)
        assert pred.shape == (4,)

    def test_score_range(self):
        """测试score返回值在0-1之间。"""
        np.random.seed(42)
        X = np.random.rand(50, 4)
        y = np.random.randint(0, 3, 50)
        model = BaggingClassifier(n_estimators=10)
        model.fit(X, y)
        score = model.score(X, y)
        assert 0.0 <= score <= 1.0

    def test_fit_returns_self(self):
        """测试fit返回self。"""
        np.random.seed(42)
        X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        y = np.array([0, 0, 0, 1])
        model = BaggingClassifier(n_estimators=5)
        result = model.fit(X, y)
        assert result is model

    def test_different_max_samples(self):
        """测试不同max_samples参数。"""
        np.random.seed(42)
        iris = load_iris()
        X, y = iris.data, iris.target
        model1 = BaggingClassifier(n_estimators=10, max_samples=1.0)
        model1.fit(X, y)
        model05 = BaggingClassifier(n_estimators=10, max_samples=0.5)
        model05.fit(X, y)
        assert 0.0 <= model1.score(X, y) <= 1.0
        assert 0.0 <= model05.score(X, y) <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
