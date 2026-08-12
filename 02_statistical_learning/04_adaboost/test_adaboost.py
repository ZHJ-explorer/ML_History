"""AdaBoost模型测试。"""
import pytest
import numpy as np
import os
import importlib.util

# Load model from the same directory using importlib to avoid path conflicts
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("adaboost_model", os.path.join(SCRIPT_DIR, "model.py"))
adaboost_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adaboost_module)
AdaBoost = adaboost_module.AdaBoost

# Load decision tree for reference
DT_SCRIPT = os.path.join(os.path.dirname(SCRIPT_DIR), '01_decision_tree')
dt_spec = importlib.util.spec_from_file_location("decision_tree", os.path.join(DT_SCRIPT, "model.py"))
dt_module = importlib.util.module_from_spec(dt_spec)
dt_spec.loader.exec_module(dt_module)


class TestAdaBoost:
    """AdaBoost分类器测试。"""

    def test_simple_binary(self):
        """测试简单二分类问题。"""
        np.random.seed(42)
        X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        y = np.array([0, 0, 0, 1])
        model = AdaBoost(n_estimators=50, learning_rate=1.0)
        model.fit(X, y)
        assert model.score(X, y) >= 0.75

    def test_predict_output_type(self):
        """测试predict返回类型和形状。"""
        np.random.seed(42)
        X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        y = np.array([0, 0, 0, 1])
        model = AdaBoost(n_estimators=10)
        model.fit(X, y)
        pred = model.predict(X)
        assert isinstance(pred, np.ndarray)
        assert pred.shape == (4,)

    def test_score_range(self):
        """测试score返回值在0-1之间。"""
        np.random.seed(42)
        X = np.random.rand(50, 4)
        y = np.random.randint(0, 2, 50)
        model = AdaBoost(n_estimators=10)
        model.fit(X, y)
        score = model.score(X, y)
        assert 0.0 <= score <= 1.0

    def test_fit_returns_self(self):
        """测试fit返回self。"""
        np.random.seed(42)
        X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        y = np.array([0, 0, 0, 1])
        model = AdaBoost(n_estimators=10)
        result = model.fit(X, y)
        assert result is model

    def test_multi_estimator_improvement(self):
        """测试增加estimator数量可提升准确率。"""
        np.random.seed(42)
        X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        y = np.array([0, 0, 0, 1])
        model5 = AdaBoost(n_estimators=5, learning_rate=1.0)
        model5.fit(X, y)
        model20 = AdaBoost(n_estimators=20, learning_rate=1.0)
        model20.fit(X, y)
        assert model20.score(X, y) >= model5.score(X, y)

    def test_random_binary_classification(self):
        """测试随机二分类数据上的表现。"""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        model = AdaBoost(n_estimators=20)
        model.fit(X, y)
        score = model.score(X, y)
        assert score >= 0.7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
