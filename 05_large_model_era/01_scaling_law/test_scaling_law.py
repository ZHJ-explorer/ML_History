"""Scaling Law pytest测试。"""
import numpy as np
import pytest
from model import ScalingLaw


class TestScalingLaw:
    """缩放定律测试类。"""
    
    def test_fit_and_predict(self):
        """测试拟合和预测功能。"""
        np.random.seed(42)
        N = np.logspace(6, 9, 20)
        L = 1.0 * N**(-0.25) + 0.1 + np.random.randn(len(N)) * 0.05
        
        law = ScalingLaw()
        law.fit(N, L)
        
        # 预测结果应该是一个数组
        predicted_L = law.predict(N)
        assert predicted_L.shape == L.shape
        assert np.all(np.isfinite(predicted_L))
        
    def test_parameters_finite(self):
        """测试参数有限性。"""
        N = np.array([1e6, 1e7, 1e8, 1e9])
        L = np.array([0.5, 0.4, 0.3, 0.2])
        
        law = ScalingLaw()
        law.fit(N, L)
        
        assert np.isfinite(law.a)
        assert np.isfinite(law.alpha)
        assert np.isfinite(law.L_infinity)
        assert law.alpha > 0  # alpha应该为正
    
    def test_predict_monotonic(self):
        """测试预测单调性。"""
        N = np.array([1e6, 1e7, 1e8, 1e9])
        L = np.array([0.5, 0.4, 0.3, 0.2])
        
        law = ScalingLaw()
        law.fit(N, L)
        
        # 更大的参数量应该产生更小的损失
        predicted = law.predict(N)
        assert predicted[0] > predicted[-1]
    
    def test_optimal_n_params(self):
        """测试最优参数量计算。"""
        N = np.array([1e6, 1e7, 1e8, 1e9])
        L = np.array([0.5, 0.4, 0.3, 0.2])
        
        law = ScalingLaw()
        law.fit(N, L)
        
        # 目标损失高于L_infinity，应返回有限值
        optimal_N = law.optimal_n_params(0.25)
        assert np.isfinite(optimal_N)
        assert optimal_N > 0
