"""缩放定律（Scaling Laws）实现。"""
import numpy as np
import matplotlib.pyplot as plt


class ScalingLaw:
    """深度学习模型的缩放定律。
    
    核心公式（Kaplan et al., 2020）：
        L(N) = a * N^(-alpha) + L_infinity
    
    其中：
    - L: 损失
    - N: 参数量
    - a, alpha, L_infinity: 拟合参数
    """
    
    def __init__(self):
        self.a = None
        self.alpha = None
        self.L_infinity = None
    
    def fit(self, N, L):
        """拟合缩放定律参数。
        
        Args:
            N: 参数量列表
            L: 对应损失列表
        """
        # 使用对数变换拟合
        log_N = np.log(N)
        log_L = np.log(L - np.min(L) + 1e-10)
        
        # 线性回归
        coeffs = np.polyfit(log_N, log_L, 1)
        self.alpha = -coeffs[0]
        self.a = np.exp(coeffs[1])
        self.L_infinity = np.min(L) * 0.5
        
        return self
    
    def predict(self, N):
        """预测给定参数量下的损失。"""
        return self.a * N**(-self.alpha) + self.L_infinity
    
    def optimal_n_params(self, target_loss, max_n_params=1e12):
        """计算达到目标损失所需的最优参数量。"""
        if target_loss <= self.L_infinity:
            return max_n_params
        return (self.a / (target_loss - self.L_infinity))**(1/self.alpha)


def demo():
    """演示缩放定律。"""
    print("=" * 50)
    print("缩放定律演示")
    print("=" * 50)
    
    # 模拟不同参数规模下的损失
    np.random.seed(42)
    N = np.logspace(6, 9, 20)  # 从1M到1B参数
    L = 1.0 * N**(-0.25) + 0.1 + np.random.randn(len(N)) * 0.05
    
    # 拟合
    law = ScalingLaw()
    law.fit(N, L)
    
    print(f"拟合参数:")
    print(f"  a = {law.a:.4f}")
    print(f"  alpha = {law.alpha:.4f}")
    print(f"  L_infinity = {law.L_infinity:.4f}")
    
    # 预测
    predicted_L = law.predict(N)
    print(f"\n预测损失范围: [{predicted_L.min():.4f}, {predicted_L.max():.4f}]")
    
    # 计算达到0.15损失所需参数量
    target_L = 0.15
    optimal_N = law.optimal_n_params(target_L)
    print(f"\n达到目标损失 {target_L} 所需参数量: {optimal_N:.2e}")
    
    print("\n缩放定律公式: L(N) = a * N^(-alpha) + L_infinity")
    print("含义: 参数量每增加10倍，损失约减少25%")


if __name__ == "__main__":
    demo()