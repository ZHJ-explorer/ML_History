"""缩放定律训练脚本。"""
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)

from model import ScalingLaw


def demo():
    print("=" * 50)
    print("缩放定律演示")
    print("=" * 50)
    
    import numpy as np
    np.random.seed(42)
    
    # 模拟数据
    N = np.logspace(6, 9, 20)
    L = 1.0 * N**(-0.25) + 0.1 + np.random.randn(len(N)) * 0.05
    
    # 拟合
    law = ScalingLaw()
    law.fit(N, L)
    
    print(f"拟合参数: a={law.a:.4f}, alpha={law.alpha:.4f}")
    print("缩放定律预测完成！")


if __name__ == "__main__":
    demo()