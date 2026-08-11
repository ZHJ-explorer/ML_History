"""线性回归训练脚本。"""
import sys
import os
import numpy as np

# 添加项目根目录到路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from model import LinearRegression
from common.utils import train_test_split, standardize

def demo_basic():
    print("=" * 50)
    print("基础演示：简单线性关系")
    print("=" * 50)
    np.random.seed(42)
    X = np.random.rand(100, 1) * 10
    y = 3 * X.squeeze() + 2 + np.random.randn(100) * 0.5
    model = LinearRegression()
    model.fit(X, y)
    r2 = model.score(X, y)
    print(f"真实关系: y = 3x + 2")
    print(f"学习到的关系: y = {model.weights[0]:.3f}x + {model.bias:.3f}")
    print(f"R²分数: {r2:.4f}")

def demo_gradient_descent():
    print("=" * 50)
    print("梯度下降演示")
    print("=" * 50)
    np.random.seed(42)
    X = np.random.rand(100, 2) * 10
    y = 2 * X[:, 0] + 3 * X[:, 1] + 1 + np.random.randn(100) * 0.5
    model = LinearRegression(learning_rate=0.01, max_iter=1000)
    model.fit(X, y, method="gradient_descent")
    print(f"学习到的权重: {model.weights}")
    print(f"学习到的偏置: {model.bias:.3f}")
    print(f"R²分数: {model.score(X, y):.4f}")

def demo_real_data():
    print("=" * 50)
    print("真实数据集演示")
    print("=" * 50)
    try:
        from sklearn.datasets import fetch_california_housing
        data = fetch_california_housing()
        X, y = data.data, data.target
    except ImportError:
        X = np.random.rand(100, 2)
        y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.randn(100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train_std, X_test_std, _, _ = standardize(X_train, X_test)
    model = LinearRegression()
    model.fit(X_train_std, y_train)
    print(f"R²分数: {model.score(X_test_std, y_test):.4f}")

if __name__ == "__main__":
    demo_basic()
    demo_gradient_descent()
    demo_real_data()
    print("训练完成！")