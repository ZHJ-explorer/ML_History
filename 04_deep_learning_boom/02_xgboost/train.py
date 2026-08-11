"""XGBoost训练脚本。"""
import sys
import os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)

from model import XGBoost


def load_data():
    """创建简单的二分类数据。"""
    np.random.seed(42)
    n_samples = 200
    X = np.random.randn(n_samples, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


def demo():
    print("=" * 50)
    print("XGBoost演示")
    print("=" * 50)
    
    X, y = load_data()
    split = int(0.8 * len(y))
    idx = np.random.RandomState(42).permutation(len(y))
    X_train, X_test = X[idx[:split]], X[idx[split:]]
    y_train, y_test = y[idx[:split]], y[idx[split:]]
    
    model = XGBoost(n_estimators=10, max_depth=3, learning_rate=0.1)
    model.fit(X_train, y_train)
    
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print(f"训练集准确率: {train_acc:.4f}")
    print(f"测试集准确率: {test_acc:.4f}")


if __name__ == "__main__":
    demo()