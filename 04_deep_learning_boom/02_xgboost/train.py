"""XGBoost训练脚本。"""
import sys
import os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)

from model import XGBoost


def load_iris_data():
    """加载iris数据集。"""
    from sklearn.datasets import load_iris
    iris = load_iris()
    return iris.data[:100], iris.target[:100]


def train_test_split(X, y, test_size=0.2):
    """划分训练测试集。"""
    n = len(y)
    split = int(n * (1 - test_size))
    idx = np.random.RandomState(42).permutation(n)
    train_idx = idx[:split]
    test_idx = idx[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def demo():
    print("=" * 50)
    print("XGBoost演示")
    print("=" * 50)
    
    X, y = load_iris_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    model = XGBoost(n_estimators=10, max_depth=3, learning_rate=0.1)
    model.fit(X_train, y_train)
    
    print(f"准确率: {model.score(X_test, y_test):.4f}")


if __name__ == "__main__":
    demo()