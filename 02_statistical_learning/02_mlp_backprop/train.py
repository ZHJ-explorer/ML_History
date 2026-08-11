"""MLP训练脚本。"""
import sys
import os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from model import MLP
from common.utils import load_iris_data, train_test_split

def demo():
    print("=" * 50)
    print("MLP演示")
    print("=" * 50)
    X, y = load_iris_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    y_train_onehot = np.eye(3)[y_train]
    y_test_onehot = np.eye(3)[y_test]
    model = MLP(layer_sizes=[4, 10, 3], learning_rate=0.01, max_iter=1000)
    model.fit(X_train, y_train_onehot)
    print(f"准确率: {model.score(X_test, y_test_onehot):.4f}")

if __name__ == "__main__":
    demo()