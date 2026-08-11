"""GBDT训练脚本。"""
import sys
import os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from model import GBDT
from common.utils import load_iris_data, train_test_split

def demo():
    print("=" * 50)
    print("GBDT演示")
    print("=" * 50)
    X, y = load_iris_data()
    X, y = X[:100], y[:100]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    y_train_bin = (y_train == 1).astype(int)
    y_test_bin = (y_test == 1).astype(int)
    model = GBDT(n_estimators=10, learning_rate=0.1)
    model.fit(X_train, y_train_bin)
    print(f"准确率: {model.score(X_test, y_test_bin):.4f}")

if __name__ == "__main__":
    demo()