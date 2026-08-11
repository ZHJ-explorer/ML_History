"""决策树训练脚本。"""
import sys
import os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from model import DecisionTree
from common.utils import load_iris_data, train_test_split, standardize

def demo():
    print("=" * 50)
    print("决策树演示")
    print("=" * 50)
    X, y = load_iris_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train_std, X_test_std, _, _ = standardize(X_train, X_test)
    model = DecisionTree(max_depth=5)
    model.fit(X_train_std, y_train)
    print(f"准确率: {model.score(X_test_std, y_test):.4f}")

if __name__ == "__main__":
    demo()