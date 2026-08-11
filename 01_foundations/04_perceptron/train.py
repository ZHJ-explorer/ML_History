"""感知机训练脚本。"""
import sys
import os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from model import Perceptron

def demo():
    print("=" * 50)
    print("感知机演示")
    print("=" * 50)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    model = Perceptron(learning_rate=0.01, max_iter=10)
    for name, y in [("AND", [0, 0, 0, 1]), ("OR", [0, 1, 1, 1])]:
        model.fit(X, np.array(y))
        print(f"{name}门准确率: {model.score(X, np.array(y)):.2f}")
    print("XOR问题线性不可分，感知机无法解决")

if __name__ == "__main__":
    demo()