"""DBN训练脚本。"""
import sys
import os
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)

from model import DBN


def load_iris_data():
    from sklearn.datasets import load_iris
    iris = load_iris()
    return iris.data[:100], iris.target[:100]


def demo():
    print("=" * 50)
    print("DBN演示")
    print("=" * 50)
    
    X, y = load_iris_data()
    X = X / X.max()  # 归一化
    
    # 构建DBN
    model = DBN(layer_sizes=[4, 10, 5])
    model.pretrain(torch.FloatTensor(X), epochs=10, lr=0.01)
    
    print("DBN预训练完成")
    print(f"特征维度: {X.shape[1]} -> 10 -> 5")


if __name__ == "__main__":
    demo()