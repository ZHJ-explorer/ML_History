"""DBN训练脚本。"""
import sys
import os
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from model import DBN
from common.utils import load_iris_data, train_test_split, standardize


def demo():
    print("=" * 50)
    print("DBN演示")
    print("=" * 50)
    X, y = load_iris_data()
    X = X[:100]  # 只用前两类
    y = y[:100]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train_std, X_test_std, _, _ = standardize(X_train, X_test)

    # 转换为torch张量
    X_train_tensor = torch.FloatTensor(X_train_std)

    # 构建DBN
    model = DBN(layer_sizes=[4, 10, 5])
    model.pretrain(X_train_tensor, epochs=10, lr=0.01)

    print("DBN预训练完成")
    print(f"特征维度: {X_train_std.shape[1]} -> 10 -> 5")


if __name__ == "__main__":
    demo()