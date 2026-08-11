"""ResNet训练脚本。"""
import sys
import os
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from model import resnet18


def demo():
    print("=" * 50)
    print("ResNet演示")
    print("=" * 50)
    
    model = resnet18()
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(model)


if __name__ == "__main__":
    demo()