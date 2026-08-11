"""Transformer训练脚本。"""
import sys
import os
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from model import Transformer


def demo():
    print("=" * 50)
    print("Transformer演示")
    print("=" * 50)
    
    vocab_size = 100
    model = Transformer(vocab_size=vocab_size, d_model=128, num_heads=4, num_layers=2)
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试前向传播
    x = torch.randint(0, vocab_size, (2, 10))
    output = model(x)
    print(f"输入shape: {x.shape}")
    print(f"输出shape: {output.shape}")


if __name__ == "__main__":
    demo()