"""ViT训练脚本。"""
import sys
import os
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)

from model import ViT


def demo():
    print("=" * 50)
    print("ViT演示")
    print("=" * 50)
    
    model = ViT(img_size=32, patch_size=4, in_channels=3, num_classes=10, embed_dim=64, num_heads=4)
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试前向传播
    x = torch.randn(2, 3, 32, 32)
    output = model(x)
    print(f"输入shape: {x.shape}")
    print(f"输出shape: {output.shape}")


if __name__ == "__main__":
    demo()