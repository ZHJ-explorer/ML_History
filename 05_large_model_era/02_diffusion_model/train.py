"""扩散模型训练脚本。"""
import sys
import os
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)

from model import SimpleDiffusionModel


def demo():
    print("=" * 50)
    print("扩散模型演示")
    print("=" * 50)
    
    model = SimpleDiffusionModel(input_dim=64, hidden_size=128, num_steps=10)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试前向传播
    x = torch.randn(2, 64)
    t = torch.randint(0, 10, (2,))
    predicted_noise, true_noise = model(x, t)
    print(f"输入shape: {x.shape}")
    print(f"预测噪声shape: {predicted_noise.shape}")
    print(f"真实噪声shape: {true_noise.shape}")


if __name__ == "__main__":
    demo()