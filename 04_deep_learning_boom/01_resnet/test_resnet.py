"""ResNet模型测试。"""
import pytest
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import resnet18

class TestResNet:
    def test_model_creation(self):
        model = resnet18()
        assert model is not None
        # 测试前向传播
        x = torch.randn(2, 3, 32, 32)
        output = model(x)
        assert output.shape == (2, 10)

    def test_parameters(self):
        model = resnet18()
        params = sum(p.numel() for p in model.parameters())
        assert params > 1000000  # 至少1M参数

if __name__ == "__main__":
    pytest.main([__file__, "-v"])