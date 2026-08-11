"""ResNet模型测试。"""
import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import ResNet, BasicBlock, resnet18


class TestResNet:
    def test_resnet18_creation(self):
        model = resnet18()
        assert isinstance(model, ResNet)
        assert sum(p.numel() for p in model.parameters()) > 0

    def test_forward_pass(self):
        model = resnet18()
        x = torch.randn(2, 3, 32, 32)
        output = model(x)
        assert output.shape == (2, 10)

    def test_forward_pass_cifar(self):
        model = resnet18()
        x = torch.randn(4, 3, 32, 32)
        output = model(x)
        assert output.shape == (4, 10)

    def test_different_batch_sizes(self):
        model = resnet18()
        for batch_size in [1, 4, 8]:
            x = torch.randn(batch_size, 3, 32, 32)
            output = model(x)
            assert output.shape == (batch_size, 10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
