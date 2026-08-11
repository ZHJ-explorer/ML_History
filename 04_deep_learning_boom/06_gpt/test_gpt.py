"""GPT-1模型测试。"""
import pytest
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import GPT1

class TestGPT1:
    def test_model_creation(self):
        model = GPT1(vocab_size=100, d_model=64, num_heads=4, num_layers=2)
        assert model is not None

    def test_forward(self):
        model = GPT1(vocab_size=100, d_model=64, num_heads=4, num_layers=2)
        x = torch.randint(0, 100, (2, 10))
        logits = model(x)
        assert logits.shape == (2, 10, 100)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])