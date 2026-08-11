"""BERT模型测试。"""
import pytest
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import BERT

class TestBERT:
    def test_model_creation(self):
        model = BERT(vocab_size=100, d_model=64, num_heads=4, num_layers=2)
        assert model is not None

    def test_forward(self):
        model = BERT(vocab_size=100, d_model=64, num_heads=4, num_layers=2)
        x = torch.randint(0, 100, (2, 10))
        mlm_out, nsp_out = model(x)
        assert mlm_out.shape == (2, 100)
        assert nsp_out.shape == (2, 2)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])