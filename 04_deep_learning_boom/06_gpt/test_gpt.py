"""GPT模型测试。"""
import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import GPT1


class TestGPT1:
    def test_creation(self):
        model = GPT1(vocab_size=100, d_model=128, num_heads=4, num_layers=2)
        assert isinstance(model, GPT1)
        assert sum(p.numel() for p in model.parameters()) > 0

    def test_forward_pass(self):
        model = GPT1(vocab_size=100, d_model=128, num_heads=4, num_layers=2)
        input_ids = torch.randint(0, 100, (2, 10))
        output = model(input_ids)
        assert output.shape == (2, 10, 100)

    def test_different_seq_lengths(self):
        model = GPT1(vocab_size=50, d_model=64, num_heads=4, num_layers=2)
        for seq_len in [5, 10, 20]:
            input_ids = torch.randint(0, 50, (2, seq_len))
            output = model(input_ids)
            assert output.shape == (2, seq_len, 50)

    def test_different_batch_sizes(self):
        model = GPT1(vocab_size=100, d_model=128, num_heads=4, num_layers=2)
        for batch_size in [1, 4, 8]:
            input_ids = torch.randint(0, 100, (batch_size, 10))
            output = model(input_ids)
            assert output.shape == (batch_size, 10, 100)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
