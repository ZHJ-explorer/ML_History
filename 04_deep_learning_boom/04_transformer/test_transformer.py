"""Transformer模型测试。"""
import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import Transformer, MultiHeadAttention, PositionWiseFeedForward, EncoderLayer


class TestTransformer:
    def test_creation(self):
        model = Transformer(vocab_size=100, d_model=128, num_heads=4, num_layers=2)
        assert isinstance(model, Transformer)
        assert sum(p.numel() for p in model.parameters()) > 0

    def test_forward_pass(self):
        model = Transformer(vocab_size=100, d_model=128, num_heads=4, num_layers=2)
        x = torch.randint(0, 100, (2, 10))
        output = model(x)
        assert output.shape == (2, 10, 100)

    def test_different_seq_lengths(self):
        model = Transformer(vocab_size=50, d_model=64, num_heads=4, num_layers=2)
        for seq_len in [5, 10, 20]:
            x = torch.randint(0, 50, (2, seq_len))
            output = model(x)
            assert output.shape == (2, seq_len, 50)

    def test_different_batch_sizes(self):
        model = Transformer(vocab_size=100, d_model=128, num_heads=4, num_layers=2)
        for batch_size in [1, 4, 8]:
            x = torch.randint(0, 100, (batch_size, 10))
            output = model(x)
            assert output.shape == (batch_size, 10, 100)

    def test_multihead_attention(self):
        d_model = 64
        num_heads = 4
        mha = MultiHeadAttention(d_model, num_heads)
        x = torch.randn(2, 10, d_model)
        output = mha(x, x, x)
        assert output.shape == (2, 10, d_model)

    def test_positionwise_feedforward(self):
        d_model = 64
        d_ff = 256
        ff = PositionWiseFeedForward(d_model, d_ff)
        x = torch.randn(2, 10, d_model)
        output = ff(x)
        assert output.shape == (2, 10, d_model)

    def test_encoder_layer(self):
        d_model = 64
        num_heads = 4
        d_ff = 256
        layer = EncoderLayer(d_model, num_heads, d_ff)
        x = torch.randn(2, 10, d_model)
        output = layer(x)
        assert output.shape == (2, 10, d_model)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
