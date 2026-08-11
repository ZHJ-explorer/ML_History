"""BERT模型测试。"""
import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import BERT


class TestBERT:
    def test_creation(self):
        model = BERT(vocab_size=100, d_model=128, num_heads=4, num_layers=2)
        assert isinstance(model, BERT)
        assert sum(p.numel() for p in model.parameters()) > 0

    def test_forward_pass(self):
        model = BERT(vocab_size=100, d_model=128, num_heads=4, num_layers=2)
        input_ids = torch.randint(0, 100, (2, 10))
        mlm_output, nsp_output = model(input_ids)
        assert mlm_output.shape == (2, 100)
        assert nsp_output.shape == (2, 2)

    def test_with_token_type_ids(self):
        model = BERT(vocab_size=100, d_model=128, num_heads=4, num_layers=2)
        input_ids = torch.randint(0, 100, (2, 10))
        token_type_ids = torch.zeros(2, 10, dtype=torch.long)
        mlm_output, nsp_output = model(input_ids, token_type_ids)
        assert mlm_output.shape == (2, 100)
        assert nsp_output.shape == (2, 2)

    def test_different_seq_lengths(self):
        model = BERT(vocab_size=50, d_model=64, num_heads=4, num_layers=2)
        for seq_len in [5, 10, 20]:
            input_ids = torch.randint(0, 50, (2, seq_len))
            mlm_output, nsp_output = model(input_ids)
            assert mlm_output.shape == (2, 50)
            assert nsp_output.shape == (2, 2)

    def test_different_batch_sizes(self):
        model = BERT(vocab_size=100, d_model=128, num_heads=4, num_layers=2)
        for batch_size in [1, 4, 8]:
            input_ids = torch.randint(0, 100, (batch_size, 10))
            mlm_output, nsp_output = model(input_ids)
            assert mlm_output.shape == (batch_size, 100)
            assert nsp_output.shape == (batch_size, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
