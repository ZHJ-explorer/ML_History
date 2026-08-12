"""Word2Vec (CBOW & SkipGram) 词向量模型测试。"""
import sys
import os
import pytest
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import CBOW, SkipGram


class TestCBOW:
    """CBOW模型测试类。"""

    def test_init(self):
        """测试CBOW初始化。"""
        model = CBOW(vocab_size=100, embed_size=16)
        assert model.embed.num_embeddings == 100
        assert model.embed.embedding_dim == 16
        assert model.output.in_features == 16
        assert model.output.out_features == 100

    def test_forward_basic(self):
        """测试基本前向传播。"""
        model = CBOW(vocab_size=50, embed_size=8)
        # context_indices: (batch, context_size)
        context = torch.tensor([[1, 2, 3, 4]])
        output = model(context)
        assert output.shape == (1, 50)

    def test_forward_batch(self):
        """测试批量前向传播。"""
        model = CBOW(vocab_size=100, embed_size=16)
        context = torch.randint(0, 100, (8, 4))
        output = model(context)
        assert output.shape == (8, 100)

    def test_forward_different_context_sizes(self):
        """测试不同上下文大小。"""
        model = CBOW(vocab_size=50, embed_size=8)
        for ctx_size in [2, 4, 6]:
            context = torch.randint(0, 50, (4, ctx_size))
            output = model(context)
            assert output.shape == (4, 50)

    def test_loss_computation(self):
        """测试交叉熵损失计算。"""
        model = CBOW(vocab_size=50, embed_size=8)
        criterion = nn.CrossEntropyLoss()
        context = torch.randint(0, 50, (4, 4))
        target = torch.randint(0, 50, (4,))
        output = model(context)
        loss = criterion(output, target)
        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_training_step(self):
        """测试一个训练步骤。"""
        model = CBOW(vocab_size=50, embed_size=8)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        context = torch.randint(0, 50, (8, 4))
        target = torch.randint(0, 50, (8,))

        optimizer.zero_grad()
        output = model(context)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_embedding_output(self):
        """测试词嵌入输出维度正确。"""
        model = CBOW(vocab_size=100, embed_size=32)
        context = torch.randint(0, 100, (2, 4))
        output = model(context)
        assert output.shape == (2, 100)

    def test_deterministic(self):
        """测试确定性前向传播。"""
        model = CBOW(vocab_size=50, embed_size=8)
        context = torch.randint(0, 50, (4, 4))
        out1 = model(context)
        out2 = model(context)
        assert torch.allclose(out1, out2)


class TestSkipGram:
    """SkipGram模型测试类。"""

    def test_init(self):
        """测试SkipGram初始化。"""
        model = SkipGram(vocab_size=100, embed_size=16)
        assert model.embed.num_embeddings == 100
        assert model.embed.embedding_dim == 16
        assert model.output.in_features == 16
        assert model.output.out_features == 100

    def test_forward_basic(self):
        """测试基本前向传播。"""
        model = SkipGram(vocab_size=50, embed_size=8)
        # center_indices: (batch,)
        center = torch.tensor([1, 2, 3])
        output = model(center)
        assert output.shape == (3, 50)

    def test_forward_batch(self):
        """测试批量前向传播。"""
        model = SkipGram(vocab_size=100, embed_size=16)
        center = torch.randint(0, 100, (8,))
        output = model(center)
        assert output.shape == (8, 100)

    def test_loss_computation(self):
        """测试交叉熵损失计算。"""
        model = SkipGram(vocab_size=50, embed_size=8)
        criterion = nn.CrossEntropyLoss()
        center = torch.randint(0, 50, (4,))
        target = torch.randint(0, 50, (4,))
        output = model(center)
        loss = criterion(output, target)
        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_training_step(self):
        """测试一个训练步骤。"""
        model = SkipGram(vocab_size=50, embed_size=8)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        center = torch.randint(0, 50, (8,))
        target = torch.randint(0, 50, (8,))

        optimizer.zero_grad()
        output = model(center)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_deterministic(self):
        """测试确定性前向传播。"""
        model = SkipGram(vocab_size=50, embed_size=8)
        center = torch.randint(0, 50, (4,))
        out1 = model(center)
        out2 = model(center)
        assert torch.allclose(out1, out2)

    def test_single_word(self):
        """测试单个词的前向传播。"""
        model = SkipGram(vocab_size=50, embed_size=8)
        center = torch.tensor([5])
        output = model(center)
        assert output.shape == (1, 50)


class TestWord2VecIntegration:
    """Word2Vec集成测试类。"""

    def test_cbow_simple_training(self):
        """测试CBOW在简单数据上的训练。"""
        torch.manual_seed(42)
        model = CBOW(vocab_size=20, embed_size=4)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        criterion = nn.CrossEntropyLoss()

        # 简单数据：用上下文预测中心词
        contexts = torch.tensor([
            [0, 1, 2, 3],
            [1, 2, 3, 4],
            [2, 3, 4, 5],
        ])
        targets = torch.tensor([2, 3, 4])

        initial_loss = None
        for epoch in range(10):
            optimizer.zero_grad()
            output = model(contexts)
            loss = criterion(output, targets)
            if epoch == 0:
                initial_loss = loss.item()
            loss.backward()
            optimizer.step()

        # 损失应该下降
        assert loss.item() < initial_loss

    def test_skipgram_simple_training(self):
        """测试SkipGram在简单数据上的训练。"""
        torch.manual_seed(42)
        model = SkipGram(vocab_size=20, embed_size=4)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        criterion = nn.CrossEntropyLoss()

        centers = torch.tensor([2, 3, 4])
        targets = torch.tensor([1, 2, 3])

        initial_loss = None
        for epoch in range(10):
            optimizer.zero_grad()
            output = model(centers)
            loss = criterion(output, targets)
            if epoch == 0:
                initial_loss = loss.item()
            loss.backward()
            optimizer.step()

        assert loss.item() < initial_loss


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
