"""DBN (Deep Belief Network) 测试。"""
import sys
import os
import pytest
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import DBN, RBM


class TestRBM:
    """受限玻尔兹曼机测试类。"""

    def test_rbm_init(self):
        """测试RBM初始化参数形状。"""
        rbm = RBM(n_visible=4, n_hidden=10)
        assert rbm.w.shape == (4, 10)
        assert rbm.hb.shape == (10,)
        assert rbm.vb.shape == (4,)
        assert torch.allclose(rbm.hb, torch.zeros(10))
        assert torch.allclose(rbm.vb, torch.zeros(4))

    def test_rbm_forward_shape(self):
        """测试RBM前向传播输出形状。"""
        rbm = RBM(n_visible=8, n_hidden=16)
        v = torch.rand(4, 8)  # batch_size=4, visible=8
        v_prob, h_prob = rbm(v)
        assert v_prob.shape == (4, 8)
        assert h_prob.shape == (4, 16)

    def test_rbm_forward_values_range(self):
        """测试RBM前向传播概率值在[0,1]范围内。"""
        rbm = RBM(n_visible=4, n_hidden=8)
        v = torch.rand(2, 4)
        v_prob, h_prob = rbm(v)
        assert torch.all(v_prob >= 0) and torch.all(v_prob <= 1)
        assert torch.all(h_prob >= 0) and torch.all(h_prob <= 1)

    def test_rbm_train_step(self):
        """测试RBM训练步骤不会报错且参数更新。"""
        rbm = RBM(n_visible=4, n_hidden=8)
        v = torch.rand(16, 4)
        initial_w = rbm.w.clone()
        rbm.train_step(v, lr=0.01)
        # 权重应有变化
        assert not torch.allclose(rbm.w, initial_w)


class TestDBN:
    """深度信念网络测试类。"""

    def test_dbn_init(self):
        """测试DBN初始化创建正确的RBM层数。"""
        model = DBN(layer_sizes=[4, 10, 5])
        assert len(model.rbms) == 2
        assert model.rbms[0].n_visible == 4
        assert model.rbms[0].n_hidden == 10
        assert model.rbms[1].n_visible == 10
        assert model.rbms[1].n_hidden == 5

    def test_dbn_forward_shape(self):
        """测试DBN前向传播输出形状与输入一致。"""
        model = DBN(layer_sizes=[4, 10, 5])
        x = torch.rand(8, 4)
        output = model.forward(x)
        assert output.shape == (8, 5)

    def test_dbn_pretrain(self):
        """测试DBN预训练运行不报错。"""
        X = torch.rand(50, 4)
        model = DBN(layer_sizes=[4, 10, 5])
        result = model.pretrain(X, epochs=3, lr=0.01)
        assert result is model  # 返回self

    def test_dbn_pretrain_output_shape(self):
        """测试预训练后输出形状。"""
        X = torch.rand(20, 4)
        model = DBN(layer_sizes=[4, 10, 5])
        model.pretrain(X, epochs=2, lr=0.01)
        output = model.forward(X)
        assert output.shape == (20, 5)

    def test_dbn_multi_layer(self):
        """测试多层DBN。"""
        model = DBN(layer_sizes=[8, 16, 8, 4])
        assert len(model.rbms) == 3
        X = torch.rand(16, 8)
        output = model.forward(X)
        assert output.shape == (16, 4)

    def test_dbn_zero_input_deterministic(self):
        """测试DBN全零输入下，初始化权重为零时输出确定性。"""
        # 创建一个权重全为零的DBN
        model = DBN(layer_sizes=[4, 8, 4])
        with torch.no_grad():
            for rbm in model.rbms:
                rbm.w.zero_()
                rbm.hb.zero_()
                rbm.vb.zero_()
        x = torch.zeros(4, 4)
        out1 = model.forward(x)
        out2 = model.forward(x)
        # 权重全零时，sigmoid(0)=0.5，bernoulli(0.5)仍是随机的
        # 但多次运行应保持相同随机性（如果固定seed）
        torch.manual_seed(42)
        out3 = model.forward(x)
        torch.manual_seed(42)
        out4 = model.forward(x)
        assert torch.allclose(out3, out4)
