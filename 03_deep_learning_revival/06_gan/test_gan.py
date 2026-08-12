"""GAN 生成对抗网络测试。"""
import sys
import os
import pytest
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import Generator, Discriminator, GAN


class TestGenerator:
    """生成器测试类。"""

    def test_init(self):
        """测试生成器初始化。"""
        gen = Generator(latent_dim=10, img_dim=20)
        assert gen.latent_dim == 10
        assert len(gen.model) == 6  # Linear, ReLU, Linear, ReLU, Linear, Tanh

    def test_forward_shape(self):
        """测试生成器输出形状。"""
        gen = Generator(latent_dim=10, img_dim=20)
        z = torch.rand(4, 10)
        output = gen(z)
        assert output.shape == (4, 20)

    def test_forward_output_range(self):
        """测试生成器输出在Tanh范围内。"""
        gen = Generator(latent_dim=10, img_dim=20)
        z = torch.rand(4, 10)
        output = gen(z)
        assert torch.all(output >= -1) and torch.all(output <= 1)

    def test_forward_different_batch_sizes(self):
        """测试不同批量大小。"""
        gen = Generator(latent_dim=8, img_dim=16)
        for batch_size in [1, 4, 16]:
            z = torch.rand(batch_size, 8)
            output = gen(z)
            assert output.shape == (batch_size, 16)

    def test_deterministic(self):
        """测试生成器的确定性。"""
        gen = Generator(latent_dim=10, img_dim=20)
        z = torch.rand(4, 10)
        out1 = gen(z)
        out2 = gen(z)
        assert torch.allclose(out1, out2)


class TestDiscriminator:
    """判别器测试类。"""

    def test_init(self):
        """测试判别器初始化。"""
        disc = Discriminator(img_dim=20)
        assert len(disc.model) == 6  # Linear, LeakyReLU, Linear, LeakyReLU, Linear, Sigmoid

    def test_forward_shape(self):
        """测试判别器输出形状。"""
        disc = Discriminator(img_dim=20)
        x = torch.rand(4, 20)
        output = disc(x)
        assert output.shape == (4, 1)

    def test_forward_output_range(self):
        """测试判别器输出在[0,1]范围内（Sigmoid）。"""
        disc = Discriminator(img_dim=20)
        x = torch.rand(4, 20)
        output = disc(x)
        assert torch.all(output >= 0) and torch.all(output <= 1)

    def test_forward_extreme_values(self):
        """测试极端输入下的输出。"""
        disc = Discriminator(img_dim=10)
        # 全零输入
        x_zero = torch.zeros(2, 10)
        out_zero = disc(x_zero)
        assert torch.all(out_zero >= 0) and torch.all(out_zero <= 1)

        # 大值输入
        x_large = torch.ones(2, 10) * 100
        out_large = disc(x_large)
        assert torch.all(out_large >= 0) and torch.all(out_large <= 1)


class TestGAN:
    """GAN模型测试类。"""

    def test_init(self):
        """测试GAN初始化。"""
        gan = GAN(latent_dim=10, img_dim=20)
        assert isinstance(gan.generator, Generator)
        assert isinstance(gan.discriminator, Discriminator)
        assert gan.generator.latent_dim == 10
        assert gan.discriminator.model[0].in_features == 20

    def test_generator_forward(self):
        """测试生成器前向传播。"""
        gan = GAN(latent_dim=10, img_dim=20)
        z = torch.rand(4, 10)
        fake_data = gan.generator(z)
        assert fake_data.shape == (4, 20)

    def test_discriminator_forward_real(self):
        """测试判别器对真实样本的判断。"""
        gan = GAN(latent_dim=10, img_dim=20)
        real_data = torch.rand(4, 20)
        output = gan.discriminator(real_data)
        assert output.shape == (4, 1)

    def test_discriminator_forward_fake(self):
        """测试判别器对生成样本的判断。"""
        gan = GAN(latent_dim=10, img_dim=20)
        z = torch.rand(4, 10)
        fake_data = gan.generator(z)
        output = gan.discriminator(fake_data)
        assert output.shape == (4, 1)

    def test_full_pipeline(self):
        """测试完整的前向传播流程。"""
        gan = GAN(latent_dim=10, img_dim=20)
        z = torch.rand(8, 10)

        # 生成 fake data
        fake_data = gan.generator(z)
        assert fake_data.shape == (8, 20)

        # 判别器判断
        real_output = gan.discriminator(fake_data)  # 这里用fake data测试判别器
        assert real_output.shape == (8, 1)

    def test_discriminator_loss(self):
        """测试判别器BCELoss计算。"""
        gan = GAN(latent_dim=10, img_dim=20)
        criterion = nn.BCELoss()

        batch_size = 4
        real_data = torch.rand(batch_size, 20)
        fake_data = gan.generator(torch.rand(batch_size, 10))

        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        real_output = gan.discriminator(real_data)
        fake_output = gan.discriminator(fake_data.detach())

        real_loss = criterion(real_output, real_labels)
        fake_loss = criterion(fake_output, fake_labels)

        assert real_loss.item() >= 0
        assert fake_loss.item() >= 0

    def test_generator_loss(self):
        """测试生成器BCELoss计算。"""
        gan = GAN(latent_dim=10, img_dim=20)
        criterion = nn.BCELoss()

        batch_size = 4
        z = torch.rand(batch_size, 10)
        fake_data = gan.generator(z)
        fake_labels = torch.ones(batch_size, 1)

        fake_output = gan.discriminator(fake_data)
        g_loss = criterion(fake_output, fake_labels)

        assert g_loss.item() >= 0
        assert not torch.isnan(g_loss)

    def test_training_step_generator(self):
        """测试生成器训练步骤。"""
        gan = GAN(latent_dim=10, img_dim=20)
        optimizer_g = torch.optim.Adam(gan.generator.parameters(), lr=0.001)
        criterion = nn.BCELoss()

        batch_size = 4
        z = torch.rand(batch_size, 10)
        fake_data = gan.generator(z)
        fake_labels = torch.ones(batch_size, 1)

        optimizer_g.zero_grad()
        fake_output = gan.discriminator(fake_data)
        g_loss = criterion(fake_output, fake_labels)
        g_loss.backward()
        optimizer_g.step()

        assert g_loss.item() > 0
        assert not torch.isnan(g_loss)

    def test_training_step_discriminator(self):
        """测试判别器训练步骤。"""
        gan = GAN(latent_dim=10, img_dim=20)
        optimizer_d = torch.optim.Adam(gan.discriminator.parameters(), lr=0.001)
        criterion = nn.BCELoss()

        batch_size = 4
        real_data = torch.rand(batch_size, 20)
        z = torch.rand(batch_size, 10)
        fake_data = gan.generator(z).detach()

        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        optimizer_d.zero_grad()
        real_output = gan.discriminator(real_data)
        fake_output = gan.discriminator(fake_data)
        d_loss = criterion(real_output, real_labels) + criterion(fake_output, fake_labels)
        d_loss.backward()
        optimizer_d.step()

        assert d_loss.item() > 0
        assert not torch.isnan(d_loss)

    def test_adversarial_training_round(self):
        """测试完整的对抗训练一轮。"""
        gan = GAN(latent_dim=10, img_dim=20)
        optimizer_g = torch.optim.Adam(gan.generator.parameters(), lr=0.001)
        optimizer_d = torch.optim.Adam(gan.discriminator.parameters(), lr=0.001)
        criterion = nn.BCELoss()

        batch_size = 8
        real_data = torch.rand(batch_size, 20)

        # 训练判别器
        optimizer_d.zero_grad()
        real_labels = torch.ones(batch_size, 1)
        real_output = gan.discriminator(real_data)
        real_loss = criterion(real_output, real_labels)

        z = torch.rand(batch_size, 10)
        fake_data = gan.generator(z).detach()
        fake_labels = torch.zeros(batch_size, 1)
        fake_output = gan.discriminator(fake_data)
        fake_loss = criterion(fake_output, fake_labels)

        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_d.step()

        # 训练生成器
        optimizer_g.zero_grad()
        z = torch.rand(batch_size, 10)
        fake_data = gan.generator(z)
        fake_labels = torch.ones(batch_size, 1)
        fake_output = gan.discriminator(fake_data)
        g_loss = criterion(fake_output, fake_labels)
        g_loss.backward()
        optimizer_g.step()

        assert d_loss.item() > 0
        assert g_loss.item() > 0
        assert not torch.isnan(d_loss)
        assert not torch.isnan(g_loss)

    def test_parameter_sharing(self):
        """测试生成器和判别器参数独立。"""
        gan = GAN(latent_dim=10, img_dim=20)
        gen_params = sum(p.numel() for p in gan.generator.parameters())
        disc_params = sum(p.numel() for p in gan.discriminator.parameters())
        total_params = sum(p.numel() for p in gan.parameters())
        assert total_params == gen_params + disc_params

    def test_gradient_flow(self):
        """测试梯度可以正常流动。"""
        gan = GAN(latent_dim=10, img_dim=20)

        z = torch.rand(4, 10, requires_grad=True)
        fake_data = gan.generator(z)
        output = gan.discriminator(fake_data)
        loss = output.sum()

        loss.backward()
        assert z.grad is not None
        assert z.grad.shape == z.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
