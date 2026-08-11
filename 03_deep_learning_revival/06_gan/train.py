"""GAN训练脚本。"""
import sys
import os
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from model import GAN


def demo():
    """演示GAN训练。"""
    print("=" * 50)
    print("GAN演示：生成简单数据")
    print("=" * 50)
    
    torch.manual_seed(42)
    
    # 超参数
    latent_dim = 10
    img_dim = 20  # 输出维度
    epochs = 50
    lr = 0.001
    
    # 创建简单数据：随机数据
    np.random.seed(42)
    real_data = torch.FloatTensor(np.random.randn(200, img_dim))  # (200, 20)
    
    # 构建模型
    gan = GAN(latent_dim, img_dim)
    optimizer_g = torch.optim.Adam(gan.generator.parameters(), lr=lr)
    optimizer_d = torch.optim.Adam(gan.discriminator.parameters(), lr=lr)
    
    # 训练
    print(f"训练 {epochs} 个epoch...")
    for epoch in range(epochs):
        batch_size = real_data.size(0)
        
        # 训练判别器
        optimizer_d.zero_grad()
        
        # 真实样本
        real_labels = torch.ones(batch_size, 1)
        real_output = gan.discriminator(real_data)
        real_loss = torch.nn.BCELoss()(real_output, real_labels)
        
        # 生成样本
        noise = torch.randn(batch_size, latent_dim)
        fake_data = gan.generator(noise)
        fake_labels = torch.zeros(batch_size, 1)
        fake_output = gan.discriminator(fake_data.detach())
        fake_loss = torch.nn.BCELoss()(fake_output, fake_labels)
        
        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_d.step()
        
        # 训练生成器
        optimizer_g.zero_grad()
        noise = torch.randn(batch_size, latent_dim)
        fake_data = gan.generator(noise)
        fake_labels = torch.ones(batch_size, 1)
        fake_output = gan.discriminator(fake_data)
        g_loss = torch.nn.BCELoss()(fake_output, fake_labels)
        g_loss.backward()
        optimizer_g.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, G Loss: {g_loss.item():.4f}, D Loss: {d_loss.item():.4f}")
    
    # 生成样本
    with torch.no_grad():
        noise = torch.randn(5, latent_dim)
        generated = gan.generator(noise)
        print(f"\n生成样本维度: {generated.shape}")
        print("训练完成！")


if __name__ == "__main__":
    demo()