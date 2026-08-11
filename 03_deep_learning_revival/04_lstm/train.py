"""LSTM训练脚本。"""
import sys
import os
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from model import LSTM


def demo():
    """演示LSTM序列预测。"""
    print("=" * 50)
    print("LSTM演示：序列预测")
    print("=" * 50)
    
    # 创建简单序列数据：预测下一个值
    np.random.seed(42)
    seq_len = 500
    X = np.random.rand(seq_len, 1)
    y = np.roll(X, -1).flatten()[:-1]  # 下一个值
    
    # 划分为训练和测试
    split = int(0.8 * seq_len)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # 创建序列
    def create_sequences(data, seq_length=10):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length].flatten())
            y.append(data[i+seq_length].flatten())
        return np.array(X), np.array(y)
    
    seq_length = 10
    X_train_seq, y_train_seq = create_sequences(X_train, seq_length)
    X_test_seq, y_test_seq = create_sequences(X_test, seq_length)
    
    # 转换为torch张量
    X_train_tensor = torch.FloatTensor(X_train_seq).unsqueeze(-1)  # (seq, batch, input)
    y_train_tensor = torch.FloatTensor(y_train_seq)
    X_test_tensor = torch.FloatTensor(X_test_seq).unsqueeze(-1)
    y_test_tensor = torch.FloatTensor(y_test_seq)
    
    # 构建模型
    model = LSTM(input_size=1, hidden_size=32, num_layers=2, output_size=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    
    # 训练
    model.train()
    for epoch in range(50):
        optimizer.zero_grad()
        # LSTM期望输入 (seq_len, batch, input_size)
        output = model(X_train_tensor)  # (batch, output_size)
        loss = criterion(output.squeeze(), y_train_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/50, Loss: {loss.item():.6f}")
    
    # 测试
    model.eval()
    with torch.no_grad():
        test_output = model(X_test_tensor)
        test_loss = criterion(test_output.squeeze(), y_test_tensor)
        print(f"Test Loss: {test_loss.item():.6f}")


if __name__ == "__main__":
    demo()