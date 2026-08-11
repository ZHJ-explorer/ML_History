"""Word2Vec训练脚本。"""
import sys
import os
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from model import CBOW, SkipGram


def demo():
    """演示Word2Vec训练。"""
    print("=" * 50)
    print("Word2Vec演示：CBOW模型")
    print("=" * 50)
    
    # 简单语料
    corpus = [
        "the cat sat on the mat".split(),
        "the dog played in the park".split(),
        "the cat and the dog are friends".split(),
        "the mat was soft and warm".split(),
    ]
    
    # 构建词汇表
    word2idx = {}
    idx2word = {}
    for sentence in corpus:
        for word in sentence:
            if word not in word2idx:
                word2idx[word] = len(word2idx)
                idx2word[len(word2idx)-1] = word
    
    vocab_size = len(word2idx)
    embed_size = 8
    context_size = 2
    
    # 创建训练数据
    def create_cbow_data(corpus, word2idx, context_size):
        data = []
        for sentence in corpus:
            for i in range(context_size, len(sentence) - context_size):
                center = word2idx[sentence[i]]
                context = []
                for j in range(i - context_size, i + context_size + 1):
                    if j != i:
                        context.append(word2idx[sentence[j]])
                data.append((torch.tensor(context), torch.tensor(center)))
        return data
    
    train_data = create_cbow_data(corpus, word2idx, context_size)
    
    # 构建模型
    model = CBOW(vocab_size, embed_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    
    # 训练
    for epoch in range(100):
        total_loss = 0
        for context, center in train_data:
            optimizer.zero_grad()
            output = model(context.unsqueeze(0))
            loss = criterion(output, center.unsqueeze(0))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/100, Loss: {total_loss:.4f}")
    
    print(f"\n词汇表大小: {vocab_size}")
    print(f"词向量维度: {embed_size}")
    print("训练完成！")


if __name__ == "__main__":
    demo()