# Transformer

**论文**: *Attention Is All You Need* — Vaswani et al., 2017  
**机构**: Google Research  
**类型**: 注意力机制 / 序列到序列模型  

---

## 概述

Transformer 是深度学习历史上最具影响力的架构之一，它完全基于自注意力机制（Self-Attention），抛弃了传统的循环和卷积结构。这一设计使模型能够并行处理整个序列，在机器翻译等任务上取得了前所未有的性能突破，并奠定了现代大语言模型（GPT、BERT 等）的基础。

## 核心创新

- **自注意力机制（Self-Attention）**：让序列中每个位置都能直接关注其他所有位置，捕获长距离依赖
- **多头注意力（Multi-Head Attention）**：并行运行多组注意力，捕捉不同类型的语义关系
- **位置编码（Positional Encoding）**：用正弦/余弦函数注入序列顺序信息，替代 RNN 的顺序处理
- **全并行化**：训练速度远超 RNN/LSTM，可充分利用 GPU 并行计算

## 架构要点

```
输入 Embedding + 位置编码
        ↓
┌─────────────────┐
│   Encoder Stack  │ × N 层（默认 6）
│  ├─ Multi-Head   │
│  │  Self-Attn   │
│  ├─ Add & Norm   │
│  ├─ Position-    │
│  │  Wise FFN    │
│  └─ Add & Norm   │
└─────────────────┘
        ↓
输出投影层 → 词表概率分布
```

## 关键超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| d_model | 512 | 模型维度 |
| num_heads | 8 | 注意力头数 |
| num_layers | 6 | Encoder/Decoder 层数 |
| d_ff | 2048 | 前馈网络隐藏维度 |
| max_seq_len | 512 | 最大序列长度 |

## 历史意义

Transformer 在 WMT 2014 英德翻译任务上达到 28.4 BLEU，超越当时所有已有模型。此后，BERT（2018）、GPT 系列（2018-）等里程碑工作均建立在 Transformer 架构之上，开启了 NLP 的大模型时代。
