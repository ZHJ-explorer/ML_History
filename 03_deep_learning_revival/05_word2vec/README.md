# Word2Vec

## 历史背景

2013年，Mikolov等人提出了Word2Vec，这是一种高效的词向量学习方法。它包括两种架构：CBOW（连续词袋模型）和Skip-gram。Word2Vec的出现标志着深度学习在NLP领域的重大突破。

**当时的问题**：
- 传统词表示方法（如One-hot）稀疏且无法捕捉语义
- 词向量训练计算成本高

**核心突破**：
- 负采样（Negative Sampling）加速训练
- 层次Softmax优化

## 数学原理

### CBOW模型

给定上下文词，预测中心词：

$$P(w_t | w_{t-c}, ..., w_{t-1}, w_{t+1}, ..., w_{t+c})$$

### Skip-gram模型

给定中心词，预测上下文词：

$$P(w_{t-c}, ..., w_{t+ c} | w_t)$$

### 负采样

将多分类问题转化为二分类问题：

$$\log \sigma(v'_{w_O}^T v_{w_I}) + \sum_{i=1}^k \mathbb{E}_{w_i \sim P_w}[\log \sigma(-v'_{w_i}^T v_{w_I})]$$

## 历史局限性

1. 无法处理多义词
2. 需要大量数据
3. 静态词向量

## 历史影响

- 开创了词向量的新时代
- 启发了ELMo、BERT等上下文词向量
- 至今仍在许多任务中使用

## 思考题

1. CBOW和Skip-gram各有什么优缺点？
2. 负采样为什么能加速训练？
3. Word2Vec和BERT的词向量有什么区别？