# 循环神经网络（RNN）

## 历史背景

1986年，Sejnowski等人首次提出循环神经网络的概念。
1990年代，RNN被广泛应用于序列建模任务，但面临梯度消失问题。
2014年，Bengio等人提出RNN变体（LSTM、GRU）来解决梯度消失问题。

**核心问题**：
- 传统神经网络无法处理变长序列
- 缺乏记忆机制，无法捕捉时序依赖

**RNN的突破**：
- 引入隐藏状态，保留历史信息
- 共享权重，处理任意长度序列

## 数学原理

### 单步RNN

隐藏状态更新公式：
$$h_t = \tanh(W_{xh} \cdot x_t + W_{hh} \cdot h_{t-1} + b_h)$$

输出公式：
$$y_t = W_{hy} \cdot h_t + b_y$$

代码对应：
```python
# ManualRNNCell.forward()
h_t = torch.tanh(
    torch.matmul(x_t, self.w_xh) +
    torch.matmul(h_prev, self.w_hh) +
    self.b_h
)
```

### 展开计算图

RNN可以看作在时间维度上展开的深度网络：
```
x_0 -> [RNN] -> h_0 -> [FC] -> y_0
x_1 -> [RNN] -> h_1 -> [FC] -> y_1
x_2 -> [RNN] -> h_2 -> [FC] -> y_2
...
```

## 梯度消失问题

当序列较长时，反向传播的梯度会指数级衰减：
$$\frac{\partial L}{\partial W_{hh}} \propto (W_{hh})^T \cdot ... \cdot (W_{hh})^T$$

这导致早期输入的信息难以传递到后期。

**解决方案**：
- LSTM：引入门控机制
- GRU：简化门控
- 梯度裁剪

## 应用

- 文本分类
- 机器翻译
- 语音识别
- 时间序列预测

## 思考题

1. RNN和MLP的本质区别是什么？
2. 为什么RNN会出现梯度消失？
3. LSTM相比RNN做了什么改进？
