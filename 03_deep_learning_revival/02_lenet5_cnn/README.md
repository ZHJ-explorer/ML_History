# LeNet-5 卷积神经网络

## 历史背景

1998年，Yann LeCun等在论文《Gradient-based learning applied to document recognition》中提出LeNet-5，
这是最早成功的卷积神经网络之一，用于手写数字识别。

**核心创新**：
- 卷积层：局部连接 + 权值共享
- 池化层：空间降采样，增加平移不变性
- 层次化特征提取

**当时背景**：
- 深度学习尚未复兴
- 训练数据有限（MNIST）
- 计算能力较弱

## 数学原理

### 卷积运算

二维卷积公式：
$$(f * g)[i, j] = \sum_m \sum_n f[m, n] \cdot g[i-m, j-n]$$

代码对应：
```python
# ManualConv2D.forward()
output[b, o, i, j] += (
    x[b, c, h_start:h_end, w_start:w_end]
    * self.weight[o, c]
).sum()
```

### 平均池化

$$\text{output}[i, j] = \frac{1}{k^2} \sum_{m,n \in \text{window}} \text{input}[i \cdot s + m, j \cdot s + n]$$

代码对应：
```python
# ManualAvgPool2D.forward()
output[b, c, i, j] = x[b, c, h_start:h_end, w_start:w_end].mean()
```

### 网络结构

```
Input(1×32×32)
    ↓ Conv1(6×5×5, stride=1)
    ↓ AvgPool(2×2, stride=2)
    ↓ ReLU
    ↓ Conv2(16×5×5, stride=1)
    ↓ AvgPool(2×2, stride=2)
    ↓ ReLU
    ↓ Flatten
    ↓ FC1(120) + ReLU
    ↓ FC2(84) + ReLU
    ↓ FC3(10)
```

## 历史影响

- 开创了卷积神经网络的研究方向
- 启发了后续LeNet系列、AlexNet、VGG等
- 权值共享思想成为CNN核心

## 思考题

1. 为什么卷积层需要权值共享？
2. 池化层的作用是什么？
3. LeNet-5与现代ResNet有什么本质区别？
