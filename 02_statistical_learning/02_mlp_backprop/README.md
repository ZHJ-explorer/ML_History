# 多层感知机（MLP）与反向传播

## 历史背景

1986年，Rumelhart、McCelland和Williams在《Parallel Distributed Processing》一书中推广了反向传播算法，标志着神经网络研究的复兴。虽然反向传播的思想早在1970年代就由Werbos提出，但直到MLP的普及才被广泛接受。

**当时的问题**：
- 单层感知机无法解决XOR问题
- 没有有效的多隐层训练算法
- 计算能力不足

**核心突破**：
- 反向传播算法：高效计算梯度
- 链式法则：逐层传播误差

## 数学原理

### 前向传播

$$z^{[l]} = W^{[l]}a^{[l-1]} + b^{[l]}$$
$$a^{[l]} = \sigma(z^{[l]})$$

### 反向传播

$$\delta^{[l]} = (W^{[l+1]})^T \delta^{[l+1]} \odot \sigma'(z^{[l]})$$
$$\frac{\partial L}{\partial W^{[l]}} = \delta^{[l]} (a^{[l-1]})^T$$
$$\frac{\partial L}{\partial b^{[l]}} = \delta^{[l]}$$

### 梯度更新

$$W^{[l]} := W^{[l]} - \alpha \frac{\partial L}{\partial W^{[l]}}$$
$$b^{[l]} := b^{[l]} - \alpha \frac{\partial L}{\partial b^{[l]}}$$

## 历史局限性

1. 梯度消失问题：深层网络难以训练
2. 局部最优：梯度下降可能陷入局部极小值
3. 需要大量数据

## 历史影响

- 奠定了深度学习的基础
- 直接启发了现代深度学习框架

## 思考题

1. 为什么反向传播比数值微分更高效？
2. 梯度消失问题如何解决？
3. 为什么ReLU成为最常用的激活函数？