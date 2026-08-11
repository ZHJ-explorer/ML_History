# ResNet — 残差网络

**作者**: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (Microsoft Research)  
**年份**: 2015  
**论文**: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)  
**会议**: CVPR 2016  
**框架**: PyTorch

---

## 简介

ResNet（残差网络）是计算机视觉领域的里程碑式工作，通过引入**跳跃连接（skip connection）**解决了深层神经网络的退化问题（degradation problem），使得训练数百甚至上千层的网络成为可能。

在 ResNet 之前，研究人员发现当网络深度增加到一定程度后，准确率会饱和，然后迅速下降——这并非过拟合，而是梯度消失/爆炸导致的训练困难。ResNet 的核心思想是让网络学习**残差函数 F(x) = H(x) - x**，而非直接学习目标函数 H(x)。这样，恒等映射就成为一个可学习的选项，网络可以更轻松地保持信息流通。

---

## 核心创新

### 残差块（Residual Block）

ResNet 的基本构建单元是残差块，其核心公式为：

```
y = F(x, {W_i}) + x
```

其中：
- `x` 是输入
- `F(x, {W_i})` 是残差函数（通常由 2-3 个卷积层构成）
- `+ x` 是恒等跳跃连接（identity shortcut）
- `y` 是输出，经过激活函数（ReLU）处理

### 两种残差块类型

1. **Basic Block**（ResNet-18/34）：
   - 两个 3×3 卷积层
   - 每层后跟 Batch Normalization 和 ReLU
   - 跳跃连接在维度不匹配时添加 1×1 卷积进行升维

2. **Bottleneck Block**（ResNet-50/101/152）：
   - 1×1 卷积（降维）→ 3×3 卷积（保持）→ 1×1 卷积（升维）
   - 减少计算量，同时保持表达能力

### 网络架构

| 变体 | 层数 | 参数 | Top-5 错误率 |
|------|------|------|--------------|
| ResNet-18 | 18 | 11.7M | 5.8% |
| ResNet-34 | 34 | 21.8M | 4.7% |
| ResNet-50 | 50 | 23.9M | 3.7% |
| ResNet-101 | 101 | 42.5M | 3.4% |
| ResNet-152 | 152 | 58.3M | 3.3% |

---

## 数学原理

### 残差学习

传统网络学习：`H(x) → y`  
ResNet 学习：`F(x) = H(x) - x → y = F(x) + x`

### 梯度传播

**无跳跃连接**：
```
∂L/∂x = ∂L/∂y · ∂y/∂x = ∂L/∂y · W_n · ... · W_2 · W_1
```
梯度需要穿越所有层，容易出现梯度消失。

**有跳跃连接**：
```
∂L/∂x = ∂L/∂y · (W_n · ... · W_1 + 1)
       = ∂L/∂y · W_n · ... · W_1 + ∂L/∂y
```
梯度有一条"高速通道"直接回传，确保信息不会在深层丢失。

---

## 影响与后续发展

ResNet 的提出彻底改变了计算机视觉的研究方向：
- 2015年 ImageNet 竞赛冠军，Top-5 错误率降至 3.6%
- 开启了"更深网络"的研究范式
- 残差思想被广泛应用于各种架构：DenseNet、SENet、Transformer 等
- 从 18 层到 152 层，甚至后来的 ResNeXt、Wide ResNet 等变体

---

## 参考

- He, K., Zhang, X., Ren, S., & Sun, J. (2015). Deep residual learning for image recognition. arXiv:1512.03385
- PyTorch 官方实现：`torchvision.models.resnet`
