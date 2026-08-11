# 线性回归

## 历史背景

线性回归是最经典、最简单的监督学习算法之一。早在19世纪，高斯和勒让德就独立发展了最小二乘法（Least Squares Method），用于天体轨道计算。

**当时的问题**：
- 科学家需要拟合实验数据，找到变量之间的线性关系
- 没有系统的机器学习理论，主要是数学统计方法
- 计算能力有限，需要高效的算法

**核心突破**：
- 最小二乘法提供了求解线性回归的解析解
- 为后续所有回归问题奠定了基础

## 数学原理

### 模型定义

线性回归假设目标变量 $y$ 是输入特征 $\mathbf{x}$ 的线性组合：

$$y = \mathbf{w}^T\mathbf{x} + b$$

其中：
- $\mathbf{w}$ 是权重向量
- $b$ 是偏置项
- $\mathbf{x} = [x_1, x_2, ..., x_n]$ 是特征向量

### 损失函数

使用均方误差（MSE）作为损失函数：

$$J(\mathbf{w}, b) = \frac{1}{n}\sum_{i=1}^{n}(y_i - (\mathbf{w}^T\mathbf{x}_i + b))^2$$

### 求解方法

**方法一：解析解（正规方程）**

$$\mathbf{w} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$

**方法二：梯度下降**

$$\mathbf{w} := \mathbf{w} - \alpha \frac{\partial J}{\partial \mathbf{w}}$$

其中梯度为：

$$\frac{\partial J}{\partial \mathbf{w}} = -\frac{2}{n}\mathbf{X}^T(\mathbf{y} - \mathbf{X}\mathbf{w})$$

## 历史局限性

1. **只能捕捉线性关系**：无法处理非线性问题
2. **对异常值敏感**：MSE损失函数会被异常值影响
3. **假设特征独立**：忽略了特征之间的相关性
4. **需要特征缩放**：不同量纲的特征会影响收敛速度

## 历史影响

线性回归直接启发了：
- 逻辑回归（二分类）
- 岭回归和Lasso回归（正则化）
- 广义线性模型
- 后续所有回归方法的理论基础

## 参考文献

- Higham, N. J. (2002). Accuracy and Stability of Numerical Algorithms. SIAM.
- Draper, N. R., & Smith, H. (1998). Applied Regression Analysis. Wiley.

## 思考题

1. 为什么需要特征缩放？
2. 正则化如何改善线性回归？
3. 岭回归和Lasso回归有什么区别？