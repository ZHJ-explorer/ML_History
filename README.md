# 机器学习发展史 - Learn by Building

以**时间线为主轴**，从零实现机器学习发展史上的里程碑模型。

## 核心理念

**Build it to understand it.**
不调库、不抄代码，从数学原理出发手写实现，亲手踩坑，感受技术演进的必然性。

## 项目结构

```
ML_History/
├── 01_foundations/          # 第一阶段：奠基时代 (1943-1969) ✅ 完成
│   ├── 01_mp_neuron/       # MP神经元模型
│   ├── 02_linear_regression/  # 线性回归
│   ├── 03_logistic_regression/  # 逻辑回归
│   ├── 04_perceptron/      # 感知机
│   ├── 05_knn/             # k近邻
│   └── 06_kmeans/          # k-Means聚类
├── 02_statistical_learning/  # 第二阶段：统计学习时代 (1980s-2000s) ✅ 完成
│   ├── 01_decision_tree/   # 决策树(ID3)
│   ├── 02_mlp_backprop/    # MLP+反向传播
│   ├── 03_svm/             # 支持向量机
│   ├── 04_adaboost/        # AdaBoost
│   ├── 05_bagging/         # Bagging
│   ├── 06_random_forest/   # 随机森林
│   └── 07_gbdt/            # GBDT
├── 03_deep_learning_revival/  # 第三阶段：深度学习复兴 (2006-2014) ✅ 完成
│   ├── 01_dbn/             # 深度信念网络
│   ├── 02_lenet5_cnn/      # LeNet-5 CNN
│   ├── 03_rnn/             # RNN
│   ├── 04_lstm/            # LSTM
│   ├── 05_word2vec/        # Word2Vec
│   └── 06_gan/             # GAN
├── 04_deep_learning_boom/  # 第四阶段：深度学习爆发 (2015-2019) 🚧 进行中
│   ├── 01_resnet/
│   ├── 02_xgboost/
│   ├── 03_lightgbm/
│   ├── 04_transformer/
│   ├── 05_bert/
│   └── 06_gpt/
├── 05_large_model_era/     # 第五阶段：大模型时代 (2020-)
│   ├── 01_scaling_law/
│   ├── 02_diffusion_model/
│   └── 03_vit/
├── common/                 # 通用工具函数
├── docs/                   # 学习笔记与历史背景
└── IDEA.md                 # 详细项目说明
```

## 实现原则

| 阶段 | 时间范围 | 实现要求 |
|------|----------|----------|
| 第一阶段 | 1943-1969 | 纯NumPy，不调用任何ML库 |
| 第二阶段 | 1980s-2000s | 核心逻辑手写，可调用sklearn对比 |
| 第三阶段 | 2006-2014 | 引入PyTorch，关键模块手写 |
| 第四阶段 | 2015-2019 | PyTorch为主，注重原理复现 |
| 第五阶段 | 2020-至今 | 小型复现，理解原理为主 |

## 每个模型的4个核心问题

1. **历史背景**：这个模型出现之前，领域遇到了什么瓶颈？
2. **核心创新**：最关键的1-2个技术突破是什么？
3. **局限与争议**：当时人们发现了什么问题？
4. **历史影响**：它直接催生了哪些后续工作？

## 快速开始

```bash
# 安装依赖
pip install numpy matplotlib scikit-learn torch pytest

# 运行测试
python -m pytest 01_foundations/*/test_*.py -v
python -m pytest 02_statistical_learning/*/test_*.py -v
python -m pytest 03_deep_learning_revival/*/test_*.py -v

# 运行训练演示
python 01_foundations/01_mp_neuron/train.py
python 02_statistical_learning/01_decision_tree/train.py
python 03_deep_learning_revival/01_dbn/train.py
```

## 推荐数据集

- **分类基准**: Iris, Breast Cancer, MNIST, CIFAR-10
- **回归基准**: Boston Housing, California Housing
- **序列基准**: 时间序列数据, 简单文本语料
- **生成基准**: MNIST (GAN/扩散模型)

## 学习节奏建议

- 每周1-2个模型
- 每个模型先读论文/教材 → 推公式 → 写代码 → 跑实验 → 写总结
- 每完成一个阶段，写一篇阶段总结

## 参考资源

- [IDEA.md](IDEA.md) - 详细项目规划
- [docs/](docs/) - 学习笔记与历史背景

---

> *"The best way to predict the future is to invent it. The best way to understand the past is to rebuild it."*