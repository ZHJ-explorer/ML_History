"""MP神经元训练脚本。

演示MP神经元实现基本逻辑门。
"""

import numpy as np
from model import MPNeuron


def test_and_gate():
    """测试AND逻辑门。"""
    print("=" * 50)
    print("测试 AND 逻辑门")
    print("=" * 50)

    neuron = MPNeuron(n_features=2)
    neuron.set_logic_gate("AND")

    # 真值表
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])

    predictions = neuron.predict(X)

    print(f"输入: {X.tolist()}")
    print(f"真实输出: {y.tolist()}")
    print(f"预测输出: {predictions.tolist()}")
    print(f"准确率: {neuron.score(X, y):.2f}")
    print()


def test_or_gate():
    """测试OR逻辑门。"""
    print("=" * 50)
    print("测试 OR 逻辑门")
    print("=" * 50)

    neuron = MPNeuron(n_features=2)
    neuron.set_logic_gate("OR")

    # 真值表
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 1])

    predictions = neuron.predict(X)

    print(f"输入: {X.tolist()}")
    print(f"真实输出: {y.tolist()}")
    print(f"预测输出: {predictions.tolist()}")
    print(f"准确率: {neuron.score(X, y):.2f}")
    print()


def test_not_gate():
    """测试NOT逻辑门。"""
    print("=" * 50)
    print("测试 NOT 逻辑门")
    print("=" * 50)

    neuron = MPNeuron(n_features=1)
    neuron.set_logic_gate("NOT")

    # 真值表
    X = np.array([[0], [1]])
    y = np.array([1, 0])

    predictions = neuron.predict(X)

    print(f"输入: {X.tolist()}")
    print(f"真实输出: {y.tolist()}")
    print(f"预测输出: {predictions.tolist()}")
    print(f"准确率: {neuron.score(X, y):.2f}")
    print()


def test_xor_impossibility():
    """演示XOR问题的不可解性。"""
    print("=" * 50)
    print("演示 XOR 问题（不可解）")
    print("=" * 50)

    print("XOR问题的真值表：")
    print("0 0 -> 0")
    print("0 1 -> 1")
    print("1 0 -> 1")
    print("1 1 -> 0")
    print()
    print("MP神经元无法解决XOR问题，因为它是线性不可分的。")
    print("需要多层网络才能解决。")
    print()


if __name__ == "__main__":
    test_and_gate()
    test_or_gate()
    test_not_gate()
    test_xor_impossibility()