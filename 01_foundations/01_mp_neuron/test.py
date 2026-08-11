"""MP神经元测试脚本。"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from train import test_and_gate, test_or_gate, test_not_gate, test_xor_impossibility


def run_all_tests():
    """运行所有测试。"""
    test_and_gate()
    test_or_gate()
    test_not_gate()
    test_xor_impossibility()
    print("所有测试完成！")


if __name__ == "__main__":
    run_all_tests()