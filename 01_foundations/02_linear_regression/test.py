"""线性回归测试脚本。"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from train import demo_basic, demo_gradient_descent, demo_real_data


def run_all_tests():
    """运行所有演示。"""
    demo_basic()
    demo_gradient_descent()
    demo_real_data()
    print("所有测试完成！")


if __name__ == "__main__":
    run_all_tests()