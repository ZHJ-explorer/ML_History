"""逻辑回归测试脚本。"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from train import demo

if __name__ == "__main__":
    demo()
    print("测试完成！")