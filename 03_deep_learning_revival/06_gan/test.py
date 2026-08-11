"""GAN测试脚本。"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train import demo

if __name__ == "__main__":
    demo()
    print("测试完成！")