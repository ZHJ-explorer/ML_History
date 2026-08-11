"""ResNet残差网络实现。"""
import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """ResNet基础块，包含残差连接。"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        """初始化BasicBlock。

        Args:
            in_channels: 输入通道数。
            out_channels: 输出通道数。
            stride: 卷积步长，默认为1。
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )
    
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet残差网络，通过残差连接解决深度网络退化问题。"""

    def __init__(self, block, num_blocks, num_classes=10):
        """初始化ResNet模型。

        Args:
            block: 基本块类型（BasicBlock或Bottleneck）。
            num_blocks: 每个层的块数量列表，如[2,2,2,2]。
            num_classes: 分类类别数，默认为10。
        """
        super().__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.linear = nn.Linear(512 * block.expansion, num_classes)
    
    def _make_layer(self, block, channels, num_blocks, stride):
        """构建多层区块。

        Args:
            block: 基本块类型。
            channels: 输出通道数。
            num_blocks: 块数量。
            stride: 第一块的步长。

        Returns:
            由多个block组成的nn.Sequential模块。
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, channels, stride))
            self.in_channels = channels * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = nn.functional.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet18():
    """创建ResNet-18模型实例。

    Returns:
        ResNet实例，使用BasicBlock和[2,2,2,2]层配置。
    """
    return ResNet(BasicBlock, [2, 2, 2, 2])