import torch.nn as nn
import torch

# 1、首先定义组成Resnet的残差结构网络，每个残差结构如何构成与运行
# 2、然后定义整个Resnet网络框架结构


class BasicBlock(nn.Module):           # 定义对应18layer、34layer的残差结构的类
    expansion = 1                                      # 主分支上残差结构第一层和第二层卷积核个数一样

                            # 传入的参数 out_channel主分支上卷积核的个数

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        # out_channel主分支上卷积核的个数 downsample=None下采样残差结构，虚线对应的残差结构
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        # stride=1代表实线残差结构    stride=2代表虚线残差结构   bias=False代表不使用偏置
        self.bn1 = nn.BatchNorm2d(out_channel)                                           # out_channel为conv1所输出的深度
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:      # if self.downsample is not None:执行下一步输入特征矩阵x，走捷径分支，if is none则跳过为实线残差结构
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)           # 第一层

        out = self.conv2(out)
        out = self.bn2(out)             # 第二层

        out += identity               # 两层相加后relu激活函数
        out = self.relu(out)

        return out
# 18、34layer定义完成

# 定义50、101、102层残差结构


class Bottleneck(nn.Module):
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    expansion = 4              # 卷积核有变化，第三层卷积核个数是上一层的4倍，64*4=256

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,             # 定义初始化函数
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups        # resnext中，group4，widthpergroup32

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,              # 第一层conv1
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):                  # 定义正向传播过程，x为输入的特征矩阵
        identity = x
        if self.downsample is not None:       # 判断输入的下采样是否为none，是则为实线残差结构，跳过下一步，if is not none则为虚线残差结构，继续执行
            identity = self.downsample(x)           # 捷径分支

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out

# 50、101残差结构定义完成


# 定义整个resnet网络框架结构


class ResNet(nn.Module):

    def __init__(self,
                 block,                      # 为上述定义的block 既Bottleneck
                 blocks_num,                # 所使用残差结构的数目（50-3 4 6 3）
                 num_classes=1000,            # 训练集分类个数
                 include_top=True,             # 以便搭载其他复杂网络
                 groups=1,
                 width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64                       # conv1开始 输入的特征矩阵7*7size64

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,        # 3为RGB深度，self.in_channel为64
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)       #
        self.layer1 = self._make_layer(block, 64, blocks_num[0])             # layer1对应conv2_x残差结构，以下类推
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1) # 平均池化下采样层（高1，宽1）
            self.fc = nn.Linear(512 * block.expansion, num_classes)  # 全连接层

        for m in self.modules():               # 卷积层初始化
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):       # channel每一块第一层对应深度，_make_layer上述layer1234传入什么参数从这定义
        downsample = None                   # block_num包含几个残差结构，如50layer的conv3_x包含4个     运行第2.3层会再判断
        if stride != 1 or self.in_channel != channel * block.expansion:      # 判断步距和深度，如果不等于，进行下一步下采样，18和34layer在此不满足直接跳过
            downsample = nn.Sequential(                                 #　高和宽不变
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))        # 对应BatchNorm2d传入特征矩阵深度

        layers = []              # 定义一个空的列表
        layers.append(block(self.in_channel,              # 添加第一层残差结构，block为class定义的
                            channel,                     # 残差结构主分支上第一个卷积核个数
                            downsample=downsample,
                            stride=stride,                        # 接着在此运行conv3_x,4x,5x虚线残差结构
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion          # block.expansion（1，4）

        for _ in range(1, block_num):         # 上述已经定义好残差结构第一层，选择实线还是虚线
            layers.append(block(self.in_channel,    # 接着运行该残差结构的第2.3实线层
                                channel,                     # 接着在这个循环运行conv3_x,4x,5x实线残差结构
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)        # 将layers = []转化为非关键字参数*layers，Sequential函数将其组合后返回对应的layer1234

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


def resnet34(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet50(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


def resnext50_32x4d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
    groups = 32
    width_per_group = 4
    return ResNet(Bottleneck, [3, 4, 6, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)


def resnext101_32x8d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
    groups = 32
    width_per_group = 8
    return ResNet(Bottleneck, [3, 4, 23, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)
