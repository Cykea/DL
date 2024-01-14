import torch.nn as nn
import torch

# official pretrain weights
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
}


class VGG(nn.Module):                                                       # 定义VGG类 继承nn.module父类
    def __init__(self, features, num_classes=1000, init_weights=False):     # features特征函数， num_classes分类个数，weights网络权重初始化
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(                                    # 分类网络结构
            nn.Linear(512*7*7, 4096),                   # 输入展平的一维个数
            nn.ReLU(True),                            # ReLU激活函数
            nn.Dropout(p=0.5),                           # 减少过拟合_以0.5的比率失活训练参数
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):               # forward正向传播 x为输入数据
        # N x 3 x 224 x 224
        x = self.features(x)
        # N x 512 x 7 x 7
        x = torch.flatten(x, start_dim=1)
        # N x 512*7*7
        x = self.classifier(x)
        return x

    def _initialize_weights(self):            # 定义初始化权重函数
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)                      # 若初始化权重函数遍历到卷积层，就用xavier初始化函数初始卷积核权重
                if m.bias is not None:                   # 如果卷积核采用bias（偏置），则将偏置默认初始化为0
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):                    # 若初始化权重函数遍历到全连接层，就用xavier初始化函数初始全连接层权重
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)                      # 同样将偏置置为变量0


def make_features(cfg: list):   # 提取特征 函数 make_feature   传入配置变量cfg list列表
    layers = []                 # 定义空列表层 用来存放 所创建的每一层
    in_channels = 3             # 定义变量 输入图片RGB 固 深度 为3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(True)]
            in_channels = v
    return nn.Sequential(*layers)    # *表示非关键字参数传入


cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],     # M代表maxpool池化层
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg(model_name="vgg16", **kwargs):
    assert model_name in cfgs, "Warning: model number {} not in cfgs dict!".format(model_name)
    cfg = cfgs[model_name]            # cfgs值传入字典，可得到上部数据

    model = VGG(make_features(cfg), **kwargs)             # **kwargs可变字典变量，可传入上边个数、网络权重初始化函数
    return model
