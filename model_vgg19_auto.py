import torch.nn as nn
import math

cfg = {'V' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]}
# 自动生成模型

class VGG_19(nn.Module):
    def __init__(self, extract, class_num=100, init_weight=True):
        super(VGG_19, self).__init__()
        self.extract = extract
        self.classifer = nn.Sequential(
            nn.Linear(2*2*512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, class_num)
        )
        if init_weight:
            self.initialize_weight()

    def forward(self, x):
        x = self.extract(x)
        x = x.view(-1, 2*2*512)
        x = self.classifer(x)
        return x

    def initialize_weight(self):
        for i in self.modules():
            if isinstance(i, nn.Conv2d):
                size = i.kernel_size[1] * i.kernel_size[0] * i.out_channels
                i.weight.data.normal_(0, math.sqrt(2.0/size))  # 正态填充
                if i.bias is not None:
                    i.bias.data.zero_()
            elif isinstance(i, nn.BatchNorm2d):
                i.weight.data.fill_(1)
                i.bias.data.zero_()
            elif isinstance(i, nn.Linear):
                i.weight.data.normal_(0,0.0001)
                i.bias.data.zero_()


def make_layers(cfg, batch_norm):
    layers = []
    in_channels = 3
    for i in cfg:
        if i == 'M':
            layers += [nn.MaxPool2d(kernel_size=2)]
        else:
            conv2d = nn.Conv2d(in_channels, out_channels=i, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(num_features=i), nn.ReLU(True)]
            else:
                layers += [conv2d, nn.ReLU(True)]

            in_channels = i

    return nn.Sequential(*layers)


def vgg_19_bn():
    return VGG_19(make_layers(cfg['V'], True))
