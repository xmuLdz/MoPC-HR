import torch
import torch.nn as nn
from ConvBlockAttention import ConvBlockAttention
from torchvision.models import resnet101
import numpy as np
device = "cuda:0" if torch.cuda.is_available() else "cpu"



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, kernel_size=2, downsample=None, attention=ConvBlockAttention):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=kernel_size, stride=stride, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)  # inplace=True 节省内存

        self.conv2 = nn.Conv1d(planes, planes, kernel_size=kernel_size, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride
        self.attention = None
        if attention is not None:
            self.attention = attention(planes, 2)

    def forward(self, x: torch.Tensor):
        residual = x
        out = self.conv1(x)

        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.attention is not None:
            out = self.attention(out)

        out_size = out.size(2)

        residual_size = residual.size(2)
        is_large = residual_size > out_size

        zero_padding = torch.zeros(
            abs(residual_size - out_size) * residual.size(1) * residual.size(0)
        ).view(residual.size(0), residual.size(1), -1)

        is_gpu = x.device.type != "cpu"

        if is_gpu:
            zero_padding = zero_padding.cuda()

        if is_large:
            out = torch.cat([out, zero_padding], dim=2)

            if is_gpu:
                out = out.cuda()

        else:
            residual = torch.cat([residual, zero_padding], dim=2)
            if is_gpu:
                residual = residual.cuda()

        out += residual
        out = self.relu(out)
        return out

class BasicBlock2(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass

class ResNet(nn.Module):
    def __init__(self, block, configs, attention):
        super(ResNet, self).__init__()

        # self.conv1_ = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        # self.bn1_ = nn.BatchNorm1d(1)
        #ADS-B100-B
        self.conv1_ = nn.Conv2d(3, 1, kernel_size=3, padding=1,
                               bias=False)

        self.in_planes = 2
        # 32 64 128
        #self.conv1 = nn.Conv1d(1, 2, kernel_size=128, stride=1, bias=False)
        self.conv1 = nn.Conv1d(1, 2, kernel_size=7, stride=1, bias=False)
        self.bn1 = nn.BatchNorm1d(2)
        self.relu = nn.ReLU(inplace=True)
        # self.attention = None
        if attention is not None:
            self.attention = attention
        self.layers = nn.Sequential(*map(
            lambda config: self.make_layer(
                block,
                config["channel"],
                config["layer"],
                kernel_size=config["kernel_size"]
            ),
            configs
        ))
        # self.make_layer(block, 4, layers[0], kernel_size=16),
        # self.make_layer(block, 8, layers[1], kernel_size=8),
        # self.make_layer(block, 32, layers[2], kernel_size=4),
        # self.make_layer(block, 64, layers[3], kernel_size=2),
        # self.make_layer(block, 128, layers[4], kernel_size=2),
        self.avg_pooling = nn.AdaptiveAvgPool2d((1, configs[-1]["channel"]))

        self.out_dim = configs[-1]["channel"] * block.expansion
        self.fc = nn.Linear(configs[-1]["channel"] * block.expansion, 100)


    def make_layer(self, block, planes, blocks, kernel_size=2, stride=1):
        downsample = None

        #shotcut
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion)
            )

        layers = [block(self.in_planes, planes, stride, kernel_size, downsample, attention=self.attention)]

        self.in_planes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes, kernel_size=kernel_size, attention=self.attention))

        return nn.Sequential(*layers)

    def feature_extract_stage1(self,x):
        x = self.conv1_(x)
        x = x.view(x.size(0), 1, 1024)
        return x

    def feature_extract_stage2(self,x):
        #针对ADS-B
        x = self.feature_extract_stage1(x)
        # print(x.size())
        x = x.view(x.size(0), 1, 1024)
        #AIS100-100
        # \\x = x.view(x.size(0), 1, x.size(1))
        # x = self.conv1_(x)
        # x = self.bn1_(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layers(x)
        x = self.avg_pooling(x1)

        x = x.view(x.size(0), -1)  # bsX256
        # x = self.fc(x)
        return x

    def forward(self, x):
        #ADS-B100-B
        # x = self.feature_extract_stage1(x)

        x=self.feature_extract_stage2(x)
        x = self.fc(x)
        return x
        # return {
        #     'fmaps': [x1],
        #     'features': features
        # }




model_config = [
    {"channel": 8, "layer": 2, "kernel_size": 8},
    {"channel": 32, "layer": 2, "kernel_size": 8},
    # {"channel": 32, "layer": 2, "kernel_size": 4},
    {"channel": 64, "layer": 2, "kernel_size": 4},
    {"channel": 128, "layer": 2, "kernel_size": 4},
    {"channel": 256, "layer": 2, "kernel_size": 2},
    # {"channel": 512, "layer": 2, "kernel_size": 2},
]

def resnet1d_ads():
    """Constructs a ResNet1d-22 model for AIS100-100."""
    model =ResNet(BasicBlock, configs=model_config,attention=ConvBlockAttention)
    return model