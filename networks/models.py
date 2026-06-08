import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision import transforms

from .blocks import *


# class Normalize:
#     def __init__(self, opt, expected_values, variance):
#         self.n_channels = opt.input_channel
#         self.expected_values = expected_values
#         self.variance = variance
#         assert self.n_channels == len(self.expected_values)
#
#     def __call__(self, x):
#         x_clone = x.clone()
#         for channel in range(self.n_channels):
#             x_clone[:, channel] = (x[:, channel] - self.expected_values[channel]) / self.variance[channel]
#         return x_clone
#
#
# class Denormalize:
#     def __init__(self, opt, expected_values, variance):
#         self.n_channels = opt.input_channel
#         self.expected_values = expected_values
#         self.variance = variance
#         assert self.n_channels == len(self.expected_values)
#
#     def __call__(self, x):
#         x_clone = x.clone()
#         for channel in range(self.n_channels):
#             x_clone[:, channel] = x[:, channel] * self.variance[channel] + self.expected_values[channel]
#         return x_clone


class Normalize:
    def __init__(self, opt, expected_values, variance):
        self.n_channels = opt.input_channel
        self.expected_values = expected_values
        self.variance = variance
        assert self.n_channels == len(self.expected_values)

    def __call__(self, x):
        x_clone = x.clone()
        for channel in range(self.n_channels):
            x_clone[:, channel] = (x[:, channel] - self.expected_values[channel]) / self.variance[channel]
        return x_clone


class Denormalize:
    def __init__(self, opt, expected_values, variance):
        self.n_channels = opt.input_channel
        self.expected_values = expected_values
        self.variance = variance
        assert self.n_channels == len(self.expected_values)

    def __call__(self, x):
        x_clone = x.clone()
        for channel in range(self.n_channels):
            x_clone[:, channel] = x[:, channel] * self.variance[channel] + self.expected_values[channel]
        return x_clone


class Normalizer:
    def __init__(self, opt):
        self.normalizer = self._get_normalizer(opt)

    def _get_normalizer(self, opt):
        if opt.dataset == "cifar10":
            normalizer = Normalize(opt, [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        elif opt.dataset == "mnist":
            normalizer = Normalize(opt, [0.5], [0.5])
        elif opt.dataset == "gtsrb":
            normalizer = None
        elif opt.dataset == "imagenet":
            normalizer = Normalize(opt,[0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        else:
            raise Exception("Invalid dataset")
        return normalizer

    def __call__(self, x):
        if self.normalizer:
            x = self.normalizer(x)
        return x


class Denormalizer:
    def __init__(self, opt):
        self.denormalizer = self._get_denormalizer(opt)

    def _get_denormalizer(self, opt):
        if opt.dataset == "cifar10":
            denormalizer = Denormalize(opt, [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        elif opt.dataset == "mnist":
            denormalizer = Denormalize(opt, [0.5], [0.5])
        elif opt.dataset == "gtsrb":
            denormalizer = None
        elif opt.dataset == "imagenet":
            denormalizer = Denormalize(opt,[0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        else:
            raise Exception("Invalid dataset")
        return denormalizer

    def __call__(self, x):
        if self.denormalizer:
            x = self.denormalizer(x)
        return x




# ---------------------------- Generators ----------------------------#


class Generator(nn.Sequential):  # 定义一个继承自nn.Sequential的生成器类
    def __init__(self, opt, out_channels=None):  # 初始化函数，接受配置参数opt和输出通道数
        super(Generator, self).__init__()  # 调用父类nn.Sequential的初始化函数
        if opt.dataset == "mnist":  # 如果数据集是MNIST
            channel_init = 16  # 初始化通道数为16
            steps = 2  # 处理步骤数为2
        else:  # 如果是其他数据集
            channel_init = 32  # 初始化通道数为32
            steps = 3  # 处理步骤数为3

        channel_current = opt.input_channel  # 设置当前通道数为输入通道数
        channel_next = channel_init  # 设置下一步的通道数为初始化通道数
        for step in range(steps):  # 遍历每一步
            self.add_module("convblock_down_{}".format(2 * step), Conv2dBlock(channel_current, channel_next))  # 添加下采样卷积块
            self.add_module("convblock_down_{}".format(2 * step + 1), Conv2dBlock(channel_next, channel_next))  # 再添加一个下采样卷积块
            self.add_module("downsample_{}".format(step), DownSampleBlock())  # 添加下采样模块
            if step < steps - 1:  # 如果不是最后一步
                channel_current = channel_next  # 更新当前通道数
                channel_next *= 2  # 更新下一步的通道数为当前的2倍

        self.add_module("convblock_middle", Conv2dBlock(channel_next, channel_next))  # 添加中间的卷积块

        channel_current = channel_next  # 更新当前通道数为中间通道数
        channel_next = channel_current // 2  # 更新下一步的通道数为当前的一半
        for step in range(steps):  # 遍历每一步
            self.add_module("upsample_{}".format(step), UpSampleBlock())  # 添加上采样模块
            self.add_module("convblock_up_{}".format(2 * step), Conv2dBlock(channel_current, channel_current))  # 添加上采样卷积块
            if step == steps - 1:  # 如果是最后一步
                self.add_module(
                    "convblock_up_{}".format(2 * step + 1), Conv2dBlock(channel_current, channel_next, relu=False)  # 添加不带relu的上采样卷积块
                )
            else:  # 如果不是最后一步
                self.add_module("convblock_up_{}".format(2 * step + 1), Conv2dBlock(channel_current, channel_next))  # 添加带relu的上采样卷积块
            channel_current = channel_next  # 更新当前通道数
            channel_next = channel_next // 2  # 更新下一步的通道数为当前的一半
            if step == steps - 2:  # 如果是倒数第二步
                if out_channels is None:  # 如果输出通道数未指定
                    channel_next = opt.input_channel  # 设置下一步的通道数为输入通道数
                else:  # 如果输出通道数已指定
                    channel_next = out_channels  # 设置下一步的通道数为输出通道数

        self._EPSILON = 1e-7  # 设置一个极小值
        self._normalizer = self._get_normalize(opt)  # 获取标准化函数
        self._denormalizer = self._get_denormalize(opt)  # 获取反标准化函数

    def _get_denormalize(self, opt):
        if opt.dataset == "cifar10":
            denormalizer = Denormalize(opt, [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        elif opt.dataset == "mnist":
            denormalizer = Denormalize(opt, [0.5], [0.5])
            # denormalizer = None
        elif opt.dataset == "gtsrb":
            denormalizer = None
        else:
            raise Exception("Invalid dataset")
        return denormalizer

    def _get_normalize(self, opt):
        if opt.dataset == "cifar10":
            normalizer = Normalize(opt, [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        elif opt.dataset == "mnist":
            normalizer = Normalize(opt, [0.5], [0.5])
            # normalizer = None
        elif opt.dataset == "gtsrb":
            normalizer = None
        else:
            raise Exception("Invalid dataset")
        return normalizer

    # def forward(self, x):
    #     for module in self.children():
    #         x = module(x)
    #     x = nn.Tanh()(x) / (2 + self._EPSILON) + 0.5
    #     return x
    def forward(self, x):
        # 遍历生成器模块进行处理
        for module in self.children():
            x = module(x)

        # 将 nn.Tanh() 的输出值限制在其他区间，并进行更细腻的调整
        x = nn.Tanh()(x)  # 使用 Tanh 激活函数
        x = x / (2 + self._EPSILON) + 0.5  # 缩放值到 [0, 1]
        x = torch.clamp(x, 0.2, 0.8)  # 使用 torch.clamp 将值限制在特定亮度区间

        return x

    def normalize_pattern(self, x):
        if self._normalizer:
            x = self._normalizer(x)
        return x

    def denormalize_pattern(self, x):
        if self._denormalizer:
            x = self._denormalizer(x)
        return x

    # def threshold(self, x):
    #     return nn.Tanh()(x * 20 - 10) / (2 + self._EPSILON) + 0.5
    def threshold(self, x):
        # 减小激活函数的输出区间来限制颜色的强度
        x = nn.Tanh()(x * 15 - 7.5)  # 缩小输出区间
        x = x / (2 + self._EPSILON) + 0.5  # 缩放值到 [0, 1]
        x = torch.clamp(x, 0.3, 0.7)  # 控制在更低的亮度范围内

        return x


# ---------------------------- Classifiers ----------------------------#


class NetC_MNIST(nn.Module):
    def __init__(self):
        super(NetC_MNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, (5, 5), 1, 0)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout3 = nn.Dropout(0.1)

        self.maxpool4 = nn.MaxPool2d((2, 2))
        self.conv5 = nn.Conv2d(32, 64, (5, 5), 1, 0)
        self.relu6 = nn.ReLU(inplace=True)
        self.dropout7 = nn.Dropout(0.1)

        self.maxpool5 = nn.MaxPool2d((2, 2))
        self.flatten = nn.Flatten()
        self.linear6 = nn.Linear(64 * 4 * 4, 512)
        self.relu7 = nn.ReLU(inplace=True)
        self.dropout8 = nn.Dropout(0.1)
        self.linear9 = nn.Linear(512, 10)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x
