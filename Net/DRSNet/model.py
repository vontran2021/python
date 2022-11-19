import torch.nn as nn
import torch
from torch import Tensor
import torch.nn.functional as F


def channel_shuffle(x: Tensor, groups: int) -> Tensor:

    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    # [batch_size, num_channels, height, width] -> [batch_size, groups, channels_per_group, height, width]
    x = x.view(batch_size, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batch_size, -1, height, width)

    return x


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=1, stride=1,  groups=1, bias=False):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=bias),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

class gConvBNReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=1, stride=1,  groups=4, bias=False):
        padding = (kernel_size - 1) // 2
        super(gConvBNReLU, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=bias),
            nn.BatchNorm2d(out_channel),
        )


class DRSNet_unit(nn.Module):
    def __init__(self, in_channel, out_channel=112, stride=2):
        super(DRSNet_unit, self).__init__()
        self.gconv1 = ConvBNReLU(in_channel=in_channel, out_channel=112, stride=1, groups=4, bias=False)
        self.Dwconv = gConvBNReLU(in_channel=112, out_channel=112, kernel_size=3,
                                  stride=stride, groups=112, bias=False)
        self.gconv2 = gConvBNReLU(in_channel=112, out_channel=112, stride=1, groups=4, bias=False)

    def forward(self, x):
        short = x
        short = F.avg_pool2d(short, kernel_size=3, stride=2, padding=1)

        x = self.gconv1(x)
        x = channel_shuffle(x, 4)
        x = self.Dwconv(x)
        x = self.gconv2(x)
        x = torch.cat((x, short), dim=1)

        return F.relu(x)


class DRSNet_unit2(nn.Module):
    def __init__(self, in_channel, out_channel, stride=2):
        super(DRSNet_unit2, self).__init__()
        self.gconv1 = ConvBNReLU(in_channel=in_channel, out_channel=out_channel, stride=1, groups=4, bias=False)
        self.Dwconv = gConvBNReLU(in_channel=out_channel, out_channel=out_channel, kernel_size=3,
                                  stride=stride, groups=out_channel, bias=False)
        self.gconv2 = gConvBNReLU(in_channel=out_channel, out_channel=out_channel, stride=1, groups=4, bias=False)

    def forward(self, x):
        short = x
        short = F.avg_pool2d(short, kernel_size=3, stride=2, padding=1)

        x = self.gconv1(x)
        x = channel_shuffle(x, 4)
        x = self.Dwconv(x)
        x = self.gconv2(x)
        x = torch.cat((x, short), dim=1)

        return F.relu(x)


class DRSNet_unit1(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(DRSNet_unit1, self).__init__()
        self.gconv1 = ConvBNReLU(in_channel=in_channel, out_channel=out_channel, stride=1, groups=4, bias=False)
        self.Dwconv = gConvBNReLU(in_channel=out_channel, out_channel=out_channel, kernel_size=3,
                                  stride=stride, groups=out_channel, bias=False)
        self.gconv2 = gConvBNReLU(in_channel=out_channel, out_channel=out_channel, stride=1, groups=4, bias=False)

    def forward(self, x):
        short = x
        x = self.gconv1(x)
        x = channel_shuffle(x, 4)
        x = self.Dwconv(x)
        x = self.gconv2(x)
        x = x + short

        return F.relu(x)


class DRSNet_unit3(nn.Module):
    def __init__(self, in_channel, out_channel, stride=2):
        super(DRSNet_unit3, self).__init__()
        self.gconv1 = ConvBNReLU(in_channel=in_channel, out_channel=136, stride=1, groups=4, bias=False)
        self.Dwconv = gConvBNReLU(in_channel=136, out_channel=136, kernel_size=3,
                                  stride=stride, groups=136, bias=False)
        self.gconv2 = gConvBNReLU(in_channel=136, out_channel=136, stride=1, groups=4, bias=False)

    def forward(self, x):
        short = x
        short = F.avg_pool2d(short, kernel_size=3, stride=2, padding=1)

        x = self.gconv1(x)
        x = channel_shuffle(x, 4)
        x = self.Dwconv(x)
        x = self.gconv2(x)
        x = torch.cat((x, short), dim=1)

        return F.relu(x)


class DRSNet_unit4(nn.Module):
    def __init__(self, in_channel, out_channel=272, stride=2):
        super(DRSNet_unit4, self).__init__()
        self.gconv1 = ConvBNReLU(in_channel=in_channel, out_channel=272, stride=1, groups=4, bias=False)
        self.Dwconv = gConvBNReLU(in_channel=272, out_channel=272, kernel_size=3,
                                  stride=stride, groups=272, bias=False)
        self.gconv2 = gConvBNReLU(in_channel=272, out_channel=272, stride=1, groups=4, bias=False)

    def forward(self, x):
        short = x
        short = F.avg_pool2d(short, kernel_size=3, stride=2, padding=1)

        x = self.gconv1(x)
        x = channel_shuffle(x, 4)
        x = self.Dwconv(x)
        x = self.gconv2(x)
        x = torch.cat((x, short), dim=1)

        return F.relu(x)


class DRSNet(nn.Module):
    """DRSNet implementation.
    """

    def __init__(self, last_channel=544, num_classes=1000):
        super(DRSNet, self).__init__()

        blocks_num = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        # input_channel = 32
        self.last_channel = last_channel
        self.num_classes = num_classes
        self.in_channel = 24
        self.in_channel1 = [24, 136, 136, 272, 272, 272, 544, 544]
        self.shortcut_channel = [136, 136, 272, 272, 272, 544]
        self.conv1 = ConvBNReLU(3, self.in_channel, kernel_size=3, stride=2,
                                bias=False)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.conv2 = ConvBNReLU(512, last_channel, kernel_size=1, stride=1, bias=False)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

# Defining the trunk Layer
        self.fc = nn.Linear(self.last_channel, self.num_classes)
        self.layer = self._make_layer(DRSNet_unit, 136, blocks_num[0], stride=2)
        self.layer1_1 = self._make_layer(DRSNet_unit1, 136, blocks_num[1], stride=1)
        self.conv2 = ConvBNReLU(136, self.in_channel1[1], kernel_size=1, stride=1,
                                bias=False)

        self.layer1_2 = self._make_layer(DRSNet_unit1, 136, blocks_num[2], stride=1)
        self.layer1_3 = self._make_layer(DRSNet_unit1, 136, blocks_num[3], stride=1)
        self.conv3 = ConvBNReLU(136, self.in_channel1[2], kernel_size=1, stride=1,
                                bias=False)

        self.layer2_1 = self._make_layer(DRSNet_unit3, 272, blocks_num[4], stride=2)
        self.layer2_2 = self._make_layer(DRSNet_unit1, 272, blocks_num[5], stride=1)
        self.conv4 = ConvBNReLU(272, self.in_channel1[3], kernel_size=1, stride=1,
                                bias=False)

        self.layer2_3 = self._make_layer(DRSNet_unit1, 272, blocks_num[6], stride=1)
        self.layer2_4 = self._make_layer(DRSNet_unit1, 272, blocks_num[7], stride=1)
        self.conv5 = ConvBNReLU(272, self.in_channel1[4], kernel_size=1, stride=1,
                                bias=False)

        self.layer2_5 = self._make_layer(DRSNet_unit1, 272, blocks_num[8], stride=1)
        self.layer2_6 = self._make_layer(DRSNet_unit1, 272, blocks_num[9], stride=1)
        self.conv6 = ConvBNReLU(272, self.in_channel1[5], kernel_size=1, stride=1,
                                bias=False)

        self.layer3_1 = self._make_layer(DRSNet_unit4, 544, blocks_num[10], stride=2)
        self.layer3_2 = self._make_layer(DRSNet_unit1, 544, blocks_num[11], stride=1)
        self.conv7 = ConvBNReLU(544, self.in_channel1[6], kernel_size=1, stride=1,
                                bias=False)

        self.layer3_3 = self._make_layer(DRSNet_unit1, 544, blocks_num[12], stride=1)
        self.layer3_4 = self._make_layer(DRSNet_unit1, 544, blocks_num[13], stride=1)
        self.conv8 = ConvBNReLU(544, self.in_channel1[7], kernel_size=1, stride=1,
                                bias=False)

# Define shortcut layer
        self.shortcut_conv1 = ConvBNReLU(24, self.shortcut_channel[0], kernel_size=1, stride=2,
                                         bias=False)
        self.shortcut_conv2 = ConvBNReLU(136, self.shortcut_channel[1], kernel_size=1, stride=1,
                                         bias=False)
        self.shortcut_conv3 = ConvBNReLU(136, self.shortcut_channel[2], kernel_size=1, stride=2,
                                         bias=False)
        self.shortcut_conv4 = ConvBNReLU(272, self.shortcut_channel[3], kernel_size=1, stride=1,
                                         bias=False)
        self.shortcut_conv5 = ConvBNReLU(272, self.shortcut_channel[4], kernel_size=1, stride=1,
                                         bias=False)
        self.shortcut_conv6 = ConvBNReLU(272, self.shortcut_channel[5], kernel_size=1, stride=2,
                                         bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _make_layer(self, block, out_channels, block_num, stride):

        layers = []
        # layers.append(block(in_channel,
        #                     out_channel,
        #                     stride=stride))
        # self.in_channel = channel * block.expansion

        for _ in range(0, block_num):
            layers.append(block(self.in_channel,
                                out_channels,
                                stride=stride))
            self.in_channel = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        shortcu1_1 = self.shortcut_conv1(x)

        x1_1 = self.layer(x)
        x = self.layer1_1(x1_1)

        x = x + shortcu1_1

        x = self.conv2(x)
    #############################################
        shortcu1_2 = self.shortcut_conv2(x)

        x1_2 = self.layer1_2(x)
        x = self.layer1_3(x1_2)

        x = shortcu1_2 + x

        x = self.conv3(x)
    #############################################
        shortcu2_1 = self.shortcut_conv3(x)

        x2_1 = self.layer2_1(x)
        x = self.layer2_2(x2_1)

        x = shortcu2_1 + x

        x = self.conv4(x)
    #############################################
        shortcu2_2 = self.shortcut_conv4(x)

        x2_2 = self.layer2_3(x)
        x = self.layer2_4(x2_2)

        x = shortcu2_2 + x

        x = self.conv5(x)
    #############################################
        shortcu2_3 = self.shortcut_conv5(x)

        x2_3 = self.layer2_5(x)
        x = self.layer2_6(x2_3)

        x = shortcu2_3 + x

        x = self.conv6(x)
    #############################################
        shortcu3_1 = self.shortcut_conv6(x)

        x3_1 = self.layer3_1(x)
        x = self.layer3_2(x3_1)

        x = shortcu3_1 + x

        x = self.conv7(x)
    #############################################
        shortcu3_2 = x

        x3_2 = self.layer3_3(x)
        x = self.layer3_4(x3_2)

        x = shortcu3_2 + x

        x = self.conv8(x)
    #############################################
        # global average pooling layer
        x = F.avg_pool2d(x, x.data.size()[-2:])

        # flatten for input to fully-connected layer
        x = x.view(x.size(0), -1)
        # x = x.view(x.size(0), x.size(1), x.size(2), x.size(3), self.num_classes)

        x = self.fc(x)
        return x




