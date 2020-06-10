import math

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    expansion = 4  # this is a class variable.

    def __init__(self, inplanes, planes, stride=1, dilation=1, adjust=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        padding = 2 - stride
        if adjust is not None and dilation > 1:
            dilation = dilation // 2
            padding = dilation

        assert stride == 1 or dilation == 1, \
            "stride and dilation must have one equals to zero at least"

        if dilation > 1:
            padding = dilation

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=padding, dilation=dilation,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * ResidualBlock.expansion, kernel_size=1,
                               bias=False)  # must add the class name when use class variable.
        self.bn3 = nn.BatchNorm2d(planes * ResidualBlock.expansion)

        self.relu = nn.ReLU(inplace=True)

        self.adjust = adjust
        self.stride = stride

    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.adjust is not None:
            res = self.adjust(res)

        out = x + res
        out = self.relu(out)  # add relu after adding.
        return out


class ResNet(nn.Module):
    def __init__(self, block, repetition, used_layers=(2, 3, 4)):
        super().__init__()
        self.used_layers = used_layers
        self.layer0 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        )
        self.inplane = 64
        self.layer1 = self._make_layer(block, repetition[0], 64)
        self.layer2 = self._make_layer(block, repetition[1], 128, stride=2)

        layer3 = True if 3 in used_layers else False
        layer4 = True if 4 in used_layers else False

        if layer3:
            self.layer3 = self._make_layer(block, repetition[2], 256, stride=1, dilation=2)
        else:
            self.layer3 = lambda x: x
        if layer4:
            self.layer4 = self._make_layer(block, repetition[3], 512, stride=1, dilation=4)
        else:
            self.layer4 = lambda x: x

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, repetition, plane, stride=1, dilation=1):
        adjust = None
        if stride != 1 or self.inplane != block.expansion:
            if stride == 1 and dilation == 1:
                adjust = nn.Sequential(
                    nn.Conv2d(self.inplane, plane * block.expansion, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(plane * block.expansion))
            else:
                if dilation > 1:
                    dd = dilation // 2
                    padding = dd
                else:
                    dd = 1
                    padding = 0
                adjust = nn.Sequential(
                    nn.Conv2d(self.inplane, plane * block.expansion, kernel_size=3, stride=stride, padding=padding,
                              dilation=dd, bias=False),
                    nn.BatchNorm2d(plane * block.expansion)
                )
        layer = []
        layer.append(block(self.inplane, plane, stride, dilation, adjust))
        self.inplane = plane * block.expansion
        for i in range(1, repetition):
            layer.append(block(self.inplane, plane, dilation=dilation, adjust=None))

        return nn.Sequential(*layer)

    def forward(self, x):
        out = []
        for i in range(5):
            layer_name = 'layer{}'.format(i)
            x = getattr(self, layer_name)(x)
            out.append(x)

        out = [out[i] for i in self.used_layers]
        if len(out) == 1:
            return out[0]
        else:
            return out


def resnet50(**kwargs):
    return ResNet(ResidualBlock, [3, 4, 6, 3], **kwargs)


if __name__ == '__main__':
    net = resnet50()
    print(net)
    net = net.cuda()

    var = torch.FloatTensor(1, 3, 127, 127).cuda()

    out = net(var)
    for o in out:
        print(o.size())
