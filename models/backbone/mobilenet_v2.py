import torch
import torch.nn as nn


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidualsBlock(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, dilation=1):
        super(InvertedResidualsBlock, self).__init__()
        self.stride = stride

        self.use_res_connect = self.stride == 1 and inp == oup

        padding = 2 - stride
        if dilation > 1:
            padding = dilation

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3,
                      stride, padding, dilation=dilation, # padding differnt from mobilenetv2
                      groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, width_mult=1.0, used_layers=[3, 5, 7]):
        super(MobileNetV2, self).__init__()
        self.used_layers=used_layers
        in_channels = int(32 * width_mult)
        self.layer0 = nn.Sequential(
            nn.Conv2d(3, in_channels, 3, 2, 0, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True)
        )
        self.interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1, 1],
            [6, 24, 2, 2, 1],
            [6, 32, 3, 2, 1],
            [6, 64, 4, 1, 2],
            [6, 96, 3, 1, 2],
            [6, 160, 3, 1, 4],
            [6, 320, 1, 1, 4],
        ]
        last_dilation = 1
        for idx, (t, c, n, s, d) in enumerate(self.interverted_residual_setting, start=1):
            # out_channels = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            out_channels=int(c*width_mult)
            bottleneck = []
            for i in range(n):
                if i == 0:
                    if d == last_dilation:
                        dd = d
                    else:
                        dd = max(d // 2, 1)
                    bottleneck.append(InvertedResidualsBlock(in_channels, out_channels, s,t, dd))
                else:
                    bottleneck.append(InvertedResidualsBlock(out_channels, out_channels, 1,t, d))
            last_dilation = d
            in_channels = out_channels
            self.add_module('layer{}'.format(idx), nn.Sequential(*bottleneck))

    def forward(self, x):
        outputs = []
        for i in range(8):
            name = 'layer{}'.format(i)
            x = getattr(self, name)(x)
            outputs.append(x)
        outs = [outputs[i] for i in self.used_layers]
        if len(outs) == 1:
            return outs[0]
        else:
            return outs
        # x=getattr(self,'layer0')[0](x)
        # x=getattr(self,'layer0')[1](x)
        # return x
        

def mobilenet_v2(**kwargs):
    return MobileNetV2(**kwargs)


if __name__ == '__main__':
    net = MobileNetV2(width_mult=1.4)
    print(net)
    input = torch.ones(1, 3, 255, 255, requires_grad=True)
    out = net(input)
    for i, o in enumerate(out):
        print(i, o.size())
