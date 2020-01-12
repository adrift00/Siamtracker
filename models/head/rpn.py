import torch.nn as nn
import torch.nn.functional as F
import torch


class RPN(nn.Module):
    def __init__(self):
        super(RPN, self).__init__()

    def forward(self, z_f, x_f):
        raise NotImplementedError


class DepthwiseRPN(RPN):
    def __init__(self, anchor_num=5, in_channels=256, out_channels=256):
        super(DepthwiseRPN, self).__init__()
        self.cls = DepthwiseXCorr(in_channels, out_channels, 2 * anchor_num)
        self.loc = DepthwiseXCorr(in_channels, out_channels, 4 * anchor_num)

    def forward(self, z_f, x_f, weight=None, bn_weight=None):
        if weight is None and bn_weight is None:
            cls = self.cls(z_f, x_f)
            loc = self.loc(z_f, x_f)
            return cls, loc
        else:
            # cls
            # cls_kernel
            cls_kernel = F.conv2d(z_f, weight['cls.conv_kernel.0.weight'])
            cls_kernel = F.batch_norm(cls_kernel, bn_weight['cls.conv_kernel.1.running_mean'],
                                      bn_weight['cls.conv_kernel.1.running_var'],
                                      weight['cls.conv_kernel.1.weight'],
                                      weight['cls.conv_kernel.1.bias'])
            cls_kernel = F.relu(cls_kernel, inplace=True)
            # cls_search
            cls_search = F.conv2d(x_f, weight['cls.conv_search.0.weight'])
            cls_search = F.batch_norm(cls_search, bn_weight['cls.conv_search.1.running_mean'],
                                      bn_weight['cls.conv_search.1.running_var'],
                                      weight['cls.conv_search.1.weight'],
                                      weight['cls.conv_search.1.bias'])
            cls_search = F.relu(cls_search, inplace=True)
            # x_corr
            cls_feat = xcorr_depthwise(cls_search, cls_kernel)
            # head
            cls_feat = F.conv2d(cls_feat, weight['cls.head.0.weight'])
            cls_feat = F.batch_norm(cls_feat, bn_weight['cls.head.1.running_mean'],
                                    bn_weight['cls.head.1.running_var'],
                                    weight['cls.head.1.weight'],
                                    weight['cls.head.1.bias'])
            cls_feat = F.relu(cls_feat, inplace=True)
            cls = F.conv2d(
                cls_feat, weight['cls.head.3.weight'], weight['cls.head.3.bias'])
            # loc
            # loc_kernel
            loc_kernel = F.conv2d(z_f, weight['loc.conv_kernel.0.weight'])
            loc_kernel = F.batch_norm(loc_kernel, bn_weight['loc.conv_kernel.1.running_mean'],
                                      bn_weight['loc.conv_kernel.1.running_var'],
                                      weight['loc.conv_kernel.1.weight'],
                                      weight['loc.conv_kernel.1.bias'])
            loc_kernel = F.relu(loc_kernel, inplace=True)
            # loc_search
            loc_search = F.conv2d(x_f, weight['loc.conv_search.0.weight'])
            loc_search = F.batch_norm(loc_search, bn_weight['loc.conv_search.1.running_mean'],
                                      bn_weight['loc.conv_search.1.running_var'],
                                      weight['loc.conv_search.1.weight'],
                                      weight['loc.conv_search.1.bias'])
            loc_search = F.relu(loc_search, inplace=True)
            # x_corr
            loc_feat = xcorr_depthwise(loc_search, loc_kernel)
            # head
            loc_feat = F.conv2d(loc_feat, weight['loc.head.0.weight'])
            loc_feat = F.batch_norm(loc_feat, bn_weight['loc.head.1.running_mean'],
                                    bn_weight['loc.head.1.running_var'],
                                    weight['loc.head.1.weight'],
                                    weight['loc.head.1.bias'])
            loc_feat = F.relu(loc_feat, inplace=True)
            loc = F.conv2d(
                loc_feat, weight['loc.head.3.weight'], weight['loc.head.3.bias'])
            return cls, loc


class MultiRPN(RPN):
    def __init__(self,  in_channels, anchor_num=5, weighted=False):
        super(MultiRPN, self).__init__()
        self.weighted = weighted
        for i in range(len(in_channels)):
            self.add_module('head' + str(i + 2),
                            DepthwiseRPN(anchor_num, in_channels[i], in_channels[i]))
        if self.weighted:
            self.cls_weight = nn.Parameter(torch.ones(len(in_channels)))
            self.loc_weight = nn.Parameter(torch.ones(len(in_channels)))

    def forward(self, z_fs, x_fs):
        cls = []
        loc = []
        for idx, (z_f, x_f) in enumerate(zip(z_fs, x_fs), start=2):
            rpn = getattr(self, 'head' + str(idx))
            c, l = rpn(z_f, x_f)
            cls.append(c)
            loc.append(l)

        if self.weighted:
            cls_weight = F.softmax(self.cls_weight, 0)
            loc_weight = F.softmax(self.loc_weight, 0)

        def avg(lst):
            return sum(lst) / len(lst)

        def weighted_avg(lst, weight):
            s = 0
            for i in range(len(weight)):
                s += lst[i] * weight[i]
            return s

        if self.weighted:
            return weighted_avg(cls, cls_weight), weighted_avg(loc, loc_weight)
        else:
            return avg(cls), avg(loc)


class DepthwiseXCorr(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, kernel_size=3, hidden_kernel_size=5):
        super(DepthwiseXCorr, self).__init__()
        self.conv_kernel = nn.Sequential(
            nn.Conv2d(in_channels, hidden,
                      kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.conv_search = nn.Sequential(
            nn.Conv2d(in_channels, hidden,
                      kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_channels, kernel_size=1)
        )

    def forward(self, kernel, search):
        kernel = self.conv_kernel(kernel)
        search = self.conv_search(search)
        feature = xcorr_depthwise(search, kernel)
        out = self.head(feature)
        return out

def xcorr_depthwise(x, kernel):
    """depthwise cross correlation
    """
    batch = kernel.size(0)
    channel = kernel.size(1)
    x = x.view(1, batch*channel, x.size(2), x.size(3))
    kernel = kernel.view(batch*channel, 1, kernel.size(2), kernel.size(3))
    out = F.conv2d(x, kernel, groups=batch*channel)
    out = out.view(batch, channel, out.size(2), out.size(3))
    return out





