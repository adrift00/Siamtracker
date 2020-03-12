import argparse

import torch
from utils.model_load import load_pretrain
from models.gdp_siam_model import GDPSiamModel
from configs.config import cfg

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default='', type=str, help='which config file to use')
parser.add_argument('--snapshot', default='', type=str, help='which model to pruning')
args = parser.parse_args()


def pruning_model(model):
    # backbone
    last_mask = None
    # layer0
    layer_name = 'layer0'
    layer = getattr(model.backbone, layer_name)
    idx = 0  # for the idx in the state dict, ignore the relu
    for i, m in enumerate(layer):  # layer is a Sequence
        if isinstance(m, torch.nn.Conv2d):
            is_pruning = False
            in_channels, out_channels = m.in_channels, m.out_channels
            if not last_mask is None:
                in_channels = last_mask.sum().item()
            state_name = 'backbone.{}.{}.weight'.format(layer_name, idx)
            if state_name in model.mask.keys():  # modify the conv kernel which is pruned
                is_pruning = True
                out_channels = model.mask[state_name].sum().item()
            new_conv = torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=m.kernel_size,
                stride=m.stride,
                padding=m.padding,
                dilation=m.dilation,
                bias=False if m.bias is None else True,
            )
            # copy the new weight
            if in_channels == m.in_channels:
                in_mask = torch.ones(in_channels).cuda() == 1
            else:
                in_mask = last_mask == 1
            if out_channels == m.out_channels
                out_mask = torch.ones(out_channels).cuda() == 1
            else:
                out_mask = model.mask[state_name] == 1
            new_conv.weight = m.weight[out_mask, in_mask, :, :]
            layer[i] = new_conv
            # modify the relative batchnorm
            batchnorm = layer[i + 1]
            new_batchnorm = torch.nn.BatchNorm2d(num_features=out_channels)
            new_batchnorm.weight = batchnorm.weight[out_mask]
            new_batchnorm.bias = batchnorm.bias[out_mask]
            layer[i + 1] = new_batchnorm
            if is_pruning:
                last_mask = model.mask[state_name]
            else:
                last_mask =None
            idx += 1
        elif isinstance(m, torch.nn.BatchNorm2d):
            idx += 1

    # layer1-7
    for i in range(1, 8):  # for layer
        layer_name = 'layer{}'.format(i)
        layer = getattr(model.backbone, layer_name)
        for block_idx, block in enumerate(layer):  # for every block in the layer
            idx = 0
            block=block.conv
            for i, m in enumerate(block.conv):  # the block has attr: conv, conv is Squential
                if isinstance(m, torch.nn.Conv2d):
                    is_pruning = False
                    in_channels, out_channels = m.in_channels, m.out_channels
                    if not last_mask is None:
                        in_channels = last_mask.sum().item()
                    state_name = 'backbone.{}.{}.weight'.format(layer_name, idx)
                    if state_name in model.mask.keys():  # modify the conv kernel which is pruned
                        is_pruning = True
                        out_channels = model.mask[state_name].sum().item()
                    new_conv = torch.nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=m.kernel_size,
                        stride=m.stride,
                        padding=m.padding,
                        dilation=m.dilation,
                        bias=False if m.bias is None else True,
                    )
                    # copy the new weight
                    if in_channels == m.in_channels:
                        in_mask = torch.ones(in_channels).cuda() == 1
                    else:
                        in_mask = last_mask == 1
                    if out_channels == m.out_channels
                        out_mask = torch.ones(out_channels).cuda() == 1
                    else:
                        out_mask = model.mask[state_name] == 1
                    new_conv.weight = m.weight[out_mask, in_mask, :, :]
                    block[i] = new_conv
                    # modify the relative batchnorm
                    batchnorm = block[i + 1]
                    new_batchnorm = torch.nn.BatchNorm2d(num_features=out_channels)
                    new_batchnorm.weight = batchnorm.weight[out_mask]
                    new_batchnorm.bias = batchnorm.bias[out_mask]
                    block[i + 1] = new_batchnorm
                    if is_pruning:
                        last_mask = model.mask[state_name]
                    else:
                        last_mask =None
                    idx += 1
                elif isinstance(m, torch.nn.BatchNorm2d):
                    idx += 1
    # for neck
    last_mask=[None,None,None] # None or not?
    branch_name=['downsample2','downsample3','downsample4']
    for i in range(3): # three branch
        block=getattr(model.neck,branch_name[i])
        idx=0
        block=getattr(block,'downsample')
        for m in block:
            if isinstance(m, torch.nn.Conv2d):
                is_pruning = False
                in_channels, out_channels = m.in_channels, m.out_channels
                if not last_mask[i] is None:
                    in_channels = last_mask[i].sum().item()
                state_name='neck.{}.downsample.{}.weight'.format(branch_name[i],idx)
                if state_name in model.mask.keys():  # modify the conv kernel which is pruned
                    is_pruning = True
                    out_channels = model.mask[state_name].sum().item()
                new_conv = torch.nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=m.kernel_size,
                    stride=m.stride,
                    padding=m.padding,
                    dilation=m.dilation,
                    bias=False if m.bias is None else True,
                )
                # copy the new weight
                if in_channels == m.in_channels:
                    in_mask = torch.ones(in_channels).cuda() == 1
                else:
                    in_mask = last_mask[i] == 1
                if out_channels == m.out_channels
                    out_mask = torch.ones(out_channels).cuda() == 1
                else:
                    out_mask = model.mask[state_name] == 1
                new_conv.weight = m.weight[out_mask, in_mask, :, :]
                block[i] = new_conv
                # modify the relative batchnorm
                batchnorm = block[i + 1]
                new_batchnorm = torch.nn.BatchNorm2d(num_features=out_channels)
                new_batchnorm.weight = batchnorm.weight[out_mask]
                new_batchnorm.bias = batchnorm.bias[out_mask]
                block[i + 1] = new_batchnorm
                if is_pruning:
                    last_mask[i] = model.mask[state_name]
                else:
                    last_mask[i] =None
                idx += 1
            elif isinstance(m, torch.nn.BatchNorm2d):
                idx += 1
    # for rpn












if __name__ == '__main__':
    cfg.merge_from_file(args.cfg)
    model = GDPSiamModel()

    model = load_pretrain(model, args.snapshot)
