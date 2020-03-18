import argparse
import json
import logging
import os

import torch

from utils.log_helper import init_log, add_file_handler
from utils.misc import commit
from utils.model_load import load_pretrain
from models.gdp_siam_model import GDPSiamModel
from configs.config import cfg

logger = logging.getLogger('global')


def pruning_model(model):
    # backbone
    last_mask = None
    # layer0
    layer_name = 'layer0'
    layer = getattr(model.backbone, layer_name)
    for idx, m in enumerate(layer):  # layer is a Sequence
        if isinstance(m, torch.nn.Conv2d):
            is_pruning = False
            in_channels, out_channels = m.in_channels, m.out_channels
            if not last_mask is None:
                in_channels = int(last_mask.sum().item())
            state_name = 'backbone.{}.{}.weight'.format(layer_name, idx)
            if state_name in model.mask.keys():  # modify the conv kernel which is pruned
                is_pruning = True
                out_channels = int(model.mask[state_name].sum().item())
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
            if out_channels == m.out_channels:
                out_mask = torch.ones(out_channels).cuda() == 1
            else:
                out_mask = model.mask[state_name] == 1
            new_conv.weight = torch.nn.Parameter(m.weight[out_mask, :, :, :][:, in_mask, :, :], requires_grad=False)
            layer[idx] = new_conv
            # modify the relative batchnorm
            batchnorm = layer[idx + 1]
            new_batchnorm = torch.nn.BatchNorm2d(num_features=out_channels)
            new_batchnorm.weight = torch.nn.Parameter(batchnorm.weight[out_mask], requires_grad=False)
            new_batchnorm.bias = torch.nn.Parameter(batchnorm.bias[out_mask], requires_grad=False)
            new_batchnorm.running_mean=batchnorm.running_mean[out_mask]
            new_batchnorm.running_var=batchnorm.running_var[out_mask]
            layer[idx + 1] = new_batchnorm
            if is_pruning:
                last_mask = model.mask[state_name]
            else:
                last_mask = None

    # layer1-7
    for i in range(1, 8):  # for layer
        layer_name = 'layer{}'.format(i)
        layer = getattr(model.backbone, layer_name)
        for block_idx, block in enumerate(layer):  # for every block in the layer
            idx = 0
            block = block.conv
            for idx, m in enumerate(block):  # the block has attr: conv, conv is Squential
                if isinstance(m, torch.nn.Conv2d):
                    is_pruning = False
                    is_depthwise = False
                    in_channels, out_channels = m.in_channels, m.out_channels
                    if not last_mask is None:
                        in_channels = int(last_mask.sum().item())
                    state_name = 'backbone.{}.{}.conv.{}.weight'.format(layer_name, block_idx, idx)
                    if state_name in model.mask.keys():  # modify the conv kernel which is pruned
                        is_pruning = True
                        out_channels = int(model.mask[state_name].sum().item())
                    if m.groups == 1:
                        new_conv = torch.nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=m.kernel_size,
                            stride=m.stride,
                            padding=m.padding,
                            dilation=m.dilation,
                            bias=False if m.bias is None else True
                        )
                    else:
                        is_depthwise = True
                        out_channels = in_channels
                        new_conv = torch.nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=out_channels,  # depth-wise the output channel depend on the inputchanel
                            kernel_size=m.kernel_size,
                            stride=m.stride,
                            padding=m.padding,
                            dilation=m.dilation,
                            bias=False if m.bias is None else False,
                            groups=in_channels
                        )
                    # copy the new weight
                    if in_channels == m.in_channels:
                        in_mask = torch.ones(in_channels).cuda() == 1
                    else:
                        in_mask = last_mask == 1
                    if out_channels == m.out_channels:
                        out_mask = torch.ones(out_channels).cuda() == 1
                    elif is_depthwise:
                        out_mask = in_mask
                    else:
                        out_mask = model.mask[state_name] == 1
                    if not is_depthwise:
                        new_conv.weight = torch.nn.Parameter(m.weight[out_mask, :, :, :][:, in_mask, :, :],
                                                             requires_grad=False)
                    else:
                        new_conv.weight = torch.nn.Parameter(m.weight[in_mask, :, :, :], requires_grad=False)

                    block[idx] = new_conv
                    # modify the relative batchnorm
                    batchnorm = block[idx + 1]
                    new_batchnorm = torch.nn.BatchNorm2d(num_features=out_channels)
                    new_batchnorm.weight = torch.nn.Parameter(batchnorm.weight[out_mask], requires_grad=False)
                    new_batchnorm.bias = torch.nn.Parameter(batchnorm.bias[out_mask], requires_grad=False)
                    new_batchnorm.running_mean=batchnorm.running_mean[out_mask]
                    new_batchnorm.running_var=batchnorm.running_var[out_mask]
                    block[idx + 1] = new_batchnorm
                    if is_pruning:
                        last_mask = model.mask[state_name]
                    elif is_depthwise:
                        last_mask = last_mask
                    else:
                        last_mask = None
    # for neck
    last_mask = [None, None, None]  # None or not?
    branch_name = ['downsample2', 'downsample3', 'downsample4']
    for i in range(3):  # three branch
        block = getattr(model.neck, branch_name[i])
        block = getattr(block, 'downsample')
        for idx, m in enumerate(block):
            if isinstance(m, torch.nn.Conv2d):
                is_pruning = False
                in_channels, out_channels = m.in_channels, m.out_channels
                if not last_mask[i] is None:
                    in_channels = int(last_mask[i].sum().item())
                state_name = 'neck.{}.downsample.{}.weight'.format(branch_name[i], idx)
                if state_name in model.mask.keys():  # modify the conv kernel which is pruned
                    is_pruning = True
                    out_channels = int(model.mask[state_name].sum().item())
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
                if out_channels == m.out_channels:
                    out_mask = torch.ones(out_channels).cuda() == 1
                else:
                    out_mask = model.mask[state_name] == 1
                new_conv.weight = torch.nn.Parameter(m.weight[out_mask, :, :, :][:, in_mask, :, :], requires_grad=False)
                block[idx] = new_conv
                # modify the relative batchnorm
                if idx + 1 == len(block):
                    continue
                batchnorm = block[idx + 1]
                new_batchnorm = torch.nn.BatchNorm2d(num_features=out_channels)
                new_batchnorm.weight = torch.nn.Parameter(batchnorm.weight[out_mask], requires_grad=False)
                new_batchnorm.bias = torch.nn.Parameter(batchnorm.bias[out_mask], requires_grad=False)
                new_batchnorm.running_mean=batchnorm.running_mean[out_mask]
                new_batchnorm.running_var=batchnorm.running_var[out_mask]
                block[idx + 1] = new_batchnorm
                if is_pruning:
                    last_mask[i] = model.mask[state_name]
                else:
                    last_mask[i] = None
            #     idx += 1
            # elif isinstance(m, torch.nn.BatchNorm2d):
            #     idx += 1
    # for rpn
    branchs = ['cls', 'loc']
    head_names = ['head2', 'head3', 'head4']
    neck_last_mask = [
        model.mask['neck.downsample2.downsample.0.weight'],
        model.mask['neck.downsample3.downsample.0.weight'],
        model.mask['neck.downsample4.downsample.0.weight']
    ]
    for branch in branchs:
        for i, head_name in enumerate(head_names):
            # for conv kernel and conv search, because the neck is changed
            head = getattr(model.rpn, head_name)
            # kernel
            block = getattr(head, branch).conv_kernel
            m = block[0]
            in_channels, out_channels = m.in_channels, m.out_channels
            if not neck_last_mask[i] is None:
                in_channels = int(neck_last_mask[i].sum().item())
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
            in_mask = neck_last_mask[i] == 1
            new_conv.weight = torch.nn.Parameter(m.weight[:, in_mask, :, :], requires_grad=False)
            block[0] = new_conv
            # search
            block = getattr(head, branch).conv_search
            m = block[0]
            in_channels, out_channels = m.in_channels, m.out_channels
            if not neck_last_mask[i] is None:
                in_channels = int(neck_last_mask[i].sum().item())
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
            in_mask = neck_last_mask[i] == 1
            new_conv.weight = torch.nn.Parameter(m.weight[:, in_mask, :, :], requires_grad=False)
            block[0] = new_conv

            # frist conv
            head = getattr(model.rpn, head_name)
            head = getattr(head, branch).head
            state_name = 'rpn.{}.{}.head.{}.weight'.format(head_name, branch, 0)
            m = head[0]
            in_channels, out_channels = m.in_channels, m.out_channels
            if state_name in model.mask.keys():
                out_channels = int(model.mask[state_name].sum().item())
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
            out_mask = model.mask[state_name] == 1
            new_conv.weight = torch.nn.Parameter(m.weight[out_mask, :, :, :], requires_grad=False)
            head[0] = new_conv
            last_mask = model.mask[state_name]
            # batch_norm
            batchnorm = head[1]
            new_batchnorm = torch.nn.BatchNorm2d(num_features=out_channels)
            new_batchnorm.weight = torch.nn.Parameter(batchnorm.weight[out_mask], requires_grad=False)
            new_batchnorm.bias = torch.nn.Parameter(batchnorm.bias[out_mask], requires_grad=False)
            new_batchnorm.running_mean = batchnorm.running_mean[out_mask]
            new_batchnorm.running_var = batchnorm.running_var[out_mask]
            head[1] = new_batchnorm
            # second_conv
            m = head[3]
            in_channels = int(last_mask.sum().item())
            new_conv = torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=m.out_channels,
                kernel_size=m.kernel_size,
                stride=m.stride,
                padding=m.padding,
                dilation=m.dilation,
                bias=False if m.bias is None else True,
            )
            # copy the new weight
            in_mask = (last_mask == 1)
            new_conv.weight = torch.nn.Parameter(m.weight[:, in_mask, :, :], requires_grad=False)
            head[3] = new_conv
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='', type=str, help='which config file to use')
    parser.add_argument('--snapshot', default='', type=str, help='which model to pruning')
    args = parser.parse_args()
    cfg.merge_from_file(args.cfg)
    if not os.path.exists(cfg.PRUNING.LOG_DIR):
        os.makedirs(cfg.PRUNING.LOG_DIR)
    init_log('global', logging.INFO)
    if cfg.PRUNING.LOG_DIR:
        add_file_handler('global',
                         os.path.join(cfg.PRUNING.LOG_DIR, 'logs.txt'),
                         logging.INFO)
    logger.info("Version Information: \n{}\n".format(commit()))
    logger.info("config \n{}".format(json.dumps(cfg, indent=4)))
    model = GDPSiamModel()
    model = load_pretrain(model, args.snapshot)
    print(model.mask_scores)

    for k, v in model.mask.items():
        print(k, v)
    model = pruning_model(model)

    torch.save(model.state_dict(), './snapshot/mobilenetv2_gdp/model_pruning.pth')
