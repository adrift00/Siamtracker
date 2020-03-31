import argparse
import json
import logging
import os

import torch

from utils.log_helper import init_log, add_file_handler
from utils.misc import commit
from utils.model_load import load_pretrain
from models.pruning_siam_model import PruningSiamModel
from configs.config import cfg

logger = logging.getLogger('global')


def prune_conv(block, prune_mask, last_mask):
    # conv
    block.in_channels = int(last_mask.sum())
    block.out_channels = int(prune_mask.sum())
    if block.groups != 1:  # for deep wise conv
        block.groups = int(prune_mask.sum())
    block.weight.data = block.weight[prune_mask, :, :, :][:, last_mask, :, :]
    block.weight.requires_grad_(False)


def prune_bn(block, prune_mask):
    block.num_features = int(prune_mask.sum())
    block.weight.data = block.weight[prune_mask]
    block.bias.data = block.bias[prune_mask]
    block.running_mean.data = block.running_mean[prune_mask]
    block.running_var.data = block.running_var[prune_mask]

    block.weight.requires_grad_(False)
    block.bias.requires_grad_(False)
    block.running_mean.requires_grad_(False)
    block.running_var.requires_grad_(False)


def prune_conv_bn(block, prune_mask, last_mask):
    # conv
    prune_conv(block[0], prune_mask, last_mask)
    # bn
    prune_bn(block[1], prune_mask)


def prune_inv_residual(block, prune_mask, last_mask):
    # point wise
    prune_conv_bn(block[0:2], prune_mask, last_mask)
    # deep wise
    last_mask = (torch.ones(1) == 1)
    prune_conv_bn(block[3:5], prune_mask, last_mask)
    # point wise
    last_mask = prune_mask
    prune_mask = (torch.ones(block[-1].weight.size(0)) == 1)
    prune_conv_bn(block[6:], prune_mask, last_mask)


def prune_model(model):
    # backbone
    # layer0
    layer_name = 'layer0'
    layer = getattr(model.backbone, layer_name)
    last_mask = (torch.ones(layer[0].in_channels) == 1)
    state_name = 'backbone.{}.{}.weight'.format(layer_name, 0)
    if state_name in model.mask.keys():  # modify the conv kernel which is pruned
        prune_mask = (model.mask[state_name] == 1)
    else:
        prune_mask = (torch.ones(layer[0].out_channels) == 1)
    prune_conv_bn(layer, prune_mask, last_mask)
    last_mask = prune_mask
    # layer1-7
    for i in range(1, 8):  # for layer1-7
        layer_name = 'layer{}'.format(i)
        layer = getattr(model.backbone, layer_name)
        for block_idx, block in enumerate(layer):  # for every block in the layer
            block = block.conv
            state_name = 'backbone.{}.{}.conv.{}.weight'.format(layer_name, block_idx, 0)
            if state_name in model.mask.keys():
                prune_mask = model.mask[state_name] == 1
            else:
                prune_mask = (torch.ones(block[0].out_channels) == 1)
            # point wise
            prune_inv_residual(block, prune_mask, last_mask)
            last_mask = (torch.ones(block[-1].weight.size(0)) == 1)

    # for neck
    branch_name = ['downsample2', 'downsample3', 'downsample4']
    for i in range(3):  # three branch
        block = getattr(model.neck, branch_name[i])
        block = getattr(block, 'downsample')
        state_name = 'neck.{}.downsample.{}.weight'.format(branch_name[i], 0)
        if state_name in model.mask.keys():  # modify the conv kernel which is pruned
            prune_mask = model.mask[state_name] == 1
        else:
            prune_mask = torch.ones(block[0].out_channels) == 1
        last_mask = (torch.ones(block[0].in_channels) == 1)
        prune_conv_bn(block, prune_mask, last_mask)

    # for rpn
    branchs = ['cls', 'loc']
    head_names = ['head2', 'head3', 'head4']
    last_masks = [
        model.mask['neck.downsample2.downsample.0.weight'] == 1,
        model.mask['neck.downsample3.downsample.0.weight'] == 1,
        model.mask['neck.downsample4.downsample.0.weight'] == 1
    ]
    for branch in branchs:
        for i, head_name in enumerate(head_names):
            # for conv kernel and conv search, because the neck is changed
            head = getattr(model.rpn, head_name)
            # kernel
            block = getattr(head, branch).conv_kernel
            last_mask = last_masks[i]
            prune_mask = (torch.ones(block[0].out_channels) == 1)
            prune_conv_bn(block, prune_mask, last_mask)
            # search
            block = getattr(head, branch).conv_search
            last_mask = last_masks[i]
            prune_mask = (torch.ones(block[0].out_channels) == 1)
            prune_conv_bn(block, prune_mask, last_mask)

            # frist conv
            head = getattr(model.rpn, head_name)
            head = getattr(head, branch).head
            block = head[0:3]
            state_name = 'rpn.{}.{}.head.{}.weight'.format(head_name, branch, 0)
            if state_name in model.mask.keys():  # modify the conv kernel which is pruned
                prune_mask = model.mask[state_name] == 1
            else:
                prune_mask = torch.ones(block[0].out_channels) == 1
            last_mask = (torch.ones(block[0].in_channels) == 1)
            prune_conv_bn(block[0:], prune_mask, last_mask)
            last_mask = prune_mask
            block = head[3]
            prune_mask = (torch.ones(block.out_channels) == 1)
            # second conv
            prune_conv(block, prune_mask, last_mask)
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
    model = PruningSiamModel()
    model = load_pretrain(model, args.snapshot)

    for k, v in model.mask.items():
        print(k, v)
    model = prune_model(model)

    # torch.save(model.state_dict(), './snapshot/mobilenetv2_gdp/model_pruning.pth')
