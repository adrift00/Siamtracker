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

def prune_conv_bn(block,prune_mask,last_mask):
    # conv
    block[0].in_channels=int(last_mask.sum())
    block[0].out_channels=int(prune_mask.sum())
    block[0].weight.data=block[0].weight[prune_mask,:,:,:][:,last_mask,:,:].clone()
    # bn
    block[1].num_features=int(prune_mask.sum())
    block[1].weight.data = block[1].weight[prune_mask].clone()
    block[1].bias.data = block[1].bias[prune_mask].clone()
    block[1].running_mean.data = block[1].running_mean[prune_mask].clone()
    block[1].running_var.data = block[1].running_var[prune_mask].clone()

def prune_conv(block,prune_mask,last_mask):
    # conv
    block[0].in_channels=int(last_mask.sum())
    block[0].out_channels=int(prune_mask.sum())
    block[0].weight.data=block[0].weight[prune_mask,:,:,:][:,last_mask,:,:].clone()

def prune_inv_res(block,prune_mask,last_mask):
    # point wise
    block[0].in_channels=int(last_mask.sum())
    block[0].out_channels=int(prune_mask.sum())
    block[0].weight.data=block[0].weight[prune_mask,:,:,:][:,last_mask,:,:].clone()

    block[1].num_features=int(prune_mask.sum())
    block[1].weight.data = block[1].weight[prune_mask].clone()
    block[1].bias.data = block[1].bias[prune_mask].clone()
    block[1].running_mean.data = block[1].running_mean[prune_mask].clone()
    block[1].running_var.data = block[1].running_var[prune_mask].clone()
    # deep wise
    block[3].in_channels=int(prune_mask.sum())
    block[3].out_channels=int(prune_mask.sum())
    block[3].groups=prune_mask.sum()
    block[3].weight.data=block[3].weight[prune_mask,:,:,:].clone()

    block[4].num_features=int(prune_mask.sum())
    block[4].weight.data = block[4].weight[prune_mask].clone()
    block[4].bias.data = block[4].bias[prune_mask].clone()
    block[4].running_mean.data = block[4].running_mean[prune_mask].clone()
    block[4].running_var.data = block[4].running_var[prune_mask].clone()
    # point wise
    block[6].in_channels=int(prune_mask.sum())
    block[6].out_channels=block[6].out_channels
    block[6].weight.data=block[6].weight[:,prune_mask,:,:].clone()

    block[7].num_features=block[7].num_features
    block[7].weight.data = block[7].weight[:].clone()
    block[7].bias.data = block[7].bias[:].clone()
    block[7].running_mean.data = block[7].running_mean[:].clone()
    block[7].running_var.data = block[7].running_var[:].clone()





def prune_model(model):
    # backbone
    # layer0
    layer_name = 'layer0'
    layer = getattr(model.backbone, layer_name)
    last_mask=(torch.ones(layer[0].in_channels)==1)
    state_name = 'backbone.{}.{}.weight'.format(layer_name, 0)
    if state_name in model.mask.keys():  # modify the conv kernel which is pruned
        prune_mask=(model.mask[state_name]==1)
    else:
        prune_mask=(torch.ones(layer[0].out_channels)==1)
    prune_conv_bn(layer,prune_mask,last_mask)
    last_mask=prune_mask
    # layer1-7
    for i in range(1, 8):  # for layer1-7
        layer_name = 'layer{}'.format(i)
        layer = getattr(model.backbone, layer_name)
        for block_idx, block in enumerate(layer):  # for every block in the layer
            block = block.conv
            state_name = 'backbone.{}.{}.conv.{}.weight'.format(layer_name, block_idx, 0)
            if state_name in model.mask.keys():
                prune_mask=model.mask[state_name]==1
            else:
                prune_mask=(torch.ones(block[0].out_channels)==1)
            # point wise
            prune_inv_res(block,prune_mask,last_mask)
            last_mask=(torch.ones(block[-1].weight.size(0))==1)

    # for neck
    branch_name = ['downsample2', 'downsample3', 'downsample4']
    for i in range(3):  # three branch
        block = getattr(model.neck, branch_name[i])
        block = getattr(block, 'downsample')
        state_name='neck.{}.downsample.{}.weight'.format(branch_name[i], 0)
        if state_name in model.mask.keys():  # modify the conv kernel which is pruned
            prune_mask=model.mask[state_name]==1
        else:
            prune_mask=torch.ones(block[0].out_channels)==1
        last_mask=(torch.ones(block[0].in_channels)==1)
        prune_conv_bn(block,prune_mask,last_mask)

    # for rpn
    branchs = ['cls', 'loc']
    head_names = ['head2', 'head3', 'head4']
    last_masks = [
        model.mask['neck.downsample2.downsample.0.weight']==1,
        model.mask['neck.downsample3.downsample.0.weight']==1,
        model.mask['neck.downsample4.downsample.0.weight']==1
    ]
    for branch in branchs:
        for i, head_name in enumerate(head_names):
            # for conv kernel and conv search, because the neck is changed
            head = getattr(model.rpn, head_name)
            # kernel
            block = getattr(head, branch).conv_kernel
            last_mask=last_masks[i]
            prune_mask=(torch.ones(block[0].out_channels)==1)
            prune_conv_bn(block,prune_mask,last_mask)
            # search
            block = getattr(head, branch).conv_search
            last_mask=last_masks[i]
            prune_mask=(torch.ones(block[0].out_channels)==1)
            prune_conv_bn(block,prune_mask,last_mask)

            # frist conv
            head = getattr(model.rpn, head_name)
            head = getattr(head, branch).head
            block=head[0:3]
            state_name = 'rpn.{}.{}.head.{}.weight'.format(head_name, branch, 0)
            if state_name in model.mask.keys():  # modify the conv kernel which is pruned
                prune_mask=model.mask[state_name]==1
            else:
                prune_mask=torch.ones(block[0].out_channels)==1
            last_mask=(torch.ones(block[0].in_channels)==1)
            prune_conv_bn(block[0:],prune_mask,last_mask)
            last_mask=prune_mask
            block=head[3:]
            prune_mask=(torch.ones(block[0].out_channels)==1)
            # second conv
            prune_conv(block,prune_mask,last_mask)
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
    print(model.mask_scores)

    for k, v in model.mask.items():
        print(k, v)
    model = prune_model(model)

    torch.save(model.state_dict(), './snapshot/mobilenetv2_gdp/model_pruning.pth')
