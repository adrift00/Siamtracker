# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

import torch

logger = logging.getLogger('global')


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())

    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    # filter 'num_batches_tracked'
    missing_keys = [x for x in missing_keys
                    if not x.endswith('num_batches_tracked')]
    if len(missing_keys) > 0:
        logger.info('[Warning] missing keys: {}'.format(missing_keys))
        logger.info('missing keys:{}'.format(len(missing_keys)))
    if len(unused_pretrained_keys) > 0:
        logger.info('[Warning] unused_pretrained_keys: {}'.format(
            unused_pretrained_keys))
        logger.info('unused checkpoint keys:{}'.format(
            len(unused_pretrained_keys)))
    logger.info('used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, \
        'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters
    share common prefix 'module.' '''
    logger.info('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def meta_load(load_pretrain):
    def wrapper(model, pretrained_path):
        model = load_pretrain(model, pretrained_path)
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path,
                                     map_location=lambda storage, loc: storage.cuda(device))
        model.init_weight = pretrained_dict['init_weight']
        model.alpha = pretrained_dict['alpha']
        model.bn_weight = pretrained_dict['bn_weight']
        return model

    return wrapper


# @meta_load
def load_pretrain(model, pretrained_path):
    logger.info('load pretrained model from {}'.format(pretrained_path))
    device = torch.cuda.current_device()
    pretrained_dict = torch.load(pretrained_path,
                                 map_location=lambda storage, loc: storage.cuda(device))
    # for meta
    if 'init_weight' in pretrained_dict.keys() \
            and 'alpha' in pretrained_dict.keys() \
            and 'bn_weight' in pretrained_dict.keys():
        model.init_weight = pretrained_dict['init_weight']
        model.alpha = pretrained_dict['alpha']
        model.bn_weight = pretrained_dict['bn_weight']
    # for prune model
    if 'mask' in pretrained_dict.keys() \
            and 'mask_scores' in pretrained_dict.keys():
        model.mask=pretrained_dict['mask']
        model.mask_scores=pretrained_dict['mask_scores']

    if "model" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['model'],
                                        'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')

    try:
        check_keys(model, pretrained_dict)
    except:
        logger.info('[Warning]: using pretrain as features.\
                Adding "features." as prefix')
        new_dict = {}
        for k, v in pretrained_dict.items():
            k = 'features.' + k
            new_dict[k] = v
        pretrained_dict = new_dict
        check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)

    return model


def restore_from(model, optimizer, ckpt_path):
    device = torch.cuda.current_device()
    ckpt = torch.load(ckpt_path,
                      map_location=lambda storage, loc: storage.cuda(device))
    epoch = ckpt['epoch']
    ckpt_model_dict = remove_prefix(ckpt['model'], 'module.')
    check_keys(model, ckpt_model_dict)
    model.load_state_dict(ckpt_model_dict, strict=False)
    check_keys(optimizer, ckpt['optimizer'])
    optimizer.load_state_dict(ckpt['optimizer'])
    # for pruning
    if 'mask' in ckpt.keys() and 'mask_scores' in ckpt.keys():
        model.mask=ckpt['mask']
        model.mask_scores=ckpt['mask_scores']
    return model, optimizer, epoch
