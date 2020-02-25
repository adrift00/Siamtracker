import argparse

import torch.nn as nn
from torch import optim

from configs.config import cfg
from models.model_builder import get_model
from utils.lr_scheduler import build_lr_scheduler


parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default='', type=str, help='which config file to use')
args = parser.parse_args()


def build_optimizer_lr(model, current_epoch=0):
    for param in model.backbone.parameters():
        param.requires_grad = False
    for m in model.backbone.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
    if current_epoch >= cfg.BACKBONE.TRAIN_EPOCH:
        for layer in cfg.BACKBONE.TRAIN_LAYERS:
            for param in getattr(model.backbone, layer).parameters():
                param.requires_grad = True
            for m in getattr(model.backbone, layer).modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.train()
    trainable_param = []
    trainable_param += [{
        'params': filter(lambda x: x.requires_grad, model.backbone.parameters()),
        'lr': cfg.BACKBONE.LAYERS_LR * cfg.TRAIN.BASE_LR
    }]

    if cfg.ADJUST.USE:
        trainable_param+=[{
            'params': model.neck.parameters(),
            'lr':cfg.TRAIN.BASE_LR
        }]
    trainable_param += [{
        'params': model.rpn.parameters(),
        'lr': cfg.TRAIN.BASE_LR
    }]
    optimizer = optim.SGD(trainable_param, momentum=cfg.TRAIN.MOMENTUM, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    lr_scheduler = build_lr_scheduler(optimizer, epochs=cfg.TRAIN.EPOCHS)
    lr_scheduler.step(cfg.TRAIN.START_EPOCH)
    return optimizer, lr_scheduler

if __name__=='__main__':
    cfg.merge_from_file(args.cfg)
    model = get_model('BaseSiamModel').cuda().train()
    optimizer, lr_scheduler = build_optimizer_lr(model,cfg.TRAIN.START_EPOCH)
    lr_scheduler.step(cfg.TRAIN.START_EPOCH)
