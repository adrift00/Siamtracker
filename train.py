import json
import math
import os
import logging
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from configs.config import cfg
from dataset.dataset import TrainDataset
from utils.log_helper import init_log, add_file_handler, print_speed
from utils.lr_scheduler import build_lr_scheduler
from models import get_model
from utils.distributed import get_world_size, dist_init, DistModule, get_rank, reduce_gradients, average_reduce
from utils.misc import commit, describe
from utils.model_load import load_pretrain, restore_from
from utils.average_meter import AverageMeter

logger = logging.getLogger('global')

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default='', type=str, help='which config file to use')
parser.add_argument('--local_rank', type=int, default=0,
                    help='compulsory for pytorch launcer')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def log_grads(model, tb_writer, tb_index):
    def weights_grads(model):
        grad = {}
        weights = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad[name] = param.grad
                weights[name] = param.data
        return grad, weights

    grad, weights = weights_grads(model)
    feature_norm, rpn_norm = 0, 0
    for k, g in grad.items():
        _norm = g.data.norm(2)
        weight = weights[k]
        w_norm = weight.norm(2)
        if 'feature' in k:
            feature_norm += _norm ** 2
        else:
            rpn_norm += _norm ** 2

        tb_writer.add_scalar('grad_all/' + k.replace('.', '/'),
                             _norm, tb_index)
        tb_writer.add_scalar('weight_all/' + k.replace('.', '/'),
                             w_norm, tb_index)
        tb_writer.add_scalar('w-g/' + k.replace('.', '/'),
                             w_norm / (1e-20 + _norm), tb_index)
    tot_norm = feature_norm + rpn_norm
    tot_norm = tot_norm ** 0.5
    feature_norm = feature_norm ** 0.5
    rpn_norm = rpn_norm ** 0.5

    tb_writer.add_scalar('grad/tot', tot_norm, tb_index)
    tb_writer.add_scalar('grad/feature', feature_norm, tb_index)
    tb_writer.add_scalar('grad/head', rpn_norm, tb_index)


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
        trainable_param += [{
            'params': model.neck.parameters(),
            'lr': cfg.TRAIN.BASE_LR
        }]
    trainable_param += [{
        'params': model.rpn.parameters(),
        'lr': cfg.TRAIN.BASE_LR
    }]
    optimizer = optim.SGD(trainable_param, momentum=cfg.TRAIN.MOMENTUM, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    lr_scheduler = build_lr_scheduler(optimizer, epochs=cfg.TRAIN.EPOCHS)
    lr_scheduler.step(cfg.TRAIN.START_EPOCH)
    return optimizer, lr_scheduler


def build_data_loader():
    logger.info("build train dataset")
    # train_dataset
    train_dataset = TrainDataset()
    logger.info("build dataset done")

    train_sampler = None
    if get_world_size() > 1:
        train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=cfg.TRAIN.BATCH_SIZE,
                                  num_workers=cfg.TRAIN.NUM_WORKERS,
                                  pin_memory=True,
                                  sampler=train_sampler)
    return train_dataloader


def train(train_dataloader, model, optimizer, lr_scheduler):
    def is_valid_number(x):
        return not (math.isnan(x) or math.isinf(x) or x > 1e4)

    logger.info("model\n{}".format(describe(model.module)))
    tb_writer = SummaryWriter(cfg.TRAIN.LOG_DIR)
    average_meter = AverageMeter()
    start_epoch = cfg.TRAIN.START_EPOCH
    world_size = get_world_size()
    num_per_epoch = len(train_dataloader.dataset) // (cfg.TRAIN.BATCH_SIZE * world_size)
    iter = 0
    if not os.path.exists(cfg.TRAIN.SNAPSHOT_DIR) and get_rank() == 0:
        os.makedirs(cfg.TRAIN.SNAPSHOT_DIR)
    for epoch in range(cfg.TRAIN.START_EPOCH, cfg.TRAIN.EPOCHS):
        if cfg.BACKBONE.TRAIN_EPOCH == epoch:
            logger.info('begin to train backbone!')
            optimizer, lr_scheduler = build_optimizer_lr(model.module, epoch)
            logger.info("model\n{}".format(describe(model.module)))
        train_dataloader.dataset.shuffle()
        lr_scheduler.step(epoch)
        # log for lr
        if get_rank() == 0:
            for idx, pg in enumerate(optimizer.param_groups):
                tb_writer.add_scalar('lr/group{}'.format(idx + 1), pg['lr'], iter)
        cur_lr = lr_scheduler.get_cur_lr()
        for data in train_dataloader:
            begin = time.time()
            examplar_img = data['examplar_img'].cuda()
            search_img = data['search_img'].cuda()
            gt_cls = data['gt_cls'].cuda()
            gt_delta = data['gt_delta'].cuda()
            delta_weight = data['delta_weight'].cuda()
            data_time = time.time() - begin
            losses = model.forward(examplar_img, search_img, gt_cls, gt_delta, delta_weight)
            cls_loss = losses['cls_loss']
            loc_loss = losses['loc_loss']
            loss = losses['total_loss']

            if is_valid_number(loss.item()):
                optimizer.zero_grad()
                loss.backward()
                reduce_gradients(model)
                if get_rank() == 0 and cfg.TRAIN.LOG_GRAD:
                    log_grads(model.module, tb_writer, iter)
                clip_grad_norm_(model.parameters(), cfg.TRAIN.GRAD_CLIP)
                optimizer.step()

            batch_time = time.time() - begin
            batch_info = {}
            batch_info['data_time'] = average_reduce(data_time)
            batch_info['batch_time'] = average_reduce(batch_time)
            for k, v in losses.items():
                batch_info[k] = average_reduce(v)
            average_meter.update(**batch_info)
            if get_rank() == 0:
                for k, v in batch_info.items():
                    tb_writer.add_scalar(k, v, iter)
                if iter % cfg.TRAIN.PRINT_EVERY == 0:
                    logger.info('epoch: {}, iter: {}, cur_lr:{}, cls_loss: {}, loc_loss: {}, loss: {}'
                                .format(epoch + 1, iter, cur_lr, cls_loss.item(), loc_loss.item(), loss.item()))
                    print_speed(iter + 1 + start_epoch * num_per_epoch,
                                average_meter.batch_time.avg,
                                cfg.TRAIN.EPOCHS * num_per_epoch)
            iter += 1
        # save model
        if get_rank() == 0:
            state = {
                'model': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1
            }
            logger.info('save snapshot to {}/checkpoint_e{}.pth'.format(cfg.TRAIN.SNAPSHOT_DIR, epoch + 1))
            torch.save(state, '{}/checkpoint_e{}.pth'.format(cfg.TRAIN.SNAPSHOT_DIR, epoch + 1))


def main():
    rank, world_size = dist_init()
    cfg.merge_from_file(args.cfg)
    if rank == 0:
        if not os.path.exists(cfg.TRAIN.LOG_DIR):
            os.makedirs(cfg.TRAIN.LOG_DIR)
        init_log('global', logging.INFO)
        if cfg.TRAIN.LOG_DIR:
            add_file_handler('global',
                             os.path.join(cfg.TRAIN.LOG_DIR, 'logs.txt'),
                             logging.INFO)
        logger.info("Version Information: \n{}\n".format(commit()))
        logger.info("config \n{}".format(json.dumps(cfg, indent=4)))

    logger.info('dist init done!')
    train_dataloader = build_data_loader()
    model = get_model('BaseSiamModel').cuda().train()
    dist_model = DistModule(model)
    optimizer, lr_scheduler = build_optimizer_lr(dist_model.module, cfg.TRAIN.START_EPOCH)
    if cfg.TRAIN.BACKBONE_PRETRAIN:
        logger.info('load backbone from {}.'.format(cfg.TRAIN.BACKBONE_PATH))
        model.backbone = load_pretrain(model.backbone, cfg.TRAIN.BACKBONE_PATH)
        logger.info('load backbone done!')
    if cfg.TRAIN.RESUME:
        logger.info('resume from {}'.format(cfg.TRAIN.RESUME_PATH))
        model, optimizer, cfg.TRAIN.START_EPOCH = restore_from(model, optimizer, cfg.TRAIN.RESUME_PATH)
        logger.info('resume done!')
    elif cfg.TRAIN.PRETRAIN:
        logger.info('load pretrain from {}.'.format(cfg.TRAIN.PRETRAIN_PATH))
        model = load_pretrain(model, cfg.TRAIN.PRETRAIN_PATH)
        logger.info('load pretrain done')
    dist_model = DistModule(model)
    train(train_dataloader, dist_model, optimizer, lr_scheduler)


if __name__ == '__main__':
    seed_torch(123456)

