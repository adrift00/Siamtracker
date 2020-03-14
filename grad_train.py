import os
import time
import logging
import json
import argparse
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.optim import Adam
from dataset.dataset import GradTrainDataset
from configs.config import cfg
from utils.model_load import load_pretrain
from models import get_model
from utils.log_helper import init_log, add_file_handler, print_speed
from utils.misc import commit, describe
from utils.average_meter import AverageMeter

logger = logging.getLogger("global")
parser = argparse.ArgumentParser()
parser.add_argument("--cfg", default="", type=str, help="which config file to use")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def build_dataloader():
    logger.info("building datalaoder!")
    grad_dataset = GradTrainDataset()
    graph_dataloader = DataLoader(grad_dataset, batch_size=cfg.GRAD.BATCH_SIZE, shuffle=False)
    return graph_dataloader


def build_optimizer(model, current_epoch=0):
    logger.info("build optimizer!")
    for param in model.backbone.parameters():
        param.requires_grad = False
    for m in model.backbone.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
    trainable_param = []
    trainable_param += [
        {"params": model.grad_layer.parameters(), "lr": cfg.GRAD.LR},  # TODO: may be can be optimized
    ]
    trainable_param += [
        {'params': model.rpn.parameters(), 'lr': cfg.GRAD.LR * 0.1}
    ]
    optimizer = Adam(trainable_param, weight_decay=cfg.GRAD.WEIGHT_DECAY)
    return optimizer


def train(dataloader, optimizer, model):
    iter = 0
    begin_time = 0.0
    average_meter = AverageMeter()
    num_per_epoch = len(dataloader.dataset) // (cfg.GRAD.BATCH_SIZE)
    tb_writer = SummaryWriter(cfg.GRAD.LOG_DIR)
    for epoch in range(cfg.GRAD.EPOCHS):
        dataloader.dataset.shuffle()
        begin_time = time.time()
        for data in dataloader:
            examplar_img = data['examplar_img'].cuda()

            train_search_img = data['train_search_img'].cuda()
            train_gt_cls = data['train_gt_cls'].cuda()
            train_gt_delta = data['train_gt_delta'].cuda()
            train_delta_weight = data['train_delta_weight'].cuda()

            test_search_img = data['test_search_img'].cuda()
            test_gt_cls = data['test_gt_cls'].cuda()
            test_gt_delta = data['test_gt_delta'].cuda()
            test_delta_weight = data['test_delta_weight'].cuda()
            data_time = time.time() - begin_time

            losses = model.forward(examplar_img,
                                   train_search_img, train_gt_cls, train_gt_delta, train_delta_weight,
                                   test_search_img, test_gt_cls, test_gt_delta, test_delta_weight)
            cls_loss = losses['cls_loss']
            loc_loss = losses['loc_loss']
            loss = losses['total_loss']
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_time = time.time() - begin_time
            batch_info = {}
            batch_info['data_time'] = data_time
            batch_info['batch_time'] = batch_time
            average_meter.update(**batch_info)
            # add summary writer
            for k, v in losses.items():
                if k.startswith('examplar'):
                    tb_writer.add_histogram(k, v, iter)
                else:
                    tb_writer.add_scalar(k, v, iter)
            if iter % cfg.TRAIN.PRINT_EVERY == 0:
                # logger.info('epoch: {}, iter: {}, init_cls_loss: {}, init_loc_loss: {}, init_loss: {}'
                #             .format(epoch + 1, iter, losses['init_cls_loss'].item(), losses['init_loc_loss'].item(),
                #                     losses['init_total_loss'].item()))
                logger.info('epoch: {}, iter: {}, cls_loss: {}, loc_loss: {}, loss: {}'
                            .format(epoch + 1, iter, cls_loss.item(), loc_loss.item(), loss.item()))
                print_speed(iter + 1,
                            average_meter.batch_time.avg,
                            cfg.GRAD.EPOCHS * num_per_epoch)
            begin_time = time.time()
            iter += 1
        # save train_state
        if not os.path.exists(cfg.GRAD.SNAPSHOT_DIR):
            os.makedirs(cfg.GRAD.SNAPSHOT_DIR)
        # put the update to the rpn state
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }
        save_path = "{}/checkpoint_e{}.pth".format(cfg.GRAD.SNAPSHOT_DIR, epoch)
        logger.info("save state to {}".format(save_path))
        torch.save(state, save_path)


def main():
    cfg.merge_from_file(args.cfg)
    if not os.path.exists(cfg.GRAD.LOG_DIR):
        os.makedirs(cfg.GRAD.LOG_DIR)
    init_log("global", logging.INFO)
    if cfg.GRAD.LOG_DIR:
        add_file_handler(
            "global", os.path.join(cfg.GRAD.LOG_DIR, "logs.txt"), logging.INFO
        )
    logger.info("Version Information: \n{}\n".format(commit()))
    logger.info("config \n{}".format(json.dumps(cfg, indent=4)))
    model = get_model('GradSiamModel').cuda()
    model = load_pretrain(model, cfg.GRAD.PRETRAIN_PATH)
    # parametes want to optim
    optimizer = build_optimizer(model)
    dataloader = build_dataloader()
    model.freeze_model()
    train(dataloader, optimizer, model)


if __name__ == "__main__":
    main()
