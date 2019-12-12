import os
import time
import logging
import json
import argparse
import torch
import numpy as np
from collections import OrderedDict
from torch.utils.data import DataLoader
from torch.optim import Adam
from dataset.dataset import GraphDataset
from configs.config import cfg
from utils.model_load import load_pretrain
from models.model_builder import get_model
from utils.log_helper import init_log, add_file_handler, print_speed
from utils.misc import commit, describe
from utils.average_meter import AverageMeter

logger = logging.getLogger("global")
parser = argparse.ArgumentParser()
parser.add_argument("--cfg", default="", type=str, help="which config file to use")
args = parser.parse_args()


def build_dataloader():
    logger.info("building datalaoder!")
    graph_dataset = GraphDataset()
    graph_dataloader = DataLoader(
        graph_dataset, batch_size=cfg.GRAPH.BATCH_SIZE, shuffle=False
    )
    return graph_dataloader


def build_optimizer(model):
    logger.info("build optimizer!")
    parameters = [
        {"params": model.gcn.parameters(), "lr": cfg.GRAPH.LR},  # TODO: may be can be optimized
        {"params": model.rpn.parameters(), "lr": cfg.GRAPH.LR*0.5},
    ]
    optimizer = Adam(parameters, weight_decay=cfg.GRAPH.WEIGHT_DECAY)
    return optimizer


def train(dataloader, optimizer, model):
    iter = 0
    begin_time = 0.0
    average_meter = AverageMeter()
    num_per_epoch = len(dataloader.dataset) // (cfg.GRAPH.BATCH_SIZE)
    for epoch in range(cfg.GRAPH.EPOCHS):
        dataloader.dataset.shuffle()
        begin_time = time.time()
        for data in dataloader:
            examplar_imgs = data['examplars'].cuda()
            search_img = data['search'].cuda()
            gt_cls = data['gt_cls'].cuda()
            gt_delta = data['gt_delta'].cuda()
            delta_weight = data['gt_delta_weight'].cuda()
            data_time = time.time()-begin_time
            examplar_imgs = examplar_imgs[0]

            losses = model.forward(examplar_imgs, search_img, gt_cls, gt_delta, delta_weight)
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
            if iter % cfg.TRAIN.PRINT_EVERY == 0:
                logger.info('epoch: {}, iter: {}, cls_loss: {}, loc_loss: {}, loss: {}'
                            .format(epoch + 1, iter, cls_loss.item(), loc_loss.item(), loss.item()))
                print_speed(iter + 1,
                            average_meter.batch_time.avg,
                            cfg.GRAPH.EPOCHS * num_per_epoch)
            data_begin_time = time.time()
            iter += 1
        # save train_state
        if not os.path.exists(cfg.GRAPH.SNAPSHOT_DIR):
            os.makedirs(cfg.GRAPH.SNAPSHOT_DIR)
        # put the update to the rpn state
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }
        save_path = "{}/checkpoint_e{}.pth".format(cfg.GRAPH.SNAPSHOT_DIR, epoch)
        logger.info("save state to {}".format(save_path))
        torch.save(state, save_path)


def main():
    cfg.merge_from_file(args.cfg)
    if not os.path.exists(cfg.GRAPH.LOG_DIR):
        os.makedirs(cfg.GRAPH.LOG_DIR)
    init_log("global", logging.INFO)
    if cfg.GRAPH.LOG_DIR:
        add_file_handler(
            "global", os.path.join(cfg.GRAPH.LOG_DIR, "logs.txt"), logging.INFO
        )
    logger.info("Version Information: \n{}\n".format(commit()))
    logger.info("config \n{}".format(json.dumps(cfg, indent=4)))
    model = get_model('GraphSiamModel').cuda()
    model = load_pretrain(model, cfg.GRAPH.PRETRAIN_PATH)
    # parametes want to optim
    optimizer = build_optimizer(model)
    dataloader = build_dataloader()
    train(dataloader, optimizer, model)


if __name__ == "__main__":
    main()
