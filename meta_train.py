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
from dataset.dataset import MetaDataset
from configs.config import cfg
from utils.model_load import load_pretrain
from models.model_builder import MetaSiamModel
from utils.log_helper import init_log, add_file_handler
from utils.misc import commit, describe

logger = logging.getLogger("global")
parser = argparse.ArgumentParser()
parser.add_argument("--cfg", default="", type=str, help="which config file to use")
args = parser.parse_args()


def build_dataloader():
    logger.info("building dataloader!")
    meta_dataset = MetaDataset()
    meta_dataloader = DataLoader(
        meta_dataset, batch_size=cfg.META.BATCH_SIZE, shuffle=False
    )
    return meta_dataloader


def build_optimizer(model):
    logger.info("build optimizer!")
    parameters = [
        {"params": model.init_weight.values(), "lr": cfg.META.LR},  # TODO: may be can be optimized
        {"params": model.alpha.values(), "lr": cfg.META.LR},
    ]
    optimizer = Adam(parameters,weight_decay=cfg.META.WEIGHT_DECAY)
    return optimizer


def meta_train(datalaoder, optimizer, model):
    data_begin_time = 0.0
    optim_begin_time = 0.0
    data_time = 0.0
    optim_begin_time = 0.0
    for epoch in range(cfg.META.TRAIN_EPOCH):
        data_begin_time = time.time()
        for iter, data in enumerate(datalaoder):
            data_time = time.time()-data_begin_time
            optim_begin_time = time.time()
            init_grads = []
            alpha_grads = []
            losses = []
            batch_size = min(cfg.META.BATCH_SIZE, data["train_examplar_imgs"].size(0))
            for i in range(batch_size):
                train_examplar_imgs = data["train_examplar_imgs"][i].cuda()
                test_examplar_imgs = data["test_examplar_imgs"][i].cuda()
                train_search_imgs = data["train_search_imgs"][i].cuda()
                train_cls = data["train_cls"][i].cuda()
                train_delta = data["train_delta"][i].cuda()
                train_delta_weight = data["train_delta_weight"][i].cuda()
                test_search_imgs = data["test_search_imgs"][i].cuda()
                test_cls = data["test_cls"][i].cuda()
                test_delta = data["test_delta"][i].cuda()
                test_delta_weight = data["test_delta_weight"][i].cuda()
                data_time = time.time()-data_begin_time
                # train in the init frame
                new_weight = model.meta_train(
                    train_examplar_imgs, train_search_imgs, train_cls, train_delta, train_delta_weight
                )
                # eval on the end frame
                init_grad, alpha_grad, loss = model.meta_eval(
                    new_weight, test_examplar_imgs, test_search_imgs, test_cls, test_delta, test_delta_weight
                )
                init_grads.append(init_grad)
                alpha_grads.append(alpha_grad)
                losses.append(loss.item())
            optimizer.zero_grad()
            # compute the sum of grads in all tasks
            init_grad_sum = OrderedDict(
                (k, sum(g[k] for g in init_grads)) for k in init_grads[0].keys()
            )
            alpha_grad_sum = OrderedDict(
                (k, sum(g[k] for g in alpha_grads)) for k in alpha_grads[0].keys()
            )
            # update the grad of weight
            for k, init_weight in model.init_weight.items():
                init_weight.grad = init_grad_sum[k]
            for k, alpha in model.alpha.items():
                alpha.grad = alpha_grad_sum[k]
            optimizer.step()
            optim_time = time.time()-optim_begin_time
            logger.info('data_time: {},optim_time:{}'.format(data_time, optim_time))
            logger.info(
                "epoch: {}, iter: {}, loss: {}".format(epoch, iter, np.mean(losses))
            )
            data_begin_time = time.time()
        # save train_state
        if not os.path.exists(cfg.META.SNAPSHOT_DIR):
            os.makedirs(cfg.META.SNAPSHOT_DIR)
        # put the update to the rpn state
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            # for meta model
            "init_weight": model.init_weight,
            "alpha": model.alpha,
            "bn_weight": model.bn_weight
        }
        save_path = "{}/checkpoint_e{}.pth".format(cfg.META.SNAPSHOT_DIR, epoch)
        logger.info("save state to {}".format(save_path))
        torch.save(state, save_path)


def main():
    cfg.merge_from_file(args.cfg)
    if not os.path.exists(cfg.META.LOG_DIR):
        os.makedirs(cfg.META.LOG_DIR)
    init_log("global", logging.INFO)
    if cfg.META.LOG_DIR:
        add_file_handler(
            "global", os.path.join(cfg.META.LOG_DIR, "logs.txt"), logging.INFO
        )
    logger.info("Version Information: \n{}\n".format(commit()))
    logger.info("config \n{}".format(json.dumps(cfg, indent=4)))
    model = MetaSiamModel().cuda()
    model = load_pretrain(model, cfg.META.PRETRAIN_PATH)
    # init meta train
    model.meta_train_init()
    # parametes want to optim
    optimizer = build_optimizer(model)
    dataloader = build_dataloader()
    meta_train(dataloader, optimizer, model)


if __name__ == "__main__":
    main()
