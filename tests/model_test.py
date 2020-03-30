import os
import random

import numpy as np
import torch
from models.model_builder import get_model
from utils.anchor import AnchorTarget
from configs.config import cfg


def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__=='__main__':
    cfg.merge_from_file('../configs/mobilenetv2_config.yaml')
    seed_torch(123456)
    data={}
    data['examplar_img']=np.random.randint(0,255,(1,3,127,127)).astype(np.float32)
    data['search_img']=np.random.randint(0,255,(1,3,255,255)).astype(np.float32)
    search_bbox=[30,50,80,100]
    anchor_target = AnchorTarget(cfg.ANCHOR.SCALES, cfg.ANCHOR.RATIOS, cfg.ANCHOR.STRIDE,
                                      cfg.TRAIN.SEARCH_SIZE // 2, cfg.TRAIN.OUTPUT_SIZE)
    data['gt_cls'], data['gt_delta'], data['delta_weight'] = anchor_target(search_bbox, False)
    model=get_model('BaseSiamModel').cuda()

    examplar_img = torch.from_numpy(data['examplar_img']).cuda()
    search_img = torch.from_numpy(data['search_img']).cuda()
    gt_cls = torch.from_numpy(data['gt_cls']).cuda()
    gt_delta = torch.from_numpy(data['gt_delta']).cuda()
    delta_weight = torch.from_numpy(data['delta_weight']).cuda()
    # losses = model.forward(examplar_img, search_img, gt_cls, gt_delta, delta_weight)
    # print(losses['total_loss'].item())
    model.eval()
    model.set_examplar(examplar_img)
    model.track(search_img)


