from utils.anchor import AnchorTarget
from configs.config import cfg
import random
import os
import numpy as np
import torch

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__=='__main__':
    seed_torch(123456)
    anchor_target = AnchorTarget(cfg.ANCHOR.SCALES, cfg.ANCHOR.RATIOS, cfg.ANCHOR.STRIDE,
                                              cfg.TRAIN.SEARCH_SIZE // 2, cfg.TRAIN.OUTPUT_SIZE)

    search_bbox=[27,47,131,79]


    gt_cls, gt_delta, delta_weight = anchor_target(search_bbox, False)
    print()