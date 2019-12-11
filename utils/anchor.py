import math
import numpy as np
from utils.bbox import bbox2delta, calc_iou, corner2center
from configs.config import cfg


class AnchorGenerator(object):
    def __init__(self, scales, ratios, stride):
        self.scales = scales
        self.ratios = ratios
        self.stride = stride
        self.basesize = self.stride
        self.anchor_num = len(scales) * len(ratios)

    def _generate_base_anchor(self):
        base_anchor = np.zeros((self.anchor_num, 4), dtype=np.float32)
        size = self.stride * self.stride
        count = 0
        for r in self.ratios:
            ws = int(math.sqrt(size * 1. / r))
            hs = int(ws * r)
            for s in self.scales:
                w = ws * s
                h = hs * s
                base_anchor[count][:] = [-w * 0.5, -h * 0.5, w * 0.5, h * 0.5][:]
                count += 1
        return base_anchor

    def generate_all_anchors(self, img_c, out_size):
        begin = img_c - out_size // 2 * self.stride
        base_anchors = self._generate_base_anchor()
        base_anchors = base_anchors + begin
        shift_x = np.arange(0, out_size) * self.stride
        shift_y = np.arange(0, out_size) * self.stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        x1 = base_anchors[:, 0]
        y1 = base_anchors[:, 1]
        x2 = base_anchors[:, 2]
        y2 = base_anchors[:, 3]
        x1, y1, x2, y2 = map(lambda x: x.reshape(self.anchor_num, 1, 1), [x1, y1, x2, y2])
        x1 = shift_x + x1
        y1 = shift_y + y1
        x2 = shift_x + x2
        y2 = shift_y + y2
        all_anchors = np.stack([x1, y1, x2, y2]).astype(np.float32)
        return all_anchors


class AnchorTarget(object):
    def __init__(self, scales, ratios, stride, img_c, out_size):
        self.out_size = out_size
        self.anchor_generator = AnchorGenerator(scales, ratios, stride)
        self.all_anchors = self.anchor_generator.generate_all_anchors(img_c, out_size)

    def __call__(self, gt_bbox, neg=False):  # corner x1,y1,x2,y2
        anchor_num = self.anchor_generator.anchor_num
        gt_cls = -1 * np.ones((anchor_num, self.out_size, self.out_size), dtype=np.int64)
        gt_delta = np.zeros((4, anchor_num, self.out_size, self.out_size), dtype=np.float32)
        delta_weight = np.zeros((anchor_num, self.out_size, self.out_size), dtype=np.float32)
        gt_cx, gt_cy, gt_w, gt_h = corner2center(gt_bbox)
        if neg:
            cx = self.out_size // 2
            cy = self.out_size // 2
            cx += int(np.ceil((gt_cx - cfg.TRAIN.SEARCH_SIZE // 2) / cfg.ANCHOR.STRIDE + 0.5))
            cy += int(np.ceil((gt_cy - cfg.TRAIN.SEARCH_SIZE // 2) / cfg.ANCHOR.STRIDE + 0.5))
            l = max(0, cx - 3)
            r = min(self.out_size, cx + 4)
            u = max(0, cy - 3)
            d = min(self.out_size, cy + 4)
            gt_cls[:, u:d, l:r] = 0
            neg_idx = np.where(gt_cls == 0)
            neg_idx = np.vstack(neg_idx).transpose()
            if (len(neg_idx) > cfg.TRAIN.NEG_NUM):
                keep_num = cfg.TRAIN.NEG_NUM
                np.random.shuffle(neg_idx)
                neg_idx = neg_idx[:keep_num, :]
            gt_cls[:] = -1
            gt_cls[neg_idx[:, 0], neg_idx[:, 1], neg_idx[:, 2]] = 0
            return gt_cls, gt_delta, delta_weight

        # NOTE: the shape of all_anchors and gt_bbox are different, need broadcast.
        iou = calc_iou(self.all_anchors, gt_bbox)

        pos_idx = np.where(iou > cfg.TRAIN.THRESH_HIGH)
        neg_idx = np.where(iou < cfg.TRAIN.THRESH_LOW)
        pos_idx = np.vstack(pos_idx).transpose()
        neg_idx = np.vstack(neg_idx).transpose()
        pos_num = len(pos_idx)
        if (pos_num > cfg.TRAIN.POS_NUM):
            keep_num = cfg.TRAIN.POS_NUM
            np.random.shuffle(pos_idx)
            pos_idx = pos_idx[:keep_num, :]
        gt_cls[pos_idx[:, 0], pos_idx[:, 1], pos_idx[:, 2]] = 1
        delta_weight[pos_idx[:, 0], pos_idx[:, 1], pos_idx[:, 2]] = 1 / (pos_num + 1e-6)
        neg_num = cfg.TRAIN.TOTAL_NUM - cfg.TRAIN.POS_NUM
        if (len(neg_idx) > neg_num):
            keep_num = neg_num
            np.random.shuffle(neg_idx)
            neg_idx = neg_idx[:keep_num, :]
        gt_cls[neg_idx[:, 0], neg_idx[:, 1], neg_idx[:, 2]] = 0
        gt_delta = bbox2delta(self.all_anchors, gt_bbox)
        return gt_cls, gt_delta, delta_weight
