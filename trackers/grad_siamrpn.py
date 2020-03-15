import numpy as np
import torch

from dataset.augmentation import Augmentation
from utils.bbox import delta2bbox, corner2center
from utils.anchor import AnchorGenerator, AnchorTarget
from trackers.base_tracker import BaseTracker
from utils.visual import show_img
from configs.config import cfg


class GradSiamRPN(BaseTracker):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.eval()
        self.anchor_generator = AnchorGenerator(cfg.ANCHOR.SCALES,
                                                cfg.ANCHOR.RATIOS,
                                                cfg.ANCHOR.STRIDE)
        self.score_size = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXAMPLAR_SIZE) // \
                          cfg.ANCHOR.STRIDE + 1 + cfg.TRACK.BASE_SIZE
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.window = np.tile(window.flatten(), self.anchor_generator.anchor_num)
        self.all_anchor = self.anchor_generator.generate_all_anchors(cfg.TRACK.INSTANCE_SIZE // 2, self.score_size)
        self.anchor_target = AnchorTarget(cfg.ANCHOR.SCALES, cfg.ANCHOR.RATIOS, cfg.ANCHOR.STRIDE,
                                          cfg.TRACK.INSTANCE_SIZE // 2, self.score_size)

        self.search_aug = Augmentation(
            cfg.DATASET.SEARCH.SHIFT,
            cfg.DATASET.SEARCH.SCALE,
            cfg.DATASET.SEARCH.BLUR,
            cfg.DATASET.SEARCH.FLIP,
            cfg.DATASET.SEARCH.COLOR
        )

    def init(self, img, bbox):
        bbox_pos = bbox[0:2]  # cx,cy
        bbox_size = bbox[2:4]  # w,h
        size_z = self._size_z(bbox_size)
        self.channel_average = img.mean((0, 1))
        self.examplar = self.get_subwindow(img, bbox_pos, cfg.TRACK.EXAMPLAR_SIZE, size_z, self.channel_average)
        examplar = torch.tensor(self.examplar[np.newaxis, :], dtype=torch.float32).permute(0, 3, 1, 2).cuda()
        size_x = self._size_x(bbox_size)
        search = self.get_subwindow(img, bbox_pos, cfg.TRACK.INSTANCE_SIZE, size_x, self.channel_average)
        # search = torch.from_numpy(search[np.newaxis, :].astype(np.float32)).permute(0, 3, 1, 2).cuda()
        # gt_cls, gt_loc, gt_loc_weight = self.anchor_target(bbox)
        # gt_cls, gt_loc, gt_loc_weight = [torch.from_numpy(x[np.newaxis, :]).cuda() for x in
        #                                  [gt_cls, gt_loc, gt_loc_weight]]

        bbox = self._get_bbox(cfg.TRACK.INSTANCE_SIZE // 2, bbox_size, cfg.TRACK.INSTANCE_SIZE / size_x)
        memory = [self.search_aug(search, bbox, cfg.TRACK.INSTANCE_SIZE) for i in range(cfg.GRAD.MEMORY_SIZE)]
        self.search_mem, self.bbox_mem = zip(*memory)
        self.search_mem, self.bbox_mem = list(self.search_mem), list(self.bbox_mem)
        # self.score_mem = [1] * cfg.META.MEMORY_SIZE

        gt_data = zip(*[self.anchor_target(bbox) for bbox in self.bbox_mem])
        gt_cls, gt_loc, gt_loc_weight = map(lambda x: torch.from_numpy(np.stack(x)).cuda(), gt_data)
        searches = torch.from_numpy(np.stack(self.search_mem).astype(np.float32).transpose((0, 3, 1, 2))).cuda()
        self.model.set_examplar(examplar, searches, gt_cls, gt_loc, gt_loc_weight)
        self.bbox_pos = bbox_pos
        self.bbox_size = bbox_size

    def track(self, img):
        bbox_size = self.bbox_size
        size_z = self._size_z(bbox_size)
        scale_z = cfg.TRACK.EXAMPLAR_SIZE / size_z
        size_x = self._size_x(bbox_size)
        search = self.get_subwindow(img, self.bbox_pos, cfg.TRACK.INSTANCE_SIZE, size_x, self.channel_average)
        # show_img(search)
        new_search = torch.from_numpy(search[np.newaxis, :].astype(np.float32)).permute(0, 3, 1, 2).cuda()
        cls, loc = self.model.track(new_search)
        score = self._convert_score(cls)

        loc = loc.reshape(4, self.anchor_generator.anchor_num, loc.size()[2], loc.size()[3])
        pred_bbox = delta2bbox(self.all_anchor, loc)
        pred_bbox = pred_bbox.transpose((1, 2, 3, 0)).reshape((-1, 4))  # x1,y1,x2,y2
        pred_bbox = corner2center(pred_bbox)  # cx,cy,w,h

        def change(r):
            return np.maximum(r, 1 / r)

        def s_z(w, h):
            w_z = w + 0.5 * (w + h)
            h_z = h + 0.5 * (w + h)
            size_z = np.sqrt(w_z * h_z)
            return size_z

        rc = change((bbox_size[0] / bbox_size[1]) / (pred_bbox[:, 2] / pred_bbox[:, 3]))
        sc = change(
            s_z(self.bbox_size[0] * scale_z, self.bbox_size[1] * scale_z) / s_z(pred_bbox[:, 2], pred_bbox[:, 3]))
        penalty = np.exp(-(rc * sc - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
                 self.window * cfg.TRACK.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore)
        best_bbox = pred_bbox[best_idx, :]
        # show_single_bbox(search,best_bbox.tolist())
        best_bbox[0] -= cfg.TRACK.INSTANCE_SIZE // 2
        best_bbox[1] -= cfg.TRACK.INSTANCE_SIZE // 2
        best_bbox = best_bbox / scale_z
        cx = best_bbox[0] + self.bbox_pos[0]
        cy = best_bbox[1] + self.bbox_pos[1]
        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR
        w = self.bbox_size[0] * (1 - lr) + lr * best_bbox[2]
        h = self.bbox_size[1] * (1 - lr) + lr * best_bbox[3]
        pred_bbox = self._clip_bbox(cx, cy, w, h, img.shape[1], img.shape[0])
        # update
        self.bbox_pos = pred_bbox[0:2]
        self.bbox_size = pred_bbox[2:4]

        return {
            'bbox': pred_bbox,
            'score': score[best_idx]
        }
