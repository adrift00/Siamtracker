import cv2
import numpy as np
import torch
import torch.nn.functional as F
from utils.bbox import delta2bbox, corner2center, Corner, Center
from utils.visual import show_single_bbox
from utils.anchor import AnchorGenerator
from trackers.base_tracker import BaseTracker
from configs.config import cfg
from dataset.augmentation import Augmentation


class GraphSiamRPN(BaseTracker):
    def __init__(self, model):
        super(GraphSiamRPN, self).__init__()
        self.model = model
        self.model.eval()
        self.anchor_generator = AnchorGenerator(cfg.ANCHOR.SCALES,
                                                cfg.ANCHOR.RATIOS,
                                                cfg.ANCHOR.STRIDE)
        self.score_size = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXAMPLAR_SIZE) // \
            cfg.ANCHOR.STRIDE + 1 + cfg.TRACK.BASE_SIZE

        self.examplar_aug = Augmentation(
            cfg.DATASET.EXAMPLAR.SHIFT,
            cfg.DATASET.EXAMPLAR.SCALE,
            cfg.DATASET.EXAMPLAR.BLUR,
            cfg.DATASET.EXAMPLAR.FLIP,
            cfg.DATASET.EXAMPLAR.COLOR
        )
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.window = np.tile(window.flatten(), self.anchor_generator.anchor_num)
        self.all_anchor = self.anchor_generator.generate_all_anchors(cfg.TRACK.INSTANCE_SIZE // 2, self.score_size)

    def init(self, img, bbox):
        bbox_pos = bbox[0:2]  # cx,cy
        bbox_size = bbox[2:4]  # w,h
        # gen aug examplars
        size_z = self._size_z(bbox_size)
        self.channel_average = img.mean((0, 1))
        examplar = self.get_subwindow(img, bbox_pos, cfg.TRACK.EXAMPLAR_SIZE, size_z, self.channel_average)
        bbox = self._get_bbox(cfg.TRACK.EXAMPLAR_SIZE // 2, bbox_size, cfg.TRACK.EXAMPLAR_SIZE / size_z)
        self.examplars = [self.examplar_aug(examplar, bbox, cfg.TRACK.INSTANCE_SIZE)[0]
                          for i in range(cfg.META.MEMORY_SIZE)]
        examplars = torch.from_numpy(np.stack(self.examplars).astype(np.float32).transpose((0, 3, 1, 2))).cuda()
        self.model.set_examplar(examplars)
        self.bbox_pos = bbox_pos
        self.bbox_size = bbox_size
        self.track_frame = 0

    def track(self, img):
        bbox_size = self.bbox_size
        size_z = self._size_z(bbox_size)
        scale_z = cfg.TRACK.EXAMPLAR_SIZE / size_z
        size_x = self._size_x(bbox_size)
        search = self.get_subwindow(img, self.bbox_pos, cfg.TRACK.INSTANCE_SIZE, size_x, self.channel_average)
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
        sc = change(s_z(self.bbox_size[0] * scale_z,
                        self.bbox_size[1] * scale_z) / s_z(pred_bbox[:, 2], pred_bbox[:, 3]))
        penalty = np.exp(-(rc * sc - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
            self.window * cfg.TRACK.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore)
        best_bbox = pred_bbox[best_idx, :]
        best_score = pscore[best_idx]
        # update track state
        best_bbox[0] -= cfg.TRACK.INSTANCE_SIZE//2
        best_bbox[1] -= cfg.TRACK.INSTANCE_SIZE//2
        best_bbox = best_bbox / scale_z
        cx = best_bbox[0] + self.bbox_pos[0]
        cy = best_bbox[1] + self.bbox_pos[1]
        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR
        w = self.bbox_size[0] * (1 - lr) + lr * best_bbox[2]
        h = self.bbox_size[1] * (1 - lr) + lr * best_bbox[3]
        pred_bbox = self._clip_bbox(cx, cy, w, h, img.shape[1], img.shape[0])
        self.bbox_pos = pred_bbox[0:2]
        self.bbox_size = pred_bbox[2:4]
        self.track_frame += 1
        # update examplars
        if self.track_frame % cfg.GRAPH.UPDATE_FREQ == 0:
            size_z = self._size_z(self.bbox_size)
            self.channel_average = img.mean((0, 1))
            examplar = self.get_subwindow(img, bbox_pos, cfg.TRACK.EXAMPLAR_SIZE, size_z, self.channel_average)
            del self.examplars[0]
            self.examplars.append(examplar)
            examplars = torch.from_numpy(np.stack(self.examplars).astype(np.float32).transpose((0, 3, 1, 2))).cuda()
            self.model.set_examplar(examplars)

        return {
            'bbox': pred_bbox,
            'best_score': score[best_idx]
        }
