import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import SGD
from utils.bbox import delta2bbox, corner2center, Corner, Center,center2corner
from utils.anchor import AnchorGenerator
from configs.config import cfg
from dataset.augmentation import Augmentation
from utils.anchor import AnchorTarget
from trackers.base_tracker import BaseTracker


class SiamRPNWithUpdate(BaseTracker):
    def __init__(self, model):
        super(SiamRPNWithUpdate, self).__init__()
        self.model = model
        self.model.eval()
        self.anchor_generator = AnchorGenerator(cfg.ANCHOR.SCALES,
                                                cfg.ANCHOR.RATIOS,
                                                cfg.ANCHOR.STRIDE)
        self.search_aug = Augmentation(
            cfg.DATASET.SEARCH.SHIFT,
            cfg.DATASET.SEARCH.SCALE,
            cfg.DATASET.SEARCH.BLUR,
            cfg.DATASET.SEARCH.FLIP,
            cfg.DATASET.SEARCH.COLOR
        )

        self.score_size = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXAMPLAR_SIZE) // \
                          cfg.ANCHOR.STRIDE + 1 + cfg.TRACK.BASE_SIZE
        self.anchor_target = AnchorTarget(cfg.ANCHOR.SCALES, cfg.ANCHOR.RATIOS, cfg.ANCHOR.STRIDE,
                                          cfg.TRACK.INSTANCE_SIZE // 2, self.score_size)

        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.window = np.tile(window.flatten(), self.anchor_generator.anchor_num)
        self.all_anchor = self.anchor_generator.generate_all_anchors(cfg.TRACK.INSTANCE_SIZE // 2, self.score_size)

    def init(self, img, bbox):
        bbox_pos = bbox[0:2]  # cx,cy
        bbox_size = bbox[2:4]  # w,h
        self.bbox_pos = bbox_pos
        self.bbox_size = bbox_size
        size_z = self._size_z(bbox_size)
        self.channel_average = img.mean((0, 1))
        examplar = self.get_subwindow(img, bbox_pos, cfg.TRACK.EXAMPLAR_SIZE, size_z, self.channel_average)
        examplar = torch.tensor(examplar[np.newaxis, :], dtype=torch.float32).permute(0, 3, 1, 2).cuda()
        self.model.set_examplar_with_update(examplar)
        self.update_optimizer = SGD([{
            'params': self.model.examplar,
            'lr': cfg.TRAIN.BASE_LR
        }], momentum=cfg.TRAIN.MOMENTUM, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
        size_x = self._size_x(bbox_size)
        search = self.get_subwindow(img, bbox_pos, cfg.TRACK.INSTANCE_SIZE, size_x, self.channel_average)
        bbox = self._get_bbox(cfg.TRACK.INSTANCE_SIZE // 2, bbox_size, cfg.TRACK.INSTANCE_SIZE / size_x)
        self.init_samples(search,bbox)
        self.update_filter(cfg.UPDATE.INIT_ITER)
        self.track_frame = 1

    def track(self, img):
        bbox_size = self.bbox_size
        size_z = self._size_z(bbox_size)
        scale_z = cfg.TRACK.EXAMPLAR_SIZE / size_z
        size_x = self._size_x(bbox_size)
        search = self.get_subwindow(img, self.bbox_pos, cfg.TRACK.INSTANCE_SIZE, round(size_x), self.channel_average)
        new_search = torch.from_numpy(search[np.newaxis, :].astype(np.float32)).permute(0, 3, 1, 2).cuda()
        cls, loc = self.model.track(new_search)
        score = self._convert_score(cls)
        loc = loc.reshape(4, self.anchor_generator.anchor_num, loc.size()[2], loc.size()[3])
        pred_bbox = delta2bbox(self.all_anchor, loc)
        pred_bbox = pred_bbox.transpose((1, 2, 3, 0)).reshape((-1, 4))  # x1,y1,x2,y2
        pred_bbox = corner2center(pred_bbox)  # cx,cy,w,h

        def change(r):
            return np.maximum(r, 1 / r)

        rc = change((bbox_size[0] / bbox_size[1]) / (pred_bbox[:, 2] / pred_bbox[:, 3]))
        sc = change(
            self._size_z([self.bbox_size[0] * scale_z, self.bbox_size[1] * scale_z])
            / self._size_z([pred_bbox[:, 2], pred_bbox[:, 3]])
        )
        penalty = np.exp(-(rc * sc - 1) * cfg.TRACK.PENALTY_K)  
        pscore = penalty * score
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
                 self.window * cfg.TRACK.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore)
        best_bbox = pred_bbox[best_idx, :] #cx,cy,w,h



        # online update
        # new_bbox = center2corner(best_bbox.tolist())
        # self.update_samples(search, new_bbox)
        # if self.track_frame % cfg.UPDATE.UPDATE_FREQ == 0:
        #     self.update_filter(cfg.UPDATE.TRACK_ITER)

        # transform the bbox to origin picture scale
        best_bbox[0]-=cfg.TRACK.INSTANCE_SIZE//2 
        best_bbox[1]-=cfg.TRACK.INSTANCE_SIZE//2
        best_bbox = best_bbox / scale_z
        cx = best_bbox[0] + self.bbox_pos[0]  # - size_x // 2
        cy = best_bbox[1] + self.bbox_pos[1]  # - size_x // 2
        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR
        w = self.bbox_size[0] * (1 - lr) + lr * best_bbox[2]
        h = self.bbox_size[1] * (1 - lr) + lr * best_bbox[3]
        pred_bbox = self._clip_bbox(cx, cy, w, h, img.shape[1], img.shape[0])

        # update
        self.bbox_pos = pred_bbox[0:2]
        self.bbox_size = pred_bbox[2:4]
        self.track_frame += 1
        return {
            'bbox': pred_bbox,
            'best_score': score[best_idx]
        }

######## online update ####################
    
    def init_samples(self,init_search,init_bbox):
        self.searchs = []
        self.bboxes = [] #x1,y1,x2,y2
        for i in range(cfg.UPDATE.NUM_SAMPLES):
            new_search, new_bbox = self.search_aug(init_search, init_bbox, cfg.TRACK.INSTANCE_SIZE)
            self.searchs.append(new_search)
            self.bboxes.append(new_bbox)
        self.sample_weight=[1./cfg.UPDATE.NUM_SAMPLES]*cfg.UPDATE.NUM_SAMPLES

    def update_samples(self, search, bbox):
        del self.searchs[0]
        del self.bboxes[0]
        self.searchs.append(search)
        self.bboxes.append(bbox)

    def update_filter(self, n_iter):
        for i in range(n_iter):
            loss=0
            for idx,(search, bbox) in enumerate(zip(self.searchs, self.bboxes)):
                gt_cls, gt_loc, gt_loc_weight = self.anchor_target(bbox)
                search = torch.from_numpy(search[np.newaxis, :, :, :].astype(np.float32)).permute(0, 3, 1, 2).cuda()
                gt_cls = torch.from_numpy(gt_cls).cuda()
                gt_loc = torch.from_numpy(gt_loc).cuda()
                gt_loc_weight = torch.from_numpy(gt_loc_weight).cuda()
                loss += self.sample_weight[idx]*self.model.update_forward(search, gt_cls, gt_loc, gt_loc_weight)
            self.update_optimizer.zero_grad()
            loss.backward()
            self.update_optimizer.step()
