import os
import cv2
import numpy as np
from ..utils.bbox import get_axis_aligned_bbox

class Video(object):
    def __init__(self, name, data_dir, init_rect, img_names, gt_rects):
        self.name = name
        self.data_dir = data_dir
        self.init_rect = init_rect
        self.img_names = img_names
        self.gt_rects = gt_rects
        self.pred_bboxes = {}

    def __iter__(self):
        for (img_name, gt_rect) in zip(self.img_names, self.gt_rects):
            img_path = os.path.join(self.data_dir, img_name)
            img = cv2.imread(img_path)
            gt_bbox = get_axis_aligned_bbox(np.array(gt_rect))
            yield img, gt_bbox

    def get_init_img_bbox(self):
        img_path = os.path.join(self.data_dir, self.img_names[0])
        init_img = cv2.imread(img_path)
        init_bbox = get_axis_aligned_bbox(np.array(self.init_rect))
        return init_img, init_bbox

    def load_tracker_result(self, tracker_names):
        if isinstance(tracker_names,str):
            tracker_names=[tracker_names]
        for tracker_name in tracker_names:
            result_file = '{}/{}.txt'.format(tracker_name, self.name)
            with open(result_file, 'r') as f:  # TODO: different from pysot
                pred_bboxes = [list(map(float, line.strip().split(','))) for line in f.readlines()]  # fancy
            self.pred_bboxes[tracker_name] = pred_bboxes
        return pred_bboxes # TODO: a little bad
