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
        self.imgs = None

    def __iter__(self):
        if self.imgs is not None:
            for (img, gt_rect) in zip(self.imgs, self.gt_rects):
                gt_bbox = get_axis_aligned_bbox(np.array(gt_rect))
                yield img, gt_bbox
        else:
            for (img_name, gt_rect) in zip(self.img_names, self.gt_rects):
                img_path = os.path.join(self.data_dir, img_name)
                img = cv2.imread(img_path)
                # convert bgr to rgb in order to match pretrain model
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                gt_bbox = get_axis_aligned_bbox(np.array(gt_rect))
                yield img, gt_bbox

    def read_imgs(self):
        # self.imgs=[cv2.imread(os.path.join(self.data_dir,img_name)) for img_name in self.img_names ]
        # convert bgr to rgb in order to match pretrain model
        self.imgs = [cv2.cvtColor(cv2.imread(os.path.join(self.data_dir, img_name)), cv2.COLOR_BGR2RGB)
                     for img_name in self.img_names]

    def free_imgs(self):
        self.imgs = None

    # def get_init_img_bbox(self):
    #     img_path = os.path.join(self.data_dir, self.img_names[0])
    #     init_img = cv2.imread(img_path)
    #     init_bbox = get_axis_aligned_bbox(np.array(self.init_rect))
    #     return init_img, init_bbox

    def load_tracker_result(self, tracker_names):
        if isinstance(tracker_names, str):
            tracker_names = [tracker_names]
        for tracker_name in tracker_names:
            result_file = '{}/{}.txt'.format(tracker_name, self.name)
            with open(result_file, 'r') as f:  # TODO: different from pysot
                pred_bboxes = [list(map(float, line.strip().split(',')))
                               for line in f.readlines()]  # fancy
            self.pred_bboxes[tracker_name] = pred_bboxes
        return pred_bboxes  # TODO: a little bad
