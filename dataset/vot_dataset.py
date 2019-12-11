import os
import json

import cv2
import numpy as np
from torch.utils.data import Dataset
import torch


class VOTDataset(Dataset):
    def __init__(self, data_dir, anno_file):
        anno_path = os.path.join(data_dir, anno_file)
        with open(anno_path, 'r') as f:
            self.anno_info = json.load(f)
        self.videos = {}
        # TODO: add tqdm
        for video, video_info in self.anno_info.items():
            self.videos[video] = VOTVideo(video,
                                          data_dir,
                                          video_info['init_rect'],
                                          video_info['img_names'],
                                          video_info['gt_rect'],
                                          video_info['camera_motion'],
                                          video_info['illum_change'],
                                          video_info['motion_change'],
                                          video_info['size_change'],
                                          video_info['occlusion'])

    def __getitem__(self, idx):
        if isinstance(idx, str):
            video = self.videos[idx]
        elif isinstance(idx, int):
            video = self.videos[sorted(list(self.videos.keys()))[idx]]
        else:
            raise Exception('video idx type not match!')
        return video


    def __len__(self):
        return len(self.videos)


class VOTVideo(object):
    def __init__(self, name, data_dir, init_rect, img_names, gt_rects,
                 camera_motion, illum_change, motion_change, size_change, occlusion):
        self.name = name
        self.data_dir = data_dir
        self.init_rect = init_rect
        self.img_names = img_names
        self.gt_rects = gt_rects
        self.pred_bboxes = {}
        self.tags = {'all': [1] * len(gt_rects)}
        self.tags['camera_motion'] = camera_motion
        self.tags['illum_change'] = illum_change
        self.tags['motion_change'] = motion_change
        self.tags['size_change'] = size_change
        self.tags['occlusion'] = occlusion

        # empty tag
        all_tag = [v for k, v in self.tags.items() if len(v) > 0]
        self.tags['empty'] = np.all(1 - np.array(all_tag), axis=1).astype(np.int32).tolist()
        ###
        img_path = os.path.join(self.data_dir, self.img_names[0])
        img = cv2.imread(img_path)
        self.width = img.shape[1]
        self.height = img.shape[0]

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

    # TODO: understand it
    def select_tag(self, tag, start=0, end=0):
        if tag == 'empty':
            return self.tags[tag]
        return self.tags[tag][start:end]


def get_axis_aligned_bbox(region):
    """ convert region to (cx, cy, w, h) that represent by axis aligned box
    """
    nv = region.size
    if nv == 8:
        cx = np.mean(region[0::2])
        cy = np.mean(region[1::2])
        x1 = min(region[0::2])
        x2 = max(region[0::2])
        y1 = min(region[1::2])
        y2 = max(region[1::2])
        A1 = np.linalg.norm(region[0:2] - region[2:4]) * \
             np.linalg.norm(region[2:4] - region[4:6])
        A2 = (x2 - x1) * (y2 - y1)
        s = np.sqrt(A1 / A2)
        w = s * (x2 - x1) + 1
        h = s * (y2 - y1) + 1
    else:
        x = region[0]
        y = region[1]
        w = region[2]
        h = region[3]
        cx = x + w / 2
        cy = y + h / 2
    return [cx, cy, w, h]
