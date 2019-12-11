import os
import json
import logging
import cv2
import numpy as np
from torch.utils.data import Dataset
from configs.config import cfg
from utils.bbox import center2corner, Center
from utils.anchor import AnchorTarget
from dataset.augmentation import Augmentation

logger = logging.getLogger('global')


class SubDataset(object):
    def __init__(self, name, data_dir, anno_file, frame_range, start_idx, num_use):
        self.name = name
        self.data_dir = data_dir
        self.anno_file = anno_file
        self.frame_range = frame_range
        self.start_idx = start_idx
        self.annos = json.load(open(anno_file, 'r'))
        self.annos = self._filter_zero()
        self.videos = list(self.annos.keys())
        self.num = len(self.annos.keys())
        self.num_use = self.num if num_use == -1 else num_use
        self.filename_format = '{}.{}.{}.jpg'
        for video, tracks in self.annos.items():
            for trackid in tracks.keys():
                frames = self.annos[video][trackid]
                frames = list(
                    map(int, filter(lambda x: x.isdigit(), frames.keys())))
                frames.sort()
                self.annos[video][trackid]['frames'] = frames

    def _filter_zero(self):
        new_annos = {}
        for video, tracks in self.annos.items():
            new_tracks = {}
            for trackid, frames in tracks.items():
                new_frames = {}
                for frame, bbox in frames.items():
                    if not isinstance(bbox, dict):
                        if len(bbox) == 4:
                            x1, y1, x2, y2 = bbox
                            w, h = x2 - x1, y2 - y1
                        else:
                            w, h = bbox
                        if w <= 0 or h <= 0:
                            continue
                    new_frames[frame] = bbox
                if len(new_frames) > 0:
                    new_tracks[trackid] = new_frames
            if len(new_tracks) > 0:
                new_annos[video] = new_tracks
        return new_annos

    def get_postive_pair(self, idx):
        video = self.videos[idx]
        trackid = np.random.choice(list(self.annos[video].keys()))
        frames = self.annos[video][trackid]['frames']
        examplar_idx = np.random.randint(0, len(frames))
        left = max(0, examplar_idx - self.frame_range)
        right = min(len(frames) - 1, examplar_idx + self.frame_range) + 1
        search_range = frames[left:right]
        examplar_frame = frames[examplar_idx]
        search_frame = np.random.choice(search_range)
        examplar_frame = '{:06d}'.format(examplar_frame)
        search_frame = '{:06d}'.format(search_frame)
        examplar_path = os.path.join(self.data_dir, video, self.filename_format.format(examplar_frame, trackid, 'x'))
        search_path = os.path.join(self.data_dir, video, self.filename_format.format(search_frame, trackid, 'x'))
        examplar_anno = self.annos[video][trackid][examplar_frame]
        search_anno = self.annos[video][trackid][search_frame]
        return (examplar_path, examplar_anno), (search_path, search_anno)

    def get_random_target(self):
        idx = np.random.randint(0, self.num)
        video = self.videos[idx]

        trackid = np.random.choice(list(self.annos[video].keys()))
        frames = self.annos[video][trackid]['frames']
        target_frame = np.random.choice(frames)
        target_frame = '{:06d}'.format(target_frame)
        target_path = os.path.join(self.data_dir, video, self.filename_format.format(target_frame, trackid, 'x'))
        target_anno = self.annos[video][trackid][target_frame]
        return target_path, target_anno

    def log(self):
        logger.info("{} start-index {} select [{}/{}]".format(
            self.name, self.start_idx, self.num_use,
            self.num))

    def shuffle(self):
        p = []
        indies = list(range(self.start_idx, self.start_idx + self.num))
        while len(p) < self.num_use:
            np.random.shuffle(indies)
            p += indies
        return p[:self.num_use]

    def __len__(self):
        return self.num


class TrainDataset(Dataset):
    def __init__(self):
        self.all_dataset = []
        self.num = 0
        start_idx = 0
        for name in cfg.DATASET.NAMES:
            sub_cfg = getattr(cfg.DATASET, name)
            sub_dataset = SubDataset(name, sub_cfg.DATA_DIR, sub_cfg.ANNO_FILE, sub_cfg.FRAME_RANGE, start_idx,
                                     sub_cfg.NUM_USE)
            sub_dataset.log()
            self.all_dataset.append(sub_dataset)
            start_idx += sub_dataset.num
            self.num += sub_dataset.num_use
        self.num = cfg.DATASET.VIDEO_PER_EPOCH if cfg.DATASET.VIDEO_PER_EPOCH > 0 else self.num
        self.anchor_target = AnchorTarget(cfg.ANCHOR.SCALES, cfg.ANCHOR.RATIOS, cfg.ANCHOR.STRIDE,
                                          cfg.TRAIN.SEARCH_SIZE // 2, cfg.TRAIN.OUTPUT_SIZE)
        self.template_aug = Augmentation(
            cfg.DATASET.EXAMPLAR.SHIFT,
            cfg.DATASET.EXAMPLAR.SCALE,
            cfg.DATASET.EXAMPLAR.BLUR,
            cfg.DATASET.EXAMPLAR.FLIP,
            cfg.DATASET.EXAMPLAR.COLOR
        )
        self.search_aug = Augmentation(
            cfg.DATASET.SEARCH.SHIFT,
            cfg.DATASET.SEARCH.SCALE,
            cfg.DATASET.SEARCH.BLUR,
            cfg.DATASET.SEARCH.FLIP,
            cfg.DATASET.SEARCH.COLOR
        )

    def shuffle(self):
        pick = []
        num = 0
        while num < self.num:
            p = []
            for sub_dataset in self.all_dataset:
                sub_p = sub_dataset.shuffle()
                p += sub_p
            np.random.shuffle(p)
            pick += p
            num = len(pick)
        self.pick = pick[:self.num]

    def _find_dataset(self, idx):
        for sub_dataset in self.all_dataset:
            if sub_dataset.start_idx + sub_dataset.num > idx:
                return sub_dataset, idx - sub_dataset.start_idx

    def get_bbox(self, image, ori_bbox):
        img_h, img_w = image.shape[:2]
        w, h = ori_bbox[2] - ori_bbox[0], ori_bbox[3] - ori_bbox[1]
        context_amount = 0.5
        wc_z = w + context_amount * (w + h)
        hc_z = h + context_amount * (w + h)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = cfg.TRAIN.EXAMPLER_SIZE / s_z
        w = w * scale_z
        h = h * scale_z
        cx, cy = img_w // 2, img_h // 2
        bbox = center2corner(Center(cx, cy, w, h))
        return bbox

    def __getitem__(self, idx):
        idx = self.pick[idx]
        sub_dataset, idx = self._find_dataset(idx)
        neg = cfg.DATASET.NEG and cfg.DATASET.NEG > np.random.random()
        gray = cfg.DATASET.GRAY and cfg.DATASET.GRAY > np.random.random()
        if neg:
            examplar = sub_dataset.get_random_target()
            search = np.random.choice(self.all_dataset).get_random_target()
        else:
            examplar, search = sub_dataset.get_postive_pair(idx)
        examplar_img = cv2.imread(examplar[0])
        search_img = cv2.imread(search[0])

        examplar_bbox = self.get_bbox(examplar_img, examplar[1])
        search_bbox = self.get_bbox(search_img, search[1])  # bbox: x1,y1,x2,y2

        examplar_img, examplar_bbox = self.template_aug(examplar_img,
                                                        examplar_bbox,
                                                        cfg.TRAIN.EXAMPLER_SIZE,
                                                        gray=gray)
        search_img, search_bbox = self.search_aug(search_img,
                                                  search_bbox,
                                                  cfg.TRAIN.SEARCH_SIZE,
                                                  gray=gray)

        examplar_img = examplar_img.transpose((2, 0, 1)).astype(np.float32)  # NOTE: set as c,h,w and type=float32
        search_img = search_img.transpose((2, 0, 1)).astype(np.float32)

        gt_cls, gt_delta, delta_weight = self.anchor_target(search_bbox, neg)
        return {
            'examplar_img': examplar_img,
            'search_img': search_img,
            'gt_cls': gt_cls,
            'gt_delta': gt_delta,
            'delta_weight': delta_weight
        }

    def __len__(self):
        return self.num


class MetaDataset(SubDataset):
    def __init__(self):
        super().__init__(name='VID',
                         data_dir=cfg.META.VID.DATA_DIR,
                         anno_file=cfg.META.VID.ANNO_FILE,
                         frame_range=cfg.META.VID.FRAME_RANGE,
                         start_idx=0,
                         num_use=cfg.META.VID.NUM_USE)
        self.anchor_target = AnchorTarget(cfg.ANCHOR.SCALES, cfg.ANCHOR.RATIOS, cfg.ANCHOR.STRIDE,
                                          cfg.TRAIN.SEARCH_SIZE // 2, cfg.TRAIN.OUTPUT_SIZE)
        self.examplar_aug = Augmentation(
            cfg.DATASET.EXAMPLAR.SHIFT,
            cfg.DATASET.EXAMPLAR.SCALE,
            cfg.DATASET.EXAMPLAR.BLUR,
            cfg.DATASET.EXAMPLAR.FLIP,
            cfg.DATASET.EXAMPLAR.COLOR
        )
        self.search_aug = Augmentation(
            cfg.DATASET.SEARCH.SHIFT,
            cfg.DATASET.SEARCH.SCALE,
            cfg.DATASET.SEARCH.BLUR,
            cfg.DATASET.SEARCH.FLIP,
            cfg.DATASET.SEARCH.COLOR
        )

    def __getitem__(self, idx):
        examplar_frame, train_frames, test_frames = self.get_anno(idx)
        examplar_img = cv2.imread(examplar_frame[0])
        examplar_bbox = self.get_bbox(examplar_img, examplar_frame[1])

        examplar_img, _ = self.examplar_aug(examplar_img,
                                            examplar_bbox,
                                            cfg.TRAIN.EXAMPLER_SIZE,
                                            gray=False)
        # train set
        train_imgs = [cv2.imread(train_path) for train_path in train_frames[0]]
        train_bboxes = [self.get_bbox(img, anno)
                        for img, anno in zip(train_imgs, train_frames[1])]
        train_set = [self.search_aug(train_img, train_bbox, cfg.TRAIN.SEARCH_SIZE, gray=False)
                     for train_img, train_bbox in zip(train_imgs, train_bboxes)]
        train_imgs, train_bboxes = zip(*train_set)
        train_imgs, train_bboxes = list(train_imgs), list(train_bboxes)
        # test set
        test_imgs = [cv2.imread(test_path) for test_path in test_frames[0]]
        test_bboxes = [self.get_bbox(img, anno) for img, anno in zip(test_imgs, test_frames[1])]
        test_set = [self.search_aug(test_img, test_bbox, cfg.TRAIN.SEARCH_SIZE, gray=False)
                    for test_img, test_bbox in zip(test_imgs, test_bboxes)]
        test_imgs, test_bboxes = zip(*test_set)
        test_imgs, test_bboxes = list(test_imgs), list(test_bboxes)

        train_examplar_imgs, test_examplar_imgs = map(
            lambda x: np.tile(x.transpose((2, 0, 1)).astype(np.float32), (len(train_imgs), 1, 1, 1)),
            [examplar_img, examplar_img])
        train_imgs, test_imgs = map(
            lambda x: np.stack(x, axis=0).transpose((0, 3, 1, 2)).astype(np.float32),
            [train_imgs, test_imgs])
        # train
        gt_data = zip(*[self.anchor_target(bbox) for bbox in train_bboxes])
        train_cls, train_delta, train_delta_weight = map(lambda x: np.stack(x, axis=0), gt_data)
        # test
        gt_data = zip(*[self.anchor_target(bbox) for bbox in test_bboxes])
        test_cls, test_delta, test_delta_weight = map(
            lambda x: np.stack(x, axis=0), gt_data)
        return {
            'train_examplar_imgs': train_examplar_imgs,
            'test_examplar_imgs': test_examplar_imgs,
            'train_search_imgs': train_imgs,
            'test_search_imgs': test_imgs,
            'train_cls': train_cls,
            'train_delta': train_delta,
            'train_delta_weight': train_delta_weight,
            'test_cls': test_cls,
            'test_delta': test_delta,
            'test_delta_weight': test_delta_weight
        }

    def get_anno(self, idx):
        video = self.videos[idx]
        trackid = np.random.choice(list(self.annos[video].keys()))
        frames = self.annos[video][trackid]['frames']
        half = len(frames) // 2
        left = 0
        right = max(half, 1)
        examplar_frame = np.random.choice(frames[left:right])
        train_range = frames[left:right]
        train_frames = np.random.choice(train_range, size=cfg.META.TRAIN_SIZE, replace=True)
        left = half
        right = max(half + 1, len(frames) - 1)
        test_range = frames[left:right]
        test_frames = np.random.choice(
            test_range, size=cfg.META.TEST_SIZE, replace=True)
        examplar_frame = '{:06d}'.format(examplar_frame)
        train_frames = ['{:06d}'.format(train_frame) for train_frame in train_frames]
        test_frames = ['{:06d}'.format(test_frame) for test_frame in test_frames]
        examplar_path = os.path.join(self.data_dir, video, self.filename_format.format(examplar_frame, trackid, 'x'))
        train_paths = [os.path.join(self.data_dir, video, self.filename_format.format(train_frame, trackid, 'x'))
                       for train_frame in train_frames]
        test_paths = [os.path.join(self.data_dir, video, self.filename_format.format(test_frame, trackid, 'x'))
                      for test_frame in test_frames]
        examplar_anno = self.annos[video][trackid][examplar_frame]
        train_annos = [self.annos[video][trackid][train_frame] for train_frame in train_frames]
        test_annos = [self.annos[video][trackid][test_frame] for test_frame in test_frames]
        return (examplar_path, examplar_anno), (train_paths, train_annos), (test_paths, test_annos)

    def get_bbox(self, image, ori_bbox):
        img_h, img_w = image.shape[:2]
        w, h = ori_bbox[2] - ori_bbox[0], ori_bbox[3] - ori_bbox[1]
        context_amount = 0.5
        wc_z = w + context_amount * (w + h)
        hc_z = h + context_amount * (w + h)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = cfg.TRAIN.EXAMPLER_SIZE / s_z
        w = w * scale_z
        h = h * scale_z
        cx, cy = img_w // 2, img_h // 2
        bbox = center2corner(Center(cx, cy, w, h))
        return bbox
