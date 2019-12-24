import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class Dataset(Dataset):
    def __init__(self, data_dir):
        anno_file = '{}.json'.format(data_dir.split('/')[-1])
        anno_path = os.path.join(data_dir, anno_file)
        with open(anno_path, 'r') as f:
            self.anno_info = json.load(f)
        self.videos = {}

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
