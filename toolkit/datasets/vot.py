import os
import cv2
import numpy as np
from tqdm import tqdm
from .dataset import Dataset
from .video import Video
class VOTDataset(Dataset):
    def __init__(self, data_dir):
        super(VOTDataset,self).__init__(data_dir)

        for video, video_info in tqdm(self.anno_info.items()):
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



class VOTVideo(Video):
    def __init__(self, name, data_dir, init_rect, img_names, gt_rects,
                 camera_motion, illum_change, motion_change, size_change, occlusion):

        super(VOTVideo,self).__init__(name, data_dir, init_rect, img_names, gt_rects)
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


    # TODO: understand it
    def select_tag(self, tag, start=0, end=0):
        if tag == 'empty':
            return self.tags[tag]
        return self.tags[tag][start:end]


