from tqdm import tqdm
from .dataset import Dataset
from .video import Video


class GOT10kDataset(Dataset):
    def __init__(self, data_dir):
        super(GOT10kDataset, self).__init__(data_dir)

        for video, video_info in tqdm(self.anno_info.items()):
            self.videos[video] = GOT10kVideo(video,
                                             data_dir,
                                             video_info['init_rect'],
                                             video_info['img_names'],
                                             video_info['gt_rect']
                                             )


class GOT10kVideo(Video):
    def __init__(self, name, data_dir, init_rect, img_names, gt_rects):
        super(GOT10kVideo, self).__init__(name, data_dir, init_rect, img_names, gt_rects)
