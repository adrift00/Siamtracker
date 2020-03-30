import os
import json

if __name__ == '__main__':
    anno = {}
    base_dir = './GOT-10k'
    for data_type in ['train', 'val']:
        list_file = open(os.path.join(base_dir, data_type, 'list.txt'), 'r')
        for video_name in list_file.readlines():
            video_id = '{}/{}'.format(data_type, video_name)
            anno[video_id] = {}
            video_dir = os.path.join(base_dir, data_type, video_name)
            frames = os.listdir(video_dir)
            groundtruth = open(os.path.join(video_dir, 'groundtruth.txt'), 'r')
            absense = open(os.path.join(video_dir, 'absense.label'), 'r')
            track_id = '00'
            anno[video_id][track_id] = {}
            for frame, gt, ab in zip(frames, groundtruth.readlines(), absense.readlines()):
                if int(ab) == 0:
                    continue
                frame_id = '{:06d}'.format(int(frame.split('.')[0]))
                bbox = [float(x) for x in gt.split(',')]
                bbox[2] = bbox[0] + bbox[2]  # x,y,w,h -> x1,y1,x2,y2
                bbox[3] = bbox[1] + bbox[3]
                anno[video_id][track_id][frame_id] = bbox
    json.dump(anno, open('GOT-10k.json','w'), indent=4, sort_keys=True)
