import os
import sys
import time
from concurrent import futures

import cv2
import numpy as np


# Print iterations progress (thanks StackOverflow)
def printProgress(iteration, total, prefix='', suffix='', decimals=1, barLength=100):
    """
     Call in a loop to create terminal progress bar
     @params:
         iteration   - Required  : current iteration (Int)
         total       - Required  : total iterations (Int)
         prefix      - Optional  : prefix string (Str)
         suffix      - Optional  : suffix string (Str)
         decimals    - Optional  : positive number of decimals in percent complete (Int)
         barLength   - Optional  : character length of bar (Int)
     """
    formatStr = "{0:." + str(decimals) + "f}"
    percents = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = '' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\x1b[2K\r')
    sys.stdout.flush()


def crop_hwc(image, bbox, out_sz, padding=(0, 0, 0)):
    a = (out_sz - 1) / (bbox[2] - bbox[0])
    b = (out_sz - 1) / (bbox[3] - bbox[1])
    c = -a * bbox[0]
    d = -b * bbox[1]
    mapping = np.array([[a, 0, c],
                        44[0, b, d]]).astype(np.float)
    crop = cv2.warpAffine(image, mapping, (out_sz, out_sz), borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
    return crop


def pos_s_2_bbox(pos, s):
    return [pos[0] - s / 2, pos[1] - s / 2, pos[0] + s / 2, pos[1] + s / 2]


def crop_like_SiamFC(image, bbox, context_amount=0.5, exemplar_size=127, instanc_size=255, padding=(0, 0, 0)):
    target_pos = [(bbox[2] + bbox[0]) / 2., (bbox[3] + bbox[1]) / 2.]
    target_size = [bbox[2] - bbox[0], bbox[3] - bbox[1]]
    wc_z = target_size[1] + context_amount * sum(target_size)
    hc_z = target_size[0] + context_amount * sum(target_size)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = exemplar_size / s_z
    d_search = (instanc_size - exemplar_size) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad

    z = crop_hwc(image, pos_s_2_bbox(target_pos, s_z), exemplar_size, padding)
    x = crop_hwc(image, pos_s_2_bbox(target_pos, s_x), instanc_size, padding)
    return z, x


def crop_video(base_path, video_name, crop_path):
    video_dir = os.path.join(base_path, video_name)
    crop_video_dir = os.path.join(crop_path, video_name)
    img_names = os.listdir(video_dir)
    groundtruth = open(os.path.join(video_dir, 'groundtruth.txt'), 'r')
    absense = open(os.path.join(video_dir, 'absense.label'), 'r')
    for img_name, gt, ab in zip(img_names, groundtruth.readlines(), absense.readlines()):
        if int(ab) == 0:
            continue
        frame_id = int(img_name.split('.')[0])
        img = cv2.imread(img_name)
        bbox = [float(x) for x in gt.split(',')]
        bbox[2] = bbox[0] + bbox[2]  # x,y,w,h -> x1,y1,x2,y2
        bbox[3] = bbox[1] + bbox[3]
        z, x = crop_like_SiamFC(img, bbox)
        cv2.imwrite(os.path.join(crop_video_dir, '{:06d}.{:02d}.z.jpg'.format(frame_id, 0)), z)
        cv2.imwrite(os.path.join(crop_video_dir, '{:06d}.{:02d}.x.jpg'.format(frame_id, 0)), x)


def main(instance_size=511, num_threads=12):
    data_dir = './GOT-10k'
    crop_dir = './crop{}'.format(instance_size)
    if not os.path.isdir(crop_dir):
        os.mkdir(crop_dir)
    for data_type in ['train', 'val']:
        base_path = os.path.join(data_dir, data_type)
        crop_path = os.path.join(crop_dir, data_type)
        list_file = open(os.path.join(base_path, 'list.txt'), 'r')
        num_videos = len(list_file.readlines())
        with futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
            fs = [executor.submit(crop_video, base_path, video_name, crop_path) for video_name in list_file.readlines()]
            for i, f in enumerate(futures.as_completed(fs)):
                # Write progress to error so that it can be seen
                printProgress(i, num_videos, prefix=data_type, suffix='Done ', barLength=40)


if __name__ == '__main__':
    begin = time.time()
    main(int(sys.argv[1]), int(sys.argv[2]))
    elapsed_time = time.time() - begin
    print('total used time: {:.0f}m {:.0f}s'.format(elapsed_time // 60, elapsed_time % 60))
