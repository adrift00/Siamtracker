import os
import argparse
import cv2
import logging
import torch
from toolkit.datasets import get_dataset
from configs.config import cfg
from utils.visual import show_double_bbox
from toolkit.utils.region import vot_overlap
from utils.log_helper import init_log
from trackers.siamrpn_MNN import SiamRPN_MNN
parser = argparse.ArgumentParser(description='test tracker')
parser.add_argument('--cfg', default='', type=str, help='cfg file to use')
parser.add_argument('--dataset', type=str, help='dataset name to eval')
parser.add_argument('--video', default='', type=str, help='choose one special video to test')
parser.add_argument('--vis', action='store_true', help='whether to visual')
args = parser.parse_args()

torch.set_num_threads(1)  # use only one threads to test the real speed


def vot_evaluate(dataset, tracker):
    total_lost = 0
    for v_idx, video in enumerate(dataset):
        if args.video != '':  # if test special video
            if video.name != args.video:
                continue
        frame_count = 0
        lost_number = 0
        pred_bboxes = []
        toc = 0
        for idx, (frame, gt_bbox) in enumerate(video):
            tic = cv2.getTickCount()
            if idx == frame_count:
                tracker.init(frame, gt_bbox)  # cx,cy,w,h
                pred_bboxes.append(1)
            elif idx > frame_count:
                track_result = tracker.track(frame)
                bbox = track_result['bbox']  # cx,cy,w,h
                score=track_result['score']
                bbox_ = [bbox[0] - bbox[2] / 2, bbox[1] - bbox[3] / 2, bbox[2], bbox[3]]  # x,y,w,h
                gt_bbox_ = [gt_bbox[0] - gt_bbox[2] / 2, gt_bbox[1] - gt_bbox[3] / 2, gt_bbox[2], gt_bbox[3]]
                if vot_overlap(bbox_, gt_bbox_, (frame.shape[1], frame.shape[0])) > 0:
                    pred_bboxes.append(bbox_)
                else:
                    pred_bboxes.append(2)
                    frame_count = idx + 5
                    lost_number += 1
            else:
                pred_bboxes.append(0)

            toc += cv2.getTickCount() - tic
            if args.vis and idx > frame_count:
                show_double_bbox(frame, bbox,score, gt_bbox, idx, lost_number)
        toc /= cv2.getTickFrequency()
        # log
        total_lost += lost_number
        print('[{:d}/{:d}] video: {}, time: {:.1f}s, speed: {:.1f}fps, lost_number: {:d} '.format(v_idx+1,
                                                                                                  len(dataset),
                                                                                                  video.name,
                                                                                                  toc, idx / toc,
                                                                                                  lost_number))
    print('total_lost: {}'.format(total_lost))



def main():
    cfg.merge_from_file(args.cfg)
    init_log('global', logging.INFO)
    tracker = SiamRPN_MNN()
    data_dir = os.path.join(cfg.TRACK.DATA_DIR, args.dataset)
    dataset = get_dataset(args.dataset, data_dir)
    if args.dataset in ['VOT2016', 'VOT2018']:
        vot_evaluate(dataset,tracker)


if __name__ == '__main__':
    main()


