import os
import argparse
import cv2
import logging
import torch
from dataset.vot_dataset import VOTDataset
from utils.model_load import load_pretrain
from models.model_builder import get_model
from configs.config import cfg
from trackers import get_tracker
from utils.visual import show_double_bbox
from toolkit.utils.region import vot_overlap
from utils.log_helper import init_log
parse = argparse.ArgumentParser(description='test tracker')
parse.add_argument('--tracker', default='', type=str, help='which tracker to use')
parse.add_argument('--dataset', default='', type=str, help='which dataset to test')
parse.add_argument('--cfg', default='', type=str, help='cfg file to use')
parse.add_argument('--snapshot', default='', type=str, help='base snapshot for track')
parse.add_argument('--video', default='', type=str, help='choose one special video to test')
parse.add_argument('--vis', action='store_true', help='whether to visual')
args = parse.parse_args()

torch.set_num_threads(1)  # use only one threads to test the real speed


def test_tracker(data_dir, anno_file, visual=False):
    cfg.merge_from_file(args.cfg)
    init_log('global', logging.INFO)
    test_dataset = VOTDataset(data_dir, anno_file)
    # get the base model, may can be refracted.
    if args.tracker == 'SiamRPN':
        model_name = 'BaseSiamModel'
    elif args.tracker == 'MetaSiamRPN':
        model_name = 'MetaSiamModel'
    elif args.trackers == 'GraphSiamRPN':
        model_name = 'GraphSiamModel'
    else:
        raise Exception('tracker is valid')
    base_model = get_model(model_name)
    base_model = load_pretrain(base_model, args.snapshot).cuda().eval()
    tracker = get_tracker(args.tracker, base_model)
    tracker_name = args.tracker
    backbone_name = args.cfg.split('/')[-1].split('_')[0]
    snapshot_name = args.snapshot.split('/')[-1].split('.')[0]
    total_lost = 0
    for v_idx, video in enumerate(test_dataset):
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
            if visual and idx > frame_count:
                show_double_bbox(frame, bbox, gt_bbox, idx, lost_number)
        toc /= cv2.getTickFrequency()
        result_dir = os.path.join(cfg.TRACK.RESULT_DIR, args.dataset, tracker_name, backbone_name, snapshot_name)
        if not os.path.isdir(result_dir):
            os.makedirs(result_dir)
        result_path = '{}/{}.txt'.format(result_dir, video.name)
        with open(result_path, 'w') as f:
            for x in pred_bboxes:
                if isinstance(x, int):
                    f.write('{:d}\n'.format(x))
                else:
                    f.write(','.join(['{:.4f}'.format(i) for i in x]) + '\n')
        # log
        total_lost += lost_number
        print('[{:d}/{:d}] video: {}, time: {:.1f}s, speed: {:.1f}fps, lost_number: {:d} '.format(v_idx+1,
                                                                                                  len(test_dataset),
                                                                                                  video.name,
                                                                                                  toc, idx / toc,
                                                                                                  lost_number))
    print('total_lost: {}'.format(total_lost))


if __name__ == '__main__':
    if args.dataset == 'VOT2016':
        data_dir = os.path.join(cfg.TRACK.DATA_DIR, 'VOT2016')
        anno_file = 'VOT2016.json'
        test_tracker(data_dir, anno_file, visual=args.vis)
    elif args.dataset == 'VOT2018':
        data_dir = os.path.join(cfg.TRACK.DATA_DIR, 'VOT2018')
        anno_file = 'VOT2018.json'
        test_tracker(data_dir, anno_file, visual=args.vis)
    else:
        raise Exception('dataset invalid!')
