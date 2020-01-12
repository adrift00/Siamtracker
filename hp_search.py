from __future__ import division
from __future__ import print_function

import os
import cv2
import argparse
import torch
import numpy as np

from toolkit.utils.region import vot_overlap, vot_float2str
from models.model_builder import BaseSiamModel
from trackers import get_tracker
from toolkit.datasets import get_dataset
from utils.model_load import load_pretrain
from configs.config import cfg
from utils.visual import show_double_bbox

torch.set_num_threads(1)


def parse_range(range_str):
    param = map(float, range_str.split(','))
    return np.arange(*param)


def parse_range_int(range_str):
    param = map(int, range_str.split(','))
    return np.arange(*param)


parser = argparse.ArgumentParser(description='Hyperparamter search')
parser.add_argument('--snapshot', type=str, help='snapshot of model')
parser.add_argument('--dataset', type=str, help='dataset name to eval')
parser.add_argument('--penalty-k', default='0.05,0.5,0.05', type=parse_range)
parser.add_argument('--lr', default='0.35,0.5,0.05', type=parse_range)
parser.add_argument('--window-influence', default='0.1,0.8,0.05', type=parse_range)
parser.add_argument('--search-region', default='255,256,8', type=parse_range_int)
parser.add_argument('--config', default='config.yaml', type=str)
parser.add_argument('--vis', action='store_true', help='whether to visual')
args = parser.parse_args()


def run_tracker(tracker, gt, video_name, restart=True):
    frame_count = 0
    lost_number = 0
    pred_bboxes = []
    toc = 0
    if restart:
        for idx, (frame, gt_bbox) in enumerate(video):
            tic = cv2.getTickCount()
            if idx == frame_count:
                tracker.init(frame, gt_bbox)  # cx,cy,w,h
                pred_bboxes.append(1)
            elif idx > frame_count:
                track_result = tracker.track(frame)
                bbox = track_result['bbox']  # cx,cy,w,h
                score = track_result['score']
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
                show_double_bbox(frame, bbox, score, gt_bbox, idx, lost_number)
        toc /= cv2.getTickFrequency()
        # log
        print('video: {}, time: {:.1f}s, speed: {:.1f}fps, lost_number: {:d} '.format(video_name,
                                                                                      toc, idx / toc,
                                                                                      lost_number))
        return pred_bboxes
    else:
        # toc = 0
        # pred_bboxes = []
        # scores = []
        # track_times = []
        # for idx, (img, gt_bbox) in enumerate(video):
        #     tic = cv2.getTickCount()
        #     if idx == 0:
        #         cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
        #         gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
        #         tracker.init(img, gt_bbox_)
        #         pred_bbox = gt_bbox_
        #         scores.append(None)
        #         pred_bboxes.append(pred_bbox)
        #     else:
        #         outputs = tracker.track(img)
        #         pred_bbox = outputs['bbox']
        #         pred_bboxes.append(pred_bbox)
        #         scores.append(outputs['best_score'])
        #     toc += cv2.getTickCount() - tic
        #     track_times.append((cv2.getTickCount() - tic) / cv2.getTickFrequency())
        # toc /= cv2.getTickFrequency()
        # print('Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
        #     video_name, toc, idx / toc))
        # return pred_bboxes, scores, track_times
        pass


def _check_and_occupation(video_path, result_path):
    if os.path.isfile(result_path):
        return True
    try:
        if not os.path.isdir(video_path):
            os.makedirs(video_path)
    except OSError as err:
        print(err)

    with open(result_path, 'w') as f:
        f.write('Occ')
    return False


if __name__ == '__main__':
    num_search = len(args.penalty_k) \
                 * len(args.window_influence) \
                 * len(args.lr) \
                 * len(args.search_region)
    print("Total search number: {}".format(num_search))

    cfg.merge_from_file(args.config)

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_root = os.path.join(cur_dir, '../testing_dataset', args.dataset)

    # create dataset
    data_dir = os.path.join(cfg.TRACK.DATA_DIR, args.dataset)
    dataset = get_dataset(args.dataset,data_dir)

    # create model
    model = BaseSiamModel()

    # load model
    model = load_pretrain(model, args.snapshot).cuda().eval()

    # build tracker
    tracker_name='SiamRPN'
    tracker = get_tracker(tracker_name, model)

    model_name = args.snapshot.split('/')[-1].split('.')[0]
    benchmark_path = os.path.join('hp_search_result', args.dataset)
    seqs = list(range(len(dataset)))
    np.random.shuffle(seqs)
    for idx in seqs:
        video = dataset[idx]
        # load image
        np.random.shuffle(args.penalty_k)
        np.random.shuffle(args.window_influence)
        np.random.shuffle(args.lr)
        for pk in args.penalty_k:
            for wi in args.window_influence:
                for lr in args.lr:
                    for ins in args.search_region:
                        cfg.TRACK.PENALTY_K = float(pk)
                        cfg.TRACK.WINDOW_INFLUENCE = float(wi)
                        cfg.TRACK.LR = float(lr)
                        cfg.TRACK.INSTANCE_SIZE = int(ins)
                        # rebuild tracker
                        tracker = get_tracker(tracker_name,model)
                        tracker_path = os.path.join(benchmark_path,
                                                    (model_name +
                                                     '_r{}'.format(ins) +
                                                     '_pk-{:.3f}'.format(pk) +
                                                     '_wi-{:.3f}'.format(wi) +
                                                     '_lr-{:.3f}'.format(lr)))
                        if 'VOT2016' == args.dataset or 'VOT2018' == args.dataset:
                            video_path = os.path.join(tracker_path, 'baseline', video.name)
                            result_path = os.path.join(video_path, video.name + '_001.txt')
                            if _check_and_occupation(video_path, result_path):
                                continue
                            pred_bboxes = run_tracker(tracker,video.gt_rects, video.name, restart=True)
                            with open(result_path, 'w') as f:
                                for x in pred_bboxes:
                                    if isinstance(x, int):
                                        f.write("{:d}\n".format(x))
                                    else:
                                        f.write(','.join([vot_float2str("%.4f", i) for i in x]) + '\n')
                        elif 'VOT2018-LT' == args.dataset:
                            video_path = os.path.join(tracker_path, 'longterm', video.name)
                            result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
                            if _check_and_occupation(video_path, result_path):
                                continue
                            pred_bboxes, scores, track_times = run_tracker(tracker,
                                                                           video.imgs, video.gt_traj, video.name,
                                                                           restart=False)
                            pred_bboxes[0] = [0]
                            with open(result_path, 'w') as f:
                                for x in pred_bboxes:
                                    f.write(','.join([str(i) for i in x]) + '\n')
                            result_path = os.path.join(video_path,
                                                       '{}_001_confidence.value'.format(video.name))
                            with open(result_path, 'w') as f:
                                for x in scores:
                                    f.write('\n') if x is None else f.write("{:.6f}\n".format(x))
                            result_path = os.path.join(video_path,
                                                       '{}_time.txt'.format(video.name))
                            with open(result_path, 'w') as f:
                                for x in track_times:
                                    f.write("{:.6f}\n".format(x))
                        elif 'GOT-10k' == args.dataset:
                            video_path = os.path.join('epoch_result', tracker_path, video.name)
                            if not os.path.isdir(video_path):
                                os.makedirs(video_path)
                            result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
                            with open(result_path, 'w') as f:
                                for x in pred_bboxes:
                                    f.write(','.join([str(i) for i in x]) + '\n')
                            result_path = os.path.join(video_path,
                                                       '{}_time.txt'.format(video.name))
                            with open(result_path, 'w') as f:
                                for x in track_times:
                                    f.write("{:.6f}\n".format(x))
                        else:
                            result_path = os.path.join(tracker_path, '{}.txt'.format(video.name))
                            if _check_and_occupation(tracker_path, result_path):
                                continue
                            pred_bboxes, _, _ = run_tracker(tracker, video.imgs,
                                                            video.gt_traj, video.name, restart=False)
                            with open(result_path, 'w') as f:
                                for x in pred_bboxes:
                                    f.write(','.join([str(i) for i in x]) + '\n')
