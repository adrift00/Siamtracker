import os
import argparse
import cv2
import logging
import torch

from pruning_model import prune_model
from toolkit.datasets import get_dataset
from utils.model_load import load_pretrain
from models import get_model
from configs.config import cfg
from trackers import get_tracker
from utils.visual import show_double_bbox
from toolkit.utils.region import vot_overlap
from utils.log_helper import init_log

parser = argparse.ArgumentParser(description='test tracker')
parser.add_argument('--tracker', default='', type=str, help='which tracker to use')
parser.add_argument('--dataset', default='', type=str, help='which dataset to test')
parser.add_argument('--cfg', default='', type=str, help='cfg file to use')
parser.add_argument('--snapshot', default='', type=str, help='base snapshot for track')
parser.add_argument('--video', default='', type=str, help='choose one special video to test')
parser.add_argument('--vis', action='store_true', help='whether to visual')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.set_num_threads(1)  # use only one threads to test the real speed


def vot_evaluate(dataset, tracker):
    tracker_name = args.tracker
    backbone_name = args.cfg.split('/')[-1].split('_')[0]
    snapshot_name = args.snapshot.split('/')[-1].split('.')[0]
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
                score = track_result['score']
                bbox_ = [bbox[0] - bbox[2] / 2, bbox[1] - bbox[3] / 2, bbox[2], bbox[3]]  # x,y,w,h
                gt_bbox_ = [gt_bbox[0] - (gt_bbox[2] - 1) / 2,
                            gt_bbox[1] - (gt_bbox[3] - 1) / 2,
                            gt_bbox[2],
                            gt_bbox[3]]
                overlap = vot_overlap(bbox_, gt_bbox_, (frame.shape[1], frame.shape[0]))
                # print('idx: {}\n pred: {}\n gt: {}\n overlap: {}\n'.format(idx, bbox_, gt_bbox_, overlap))
                if overlap > 0:
                    pred_bboxes.append(bbox_)
                else:
                    # print('lost idx: {}'.format(idx))
                    pred_bboxes.append(2)
                    frame_count = idx + 5
                    lost_number += 1
            else:
                pred_bboxes.append(0)

            toc += cv2.getTickCount() - tic
            if args.vis and idx > frame_count:
                show_double_bbox(frame, bbox, score, gt_bbox, idx, lost_number)
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

        print('[{:d}/{:d}] | video: {:12s} | time: {:4.1f}s | speed: {:3.1f}fps | lost_number: {:d} ' \
              .format(v_idx + 1, len(dataset), video.name, toc, idx / toc, lost_number))
    print('total_lost: {}'.format(total_lost))


def ope_evaluate(dataset, tracker):
    tracker_name = args.tracker
    backbone_name = args.cfg.split('/')[-1].split('_')[0]
    snapshot_name = args.snapshot.split('/')[-1].split('.')[0]
    for v_idx, video in enumerate(dataset):
        if args.video != '':  # if test special video
            if video.name != args.video:
                continue
        pred_bboxes = []
        runtime = []
        toc = 0
        for idx, (frame, gt_bbox) in enumerate(video):
            tic = cv2.getTickCount()
            if idx == 0:
                tracker.init(frame, gt_bbox)  # cx,cy,w,h
                track_result = tracker.track(frame)
                bbox = track_result['bbox']  # cx,cy,w,h
                score = track_result['score']
                bbox_ = [bbox[0] - bbox[2] / 2, bbox[1] - bbox[3] / 2, bbox[2], bbox[3]]  # x,y,w,h
                gt_bbox_ = [gt_bbox[0] - gt_bbox[2] / 2, gt_bbox[1] - gt_bbox[3] / 2, gt_bbox[2], gt_bbox[3]]
                pred_bboxes.append(bbox_)
            else:
                track_result = tracker.track(frame)
                bbox = track_result['bbox']  # cx,cy,w,h
                score = track_result['score']
                bbox_ = [bbox[0] - bbox[2] / 2, bbox[1] - bbox[3] / 2, bbox[2], bbox[3]]  # x,y,w,h
                gt_bbox_ = [gt_bbox[0] - gt_bbox[2] / 2, gt_bbox[1] - gt_bbox[3] / 2, gt_bbox[2], gt_bbox[3]]
                pred_bboxes.append(bbox_)

            toc += cv2.getTickCount() - tic
            runtime.append((cv2.getTickCount() - tic) / cv2.getTickFrequency())
            if args.vis and idx > 0:
                show_double_bbox(frame, bbox, score, gt_bbox, idx, 0)
        toc /= cv2.getTickFrequency()
        result_dir = os.path.join(cfg.TRACK.RESULT_DIR, args.dataset, tracker_name, backbone_name, snapshot_name,
                                  video.name)
        if not os.path.isdir(result_dir):
            os.makedirs(result_dir)
        result_path = '{}/{}_001.txt'.format(result_dir, video.name)
        runtime_path = '{}/{}_time.txt'.format(result_dir, video.name)
        # write result
        with open(result_path, 'w') as f:
            for x in pred_bboxes:
                if isinstance(x, int):
                    f.write('{:d}\n'.format(x))
                else:
                    f.write(','.join(['{:.4f}'.format(i) for i in x]) + '\n')
        # write runtime
        with open(runtime_path, 'w') as f:
            for time in runtime:
                f.write('{:.6f}\n'.format(time))

        # log
        print('[{:d}/{:d}] video: {}, time: {:.1f}s, speed: {:.1f}fps'.format(v_idx + 1,
                                                                              len(dataset),
                                                                              video.name,
                                                                              toc, idx / toc))


def seed_torch(seed=0):
    import random
    import numpy as np
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    seed_torch(123456)
    cfg.merge_from_file(args.cfg)
    init_log('global', logging.INFO)
    # get the base model, may can be refracted.
    # TODO: use the config in the yaml file,instead of this
    if args.tracker == 'SiamRPN':
        model_name = 'BaseSiamModel'
    elif args.tracker == 'MetaSiamRPN':
        model_name = 'MetaSiamModel'
    elif args.tracker == 'GraphSiamRPN':
        model_name = 'GraphSiamModel'
    elif args.tracker == 'GradSiamRPN':
        model_name = 'GradSiamModel'
    else:
        raise Exception('tracker is valid')
    # normal use
    # base_model = get_model(model_name)
    # base_model = load_pretrain(base_model, args.snapshot).cuda().eval()
    # base_model = base_model.cuda().eval()

    # test the pruning model
    # base_model = get_model('PruningSiamModel')
    # base_model = load_pretrain(base_model, args.snapshot)
    # base_model = pruning_model(base_model).cuda().eval()
    # torch.save(base_model.state_dict(), './snapshot/mobilenetv2_gdp/model_pruning.pth')

    # test the model after real prunging
    base_model = get_model('PruningSiamModel')
    base_model = load_pretrain(base_model, cfg.PRUNING.FINETUNE.PRETRAIN_PATH) # load the mask
    base_model = prune_model(base_model) # refine the model
    base_model=load_pretrain(base_model,args.snapshot).cuda().eval() # load the finetune weight
    # grad
    # base_model = get_model(model_name)
    # base_model = load_pretrain(base_model, args.snapshot)
    # base_model = base_model.cuda().eval()
    # base_model.freeze_model()

    tracker = get_tracker(args.tracker, base_model)
    data_dir = os.path.join(cfg.TRACK.DATA_DIR, args.dataset)
    dataset = get_dataset(args.dataset, data_dir)
    if args.dataset in ['VOT2016', 'VOT2018']:
        vot_evaluate(dataset, tracker)
    elif args.dataset == 'GOT-10k':
        ope_evaluate(dataset, tracker)


if __name__ == '__main__':
    main()
