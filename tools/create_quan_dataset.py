import os
import cv2
import numpy as np
from torch.utils.data import DataLoader

from configs.config import cfg
from dataset.dataset import TrainDataset

if __name__ == '__main__':
    cfg.TRAIN.BATCH_SIZE = 1
    cfg.TRAIN.OUTPUT_SIZE=25
    quan_dataset_dir = '/home/keyan/NewDisk/ZhangXiong/quan_dataset'
    if not os.path.isdir(quan_dataset_dir):
        os.makedirs(quan_dataset_dir)
    train_dataset = TrainDataset()

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=cfg.TRAIN.BATCH_SIZE,
                                  num_workers=cfg.TRAIN.NUM_WORKERS,
                                  pin_memory=True
                                  )
    train_dataloader.dataset.shuffle()

    for idx, data in enumerate(train_dataloader):
        print(idx)
        if idx > 5000:
            break
        examplar_img = data['examplar_img'].cuda()
        search_img = data['search_img'].cuda()
        bbox = data['bbox'].cpu().numpy()
        gt_cls = data['gt_cls'].cuda()
        gt_delta = data['gt_delta'].cuda()
        gt_delta_weight = data['delta_weight'].cuda()

        examplar_img = examplar_img.cpu().numpy().astype(np.uint8)
        examplar_img = examplar_img.squeeze(0).transpose((1, 2, 0))
        examplar_path = os.path.join(quan_dataset_dir, 'examplar', 'examplar_{}.png'.format(idx + 1))
        cv2.imwrite(examplar_path, examplar_img)

        search_img = search_img.cpu().numpy().astype(np.uint8)
        search_img = search_img.squeeze(0).transpose((1, 2, 0))
        search_path = os.path.join(quan_dataset_dir, 'search', 'search_{}.png'.format(idx + 1))
        cv2.imwrite(search_path, search_img)

        gt_cls = gt_cls.cpu().numpy().squeeze(0)
        np.save(os.path.join(quan_dataset_dir, 'gt_cls', 'gt_cls_{}.npy'.format(idx + 1)), gt_cls)
        gt_delta = gt_delta.cpu().numpy().squeeze(0)
        gt_delta = gt_delta.reshape(-1, 25, 25)
        np.save(os.path.join(quan_dataset_dir, 'gt_delta', 'gt_delta_{}.npy'.format(idx + 1)), gt_delta)
        gt_delta_weight = gt_delta_weight.cpu().numpy().squeeze(0)
        np.save(os.path.join(quan_dataset_dir, 'gt_delta_weight', 'gt_delta_weight_{}.npy'.format(idx + 1)),
                gt_delta_weight)
