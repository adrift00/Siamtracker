import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from configs.config import cfg
from dataset.dataset import TrainDataset
from models import get_model
from pruning_model import prune_model
from utils.model_load import load_pretrain


def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    seed_torch(123456)
    cfg.merge_from_file("configs/mobilenetv2_pruning.yaml")
    cfg.TRAIN.BATCH_SIZE = 1
    cfg.TRAIN.OUTPUT_SIZE=25
    quan_dataset_dir = '/home/keyan/NewDisk/ZhangXiong/quant_dataset'
    if not os.path.isdir(quan_dataset_dir):
        os.makedirs(quan_dataset_dir)
    train_dataset = TrainDataset()

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=cfg.TRAIN.BATCH_SIZE,
                                  num_workers=cfg.TRAIN.NUM_WORKERS,
                                  pin_memory=True
                                  )
    train_dataloader.dataset.shuffle()

    base_model = get_model('PruningSiamModel')
    base_model = load_pretrain(base_model, cfg.PRUNING.FINETUNE.PRETRAIN_PATH) # load the mask
    base_model = prune_model(base_model) # refine the model
    base_model=load_pretrain(base_model,'./snapshot/mobilenetv2_sfp_0_75_finetune/checkpoint_e20.pth').cuda()

    for idx, data in enumerate(train_dataloader):
        print(idx)
        if idx > 5000:
            break
        # examplar_img = data['examplar_img'].cuda()
        # search_img = data['search_img'].cuda()
        # # print(examplar_img[0,0,0,0])
        # # print(examplar_img[0,1,0,0])
        # # print(examplar_img[0,2,0,0])
        # bbox = data['bbox'].cpu().numpy()
        # gt_cls = data['gt_cls'].cuda()
        # gt_delta = data['gt_delta'].cuda()
        # gt_delta_weight = data['delta_weight'].cuda()
        #
        # np.set_printoptions(threshold=np.inf)
        # print(examplar_img.detach().cpu().numpy()[0,0,:,:],)
        # examplar_img=torch.floor(examplar_img)
        # search_img=torch.floor(search_img)
        # print(examplar_img[0,0,0,0])
        # print(examplar_img[0,1,0,0])
        # print(examplar_img[0,2,0,0])
        # losses = base_model.forward(examplar_img, search_img, gt_cls, gt_delta, gt_delta_weight)

        # examplar_img=cv2.imread('../quant_dataset/examplar/examplar_1.png')
        examplar_img=np.ones((127,127,3))*255
        examplar_img=torch.from_numpy(examplar_img.astype(np.float32))[None,:,:,:].permute(0,3,1,2).cuda()
        base_model.forward(examplar_img)

        # examplar_img = examplar_img.cpu().numpy()
        # examplar_img = examplar_img.squeeze(0).transpose((1, 2, 0))
        # examplar_path = os.path.join(quan_dataset_dir, 'examplar', 'examplar_{}.png'.format(idx + 1))
        # cv2.imwrite(examplar_path, examplar_img)
        #
        # search_img = search_img.cpu().numpy()
        # search_img = search_img.squeeze(0).transpose((1, 2, 0))
        # search_path = os.path.join(quan_dataset_dir, 'search', 'search_{}.png'.format(idx + 1))
        # cv2.imwrite(search_path, search_img)
        #
        # gt_cls = gt_cls.cpu().numpy().squeeze(0).astype(np.float32)
        # np.save(os.path.join(quan_dataset_dir, 'gt_cls', 'gt_cls_{}.npy'.format(idx + 1)), gt_cls)
        # gt_delta = gt_delta.cpu().numpy().squeeze(0)
        # gt_delta = gt_delta.reshape(-1, 25, 25)
        # np.save(os.path.join(quan_dataset_dir, 'gt_delta', 'gt_delta_{}.npy'.format(idx + 1)), gt_delta)
        # gt_delta_weight = gt_delta_weight.cpu().numpy().squeeze(0)
        # np.save(os.path.join(quan_dataset_dir, 'gt_delta_weight', 'gt_delta_weight_{}.npy'.format(idx + 1)),
        #         gt_delta_weight)
