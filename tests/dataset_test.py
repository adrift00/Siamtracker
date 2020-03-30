from torch.utils.data import DataLoader

from dataset.dataset import TrainDataset
import random
import os
import numpy as np
import torch

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
    train_dataset = TrainDataset()
    train_dataloader = DataLoader(train_dataset, batch_size=1)
    # train_dataloader.dataset.shuffle()
    for i, data in enumerate(train_dataloader):
        print(i)


