from yacs.config import CfgNode

cfg = CfgNode(new_allowed=True)

cfg.MODEL_ARC = 'BaseSiamModel'

cfg.DATASET = CfgNode()
cfg.DATASET.NAMES = ['VID','DET','COCO','YOUTUBEBB']
cfg.DATASET.COCO = CfgNode()
cfg.DATASET.COCO.DATA_DIR = '../pysot/training_dataset/coco/crop511'
cfg.DATASET.COCO.ANNO_FILE = '../pysot/training_dataset/coco/train2017.json'
cfg.DATASET.COCO.FRAME_RANGE = 1

cfg.DATASET.COCO.NUM_USE = -1

cfg.DATASET.DET = CfgNode()
cfg.DATASET.DET.DATA_DIR = '../pysot/training_dataset/det/crop511'
cfg.DATASET.DET.ANNO_FILE = '../pysot/training_dataset/det/train.json'
cfg.DATASET.DET.FRAME_RANGE = 1
cfg.DATASET.DET.NUM_USE = -1

cfg.DATASET.VID = CfgNode()
cfg.DATASET.VID.DATA_DIR = '../pysot/training_dataset/vid/crop511'
cfg.DATASET.VID.ANNO_FILE = '../pysot/training_dataset/vid/train.json'
cfg.DATASET.VID.FRAME_RANGE = 100
cfg.DATASET.VID.NUM_USE = 100000

cfg.DATASET.YOUTUBEBB = CfgNode()
cfg.DATASET.YOUTUBEBB.DATA_DIR = '../pysot/training_dataset/yt_bb/youtube/crop511'
cfg.DATASET.YOUTUBEBB.ANNO_FILE = '../pysot/training_dataset/yt_bb/youtube/train.json'
cfg.DATASET.YOUTUBEBB.FRAME_RANGE = 3
cfg.DATASET.YOUTUBEBB.NUM_USE = -1


cfg.DATASET.NEG = 0.05
cfg.DATASET.GRAY = 0.0

cfg.DATASET.EXAMPLAR = CfgNode()
# Random shift see [SiamPRN++](https://arxiv.org/pdf/1812.11703)
# for detail discussion
cfg.DATASET.EXAMPLAR.SHIFT = 4
cfg.DATASET.EXAMPLAR.SCALE = 0.05
cfg.DATASET.EXAMPLAR.BLUR = 0.0
cfg.DATASET.EXAMPLAR.FLIP = 0.0
cfg.DATASET.EXAMPLAR.COLOR = 1.0

cfg.DATASET.SEARCH = CfgNode()
cfg.DATASET.SEARCH.SHIFT = 64
cfg.DATASET.SEARCH.SCALE = 0.18
cfg.DATASET.SEARCH.BLUR = 0.2
cfg.DATASET.SEARCH.FLIP = 0.0
cfg.DATASET.SEARCH.COLOR = 1.0

cfg.DATASET.VIDEO_PER_EPOCH = 600000

cfg.ANCHOR = CfgNode()
cfg.ANCHOR.RATIOS = [0.33, 0.5, 1, 2, 3]
cfg.ANCHOR.SCALES = [8]
cfg.ANCHOR.STRIDE = 8

cfg.BACKBONE = CfgNode()
cfg.BACKBONE.TYPE = 'alexnet'
cfg.BACKBONE.TRAIN_LAYERS = ['layer4', 'layer5']
cfg.BACKBONE.TRAIN_EPOCH = 10
cfg.BACKBONE.LAYERS_LR = 1.0
cfg.BACKBONE.KWARGS = CfgNode(new_allowed=True)
cfg.BACKBONE.KWARGS.width_mult = 1.0

cfg.ADJUST = CfgNode()
cfg.ADJUST.USE = False
cfg.ADJUST.TYPE = "AdjustAllLayer"
cfg.ADJUST.KWARGS = CfgNode(new_allowed=True)

cfg.RPN = CfgNode()
cfg.RPN.TYPE = 'DepthwiseRPN'
cfg.RPN.KWARGS = CfgNode(new_allowed=True)

cfg.MASK = CfgNode()
# Whether to use mask generate segmentation
cfg.MASK.USE = False

cfg.TRAIN = CfgNode()
cfg.TRAIN.THRESH_HIGH = 0.6
cfg.TRAIN.THRESH_LOW = 0.3
cfg.TRAIN.TOTAL_NUM = 64
cfg.TRAIN.POS_NUM = 16
cfg.TRAIN.NEG_NUM = 16
cfg.TRAIN.EXAMPLER_SIZE = 127
cfg.TRAIN.SEARCH_SIZE = 255
cfg.TRAIN.BASE_SIZE = 0
cfg.TRAIN.OUTPUT_SIZE = 17
cfg.TRAIN.BATCH_SIZE = 128
cfg.TRAIN.NUM_WORKERS = 1

cfg.TRAIN.BASE_LR = 0.005
cfg.TRAIN.MOMENTUM = 0.9
cfg.TRAIN.WEIGHT_DECAY = 0.0001
cfg.TRAIN.CLS_WEIGHT = 1.0
cfg.TRAIN.LOC_WEIGHT = 1.2

cfg.TRAIN.GRAD_CLIP = 10.0
cfg.TRAIN.RESUME = False
cfg.TRAIN.RESUME_PATH = ''
cfg.TRAIN.PRETRAIN = False
cfg.TRAIN.PRETRAIN_PATH = ''
cfg.TRAIN.BACKBONE_PRETRAIN = True
cfg.TRAIN.BACKBONE_PATH = './pretrained_models/alexnet-bn.pth'
cfg.TRAIN.SNAPSHOT_DIR = './snapshot'
cfg.TRAIN.EPOCHS = 50
cfg.TRAIN.START_EPOCH = 0
cfg.TRAIN.PRINT_EVERY = 20
cfg.TRAIN.LOG_DIR = './logs'
cfg.TRAIN.LOG_GRAD = False

cfg.TRAIN.LR = CfgNode()
cfg.TRAIN.LR.TYPE = 'log'
cfg.TRAIN.LR.KWARGS = CfgNode()
cfg.TRAIN.LR.KWARGS.start_lr = 0.01
cfg.TRAIN.LR.KWARGS.end_lr = 0.0005
cfg.TRAIN.LR_WARMUP = CfgNode()
cfg.TRAIN.LR_WARMUP.WARMUP = True
cfg.TRAIN.LR_WARMUP.TYPE = 'step'
cfg.TRAIN.LR_WARMUP.EPOCH = 5
cfg.TRAIN.LR_WARMUP.KWARGS = CfgNode(new_allowed=True)
cfg.TRAIN.LR_WARMUP.KWARGS.start_lr = 0.005
cfg.TRAIN.LR_WARMUP.KWARGS.end_lr = 0.01
cfg.TRAIN.LR_WARMUP.KWARGS.step = 1

# track
cfg.TRACK = CfgNode()
cfg.TRACK.TYPE = 'SiamRPNTracker'
cfg.TRACK.DATA_DIR = '../pysot/testing_dataset/'
cfg.TRACK.RESULT_DIR = './result'
cfg.TRACK.EXAMPLAR_SIZE = 127
cfg.TRACK.INSTANCE_SIZE = 287
cfg.TRACK.BASE_SIZE = 0
cfg.TRACK.PENALTY_K = 0.16
cfg.TRACK.WINDOW_INFLUENCE = 0.40
cfg.TRACK.LR = 0.3



