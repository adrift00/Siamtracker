from models.backbone.alexnet import alexnet
from models.backbone.mobilenet_v2 import mobilenet_v2
from models.backbone.resnet import resnet50

BACKBONES = {
    'alexnet': alexnet,
    'mobilenetv2': mobilenet_v2,
    'resnet50': resnet50
}


def get_backbone(name, **kwargs):
    return BACKBONES[name](**kwargs)
