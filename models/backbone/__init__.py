from models.backbone.alex_net import alexnet,AlexNet
from models.backbone.mobilenet_v2 import mobilenet_v2,MobileNetV2

BACKBONES = {
    'alexnet': AlexNet,
    'mobilenetv2': MobileNetV2
}


def get_backbone(name, **kwargs):
    return BACKBONES[name](**kwargs)
