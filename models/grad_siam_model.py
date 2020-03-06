import torch
import torch.nn as nn

from configs.config import cfg
from models.base_siam_model import BaseSiamModel
from utils.loss import select_cross_entropy_loss, weight_l1_loss


class GradSiamModel(BaseSiamModel):
    def __init__(self):
        super().__init__()
        self.grad_layer= nn.Sequential(
            nn.Conv2d(256,256,3,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256,256,3,1,1),
            nn.ReLU()
        )

    def freeze_model(self):
        def freeze_module(module):
            for param in module.parameters():
                param.requires_grad = False
            for m in module.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        # freeze the backbone,neck and rpn, only the grad_layer can be trained.
        freeze_module(self.backbone)
        if cfg.ADJUST.USE:
            freeze_module(self.neck)
        freeze_module(self.rpn)


    def forward(self, examplar, search, gt_cls, gt_loc, gt_loc_weight):
        examplar = self.backbone(examplar)
        search = self.backbone(search)
        if cfg.ADJUST.USE:
            examplar = self.neck(examplar)
            search = self.neck(search)
        pred_cls, pred_loc = self.rpn(examplar, search)
        pred_cls = self.log_softmax(pred_cls)
        cls_loss = select_cross_entropy_loss(pred_cls, gt_cls)
        loc_loss = weight_l1_loss(pred_loc, gt_loc, gt_loc_weight)
        total_loss = cfg.TRAIN.CLS_WEIGHT * cls_loss + cfg.TRAIN.LOC_WEIGHT * loc_loss
        # backward for the grad
        examplar.requires_grad_(True)
        examplar_grads=torch.autograd.grad(total_loss,self.examplar)
        examplar=examplar+self.grad_layer(examplar_grads)
        # use the new examplar to get the final loss
        examplar.requires_grad_(False)
        pred_cls, pred_loc = self.rpn(examplar, search)
        pred_cls = self.log_softmax(pred_cls)
        cls_loss = select_cross_entropy_loss(pred_cls, gt_cls)
        loc_loss = weight_l1_loss(pred_loc, gt_loc, gt_loc_weight)
        total_loss = cfg.TRAIN.CLS_WEIGHT * cls_loss + cfg.TRAIN.LOC_WEIGHT * loc_loss
        return {
            'cls_loss': cls_loss,
            'loc_loss': loc_loss,
            'total_loss': total_loss
        }

    def set_examplar(self, examplar,search,gt_cls, gt_loc, gt_loc_weight):
        examplar = self.backbone(examplar)
        search = self.backbone(search)
        if cfg.ADJUST.USE:
            examplar = self.neck(examplar)
            search = self.neck(search)
        self.examplar=examplar
        pred_cls, pred_loc = self.rpn(self.examplar, search)
        pred_cls = self.log_softmax(pred_cls)
        cls_loss = select_cross_entropy_loss(pred_cls, gt_cls)
        loc_loss = weight_l1_loss(pred_loc, gt_loc, gt_loc_weight)
        total_loss = cfg.TRAIN.CLS_WEIGHT * cls_loss + cfg.TRAIN.LOC_WEIGHT * loc_loss
        # backward for the grad
        self.examplar.requires_grad_(True)
        examplar_grads=torch.autograd.grad(total_loss,self.examplar)
        self.examplar=self.examplar+self.grad_layer(examplar_grads)








