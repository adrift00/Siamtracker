import torch
from torch import nn
import torch.nn.functional as F

from configs.config import cfg
from models.backbone import get_backbone
from models.head import get_rpn_head
from models.neck import get_neck
from utils.loss import select_cross_entropy_loss, weight_l1_loss


class BaseSiamModel(nn.Module):
    def __init__(self):
        super(BaseSiamModel, self).__init__()
        self.backbone = get_backbone(cfg.BACKBONE.TYPE, **cfg.BACKBONE.KWARGS)
        if cfg.ADJUST.USE:
            self.neck = get_neck(cfg.ADJUST.TYPE, **cfg.ADJUST.KWARGS)

        self.rpn = get_rpn_head(cfg.RPN.TYPE, **cfg.RPN.KWARGS)

    # def forward(self, examplar, search, gt_cls, gt_loc, gt_loc_weight):
    #     examplar = self.backbone(examplar)
    #     search = self.backbone(search)
    #     if cfg.ADJUST.USE:
    #         examplar = self.neck(examplar)
    #         search = self.neck(search)
    #     pred_cls, pred_loc = self.rpn(examplar, search)
    #     pred_cls = self.log_softmax(pred_cls)
    #     cls_loss = select_cross_entropy_loss(pred_cls, gt_cls)
    #     loc_loss = weight_l1_loss(pred_loc, gt_loc, gt_loc_weight)
    #     total_loss = cfg.TRAIN.CLS_WEIGHT * cls_loss + cfg.TRAIN.LOC_WEIGHT * loc_loss
    #     return {
    #         'cls_loss': cls_loss,
    #         'loc_loss': loc_loss,
    #         'total_loss': total_loss
    #     }

    def track(self, search):
        search = self.backbone(search)
        examplar = self.examplar
        if cfg.ADJUST.USE:
            search = self.neck(search)
        pred_cls, pred_loc = self.rpn(examplar, search)
        return pred_cls, pred_loc

    def set_examplar(self, examplar):
        examplar = self.backbone(examplar)
        if cfg.ADJUST.USE:
            examplar=self.neck(examplar)
        self.examplar=examplar



    # for model conveter
    # @torch.no_grad()
    # def forward(self, examplar):
    #     examplar = self.backbone(examplar)
    #     if cfg.ADJUST.USE:
    #         examplar=self.neck(examplar)
    #     return examplar[0],examplar[1],examplar[2]

    # def get_examplar(self, examplar):
    #     examplar = self.backbone(examplar)
    #     if cfg.ADJUST.USE:
    #         examplar=self.neck(examplar)
    #     return examplar[0],examplar[1],examplar[2]
    #
    # @torch.no_grad()
    # def forward(self, e0,e1,e2, search):
    #     examplar=[e0,e1,e2]
    #     search = self.backbone(search)
    #     if cfg.ADJUST.USE:
    #         search = self.neck(search)
    #     pred_cls, pred_loc = self.rpn(examplar, search)
    #     return pred_cls, pred_loc

    @torch.no_grad()
    def forward(self,examplar,search):
        examplar = self.backbone(examplar)
        search = self.backbone(search)
        if cfg.ADJUST.USE:
            examplar = self.neck(examplar)
            search = self.neck(search)
        pred_cls, pred_loc = self.rpn(examplar, search)
        return pred_cls,pred_loc


    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2 // 2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls
