import torch
import torch.nn as nn

from configs.config import cfg
from models.base_siam_model import BaseSiamModel
from utils.loss import select_cross_entropy_loss, weight_l1_loss


class GradSiamModel(BaseSiamModel):
    def __init__(self):
        super().__init__()
        self.grad_layer = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.Sigmoid()
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

    def forward(self,
                examplar,
                train_search, train_gt_cls, train_gt_loc, train_gt_loc_weight,
                test_search, test_gt_cls, test_gt_loc, test_gt_loc_weight):
        examplar = self.backbone(examplar)
        search = self.backbone(train_search)
        test_search = self.backbone(test_search)
        if cfg.ADJUST.USE:
            examplar = self.neck(examplar)
            search = self.neck(search)
            test_search = self.neck(test_search)

        # examplar.requires_grad_(True)
        # pred_cls, pred_loc = self.rpn(examplar, search)
        # pred_cls = self.log_softmax(pred_cls)
        # init_cls_loss = select_cross_entropy_loss(pred_cls, train_gt_cls)
        # init_loc_loss = weight_l1_loss(pred_loc, train_gt_loc, train_gt_loc_weight)
        # init_total_loss = cfg.TRAIN.CLS_WEIGHT * init_cls_loss + cfg.TRAIN.LOC_WEIGHT * init_loc_loss
        # examplar_grad = torch.autograd.grad(init_total_loss, examplar)[0] * 1000
        # examplar = examplar + self.grad_layer(examplar_grad)
        #
        # # for test search
        # pred_cls, pred_loc = self.rpn(examplar, test_search)
        # pred_cls = self.log_softmax(pred_cls)
        # cls_loss = select_cross_entropy_loss(pred_cls, test_gt_cls)
        # loc_loss = weight_l1_loss(pred_loc, test_gt_loc, test_gt_loc_weight)
        # total_loss = cfg.TRAIN.CLS_WEIGHT * cls_loss + cfg.TRAIN.LOC_WEIGHT * loc_loss

        # return {
        #     'cls_loss': cls_loss,
        #     'loc_loss': loc_loss,
        #     'total_loss': total_loss,
        #     'init_cls_loss': init_cls_loss,
        #     'init_loc_loss': init_loc_loss,
        #     'init_total_loss': init_total_loss,
        #     'examplar_grad': examplar_grad
        # }

        # examplar0 = examplar[0, :, :, :][None, :, :, :]
        # new_examplar = examplar0.repeat(examplar.size(0), 1, 1, 1)
        # new_examplar.requires_grad_(True)
        # pred_cls, pred_loc = self.rpn(new_examplar, search)
        # pred_cls = self.log_softmax(pred_cls)
        # init_cls_loss = select_cross_entropy_loss(pred_cls, train_gt_cls)
        # init_loc_loss = weight_l1_loss(pred_loc, train_gt_loc, train_gt_loc_weight)
        # init_total_loss = cfg.TRAIN.CLS_WEIGHT * init_cls_loss + cfg.TRAIN.LOC_WEIGHT * init_loc_loss
        # examplar_grad = torch.autograd.grad(init_total_loss, new_examplar)[0] * 1000
        # new_examplar = new_examplar + self.grad_layer(examplar_grad)
        #
        # pred_cls, pred_loc = self.rpn(new_examplar, test_search)
        # pred_cls = self.log_softmax(pred_cls)
        # cls_loss = select_cross_entropy_loss(pred_cls, test_gt_cls)
        # loc_loss = weight_l1_loss(pred_loc, test_gt_loc, test_gt_loc_weight)
        # total_loss = cfg.TRAIN.CLS_WEIGHT * cls_loss + cfg.TRAIN.LOC_WEIGHT * loc_loss
        # return {
        #     'cls_loss': cls_loss,
        #     'loc_loss': loc_loss,
        #     'total_loss': total_loss,
        #     'init_cls_loss': init_cls_loss,
        #     'init_loc_loss': init_loc_loss,
        #     'init_total_loss': init_total_loss,
        #     'examplar_grad': examplar_grad
        # }
        loc_examplar=examplar.detach()
        # examplar0 = examplar[0, :, :, :][None, :, :, :]
        # new_examplar = examplar0.repeat(examplar.size(0), 1, 1, 1)
        new_examplar=examplar
        new_examplar.requires_grad_(True)
        pred_cls, pred_loc = self.rpn(new_examplar, search)
        pred_cls = self.log_softmax(pred_cls)
        init_cls_loss = select_cross_entropy_loss(pred_cls, train_gt_cls)
        init_loc_loss = weight_l1_loss(pred_loc, train_gt_loc, train_gt_loc_weight)
        init_total_loss = cfg.TRAIN.CLS_WEIGHT * init_cls_loss + cfg.TRAIN.LOC_WEIGHT * init_loc_loss
        examplar_grad = torch.autograd.grad(init_cls_loss, new_examplar)[0] * 1000
        new_examplar = new_examplar + self.grad_layer(examplar_grad)

        pred_cls, _ = self.rpn(new_examplar, test_search)
        pred_cls = self.log_softmax(pred_cls)
        cls_loss = select_cross_entropy_loss(pred_cls, test_gt_cls)

        _,pred_loc=self.rpn(loc_examplar,test_search)
        loc_loss = weight_l1_loss(pred_loc, test_gt_loc, test_gt_loc_weight)
        total_loss = cfg.TRAIN.CLS_WEIGHT * cls_loss + cfg.TRAIN.LOC_WEIGHT * loc_loss
        return {
            'cls_loss': cls_loss,
            'loc_loss': loc_loss,
            'total_loss': total_loss,
            'init_cls_loss': init_cls_loss,
            'init_loc_loss': init_loc_loss,
            'init_total_loss': init_total_loss,
            'examplar_grad': examplar_grad
        }

    def set_examplar(self, examplar, search, gt_cls, gt_loc, gt_loc_weight):
        examplar = self.backbone(examplar)
        search = self.backbone(search)
        if cfg.ADJUST.USE:
            examplar = self.neck(examplar)
            search = self.neck(search)
        self.examplar = examplar
        self.loc_examplar=examplar.detach()
        self.examplar.requires_grad_(True)

        pred_cls, pred_loc = self.rpn(self.examplar, search)
        pred_cls = self.log_softmax(pred_cls)
        cls_loss = select_cross_entropy_loss(pred_cls, gt_cls)
        loc_loss = weight_l1_loss(pred_loc, gt_loc, gt_loc_weight)
        total_loss = cfg.TRAIN.CLS_WEIGHT * cls_loss + cfg.TRAIN.LOC_WEIGHT * loc_loss
        # backward for the grad
        examplar_grad = torch.autograd.grad(cls_loss, self.examplar)[0] * 1000
        self.examplar = self.examplar + self.grad_layer(examplar_grad)


    def track(self, search):
        search = self.backbone(search)
        examplar = self.examplar
        if cfg.ADJUST.USE:
            search = self.neck(search)
        pred_cls, _ = self.rpn(examplar, search)
        _,pred_loc=self.rpn(self.loc_examplar,search)

        return pred_cls, pred_loc




