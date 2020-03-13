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
            nn.Tanh()
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
        # don't freeze the rpn, finetune it
        # freeze_module(self.rpn)

    def forward(self, examplar, search, gt_cls, gt_loc, gt_loc_weight):
        examplar = self.backbone(examplar)
        search = self.backbone(search)
        if cfg.ADJUST.USE:
            examplar = self.neck(examplar)
            search = self.neck(search)
        # # use the first examplar to update the frame
        # examplar0 = examplar[0, :, :, :][None, :, :, :]
        # # examplar0.requires_grad_(True)
        # examplar0.requires_grad_(True)
        # init_cls_losses = torch.zeros([examplar.size(0)]).cuda()
        # init_loc_losses = torch.zeros([examplar.size(0)]).cuda()
        # init_total_losses = torch.zeros([examplar.size(0)]).cuda()
        # for i in range(examplar.size(0)):
        #     pred_cls, pred_loc = self.rpn(examplar0, search[i][None, :, :, :])
        #     pred_cls = self.log_softmax(pred_cls)
        #     cls_loss = select_cross_entropy_loss(pred_cls, gt_cls[i][None, :, :, :])
        #     loc_loss = weight_l1_loss(pred_loc, gt_loc[i][None, :, :, :], gt_loc_weight[i][None, :, :, :])
        #     total_loss = cfg.TRAIN.CLS_WEIGHT * cls_loss + cfg.TRAIN.LOC_WEIGHT * loc_loss
        #     init_cls_losses[i], init_loc_losses[i], init_total_losses[i] = cls_loss, loc_loss, total_loss
        # # backward for the grad
        # init_cls_losses, init_loc_losses, init_total_losses = init_cls_losses.mean(), init_loc_losses.mean(), init_total_losses.mean()
        # examplar_grad = torch.autograd.grad(init_total_losses, examplar0)[0] * 1000
        # examplar0 = examplar0 + self.grad_layer(examplar_grad)
        # # use the new examplar to get the final loss
        # cls_losses = torch.zeros([examplar.size(0)]).cuda()
        # loc_losses = torch.zeros([examplar.size(0)]).cuda()
        # total_losses = torch.zeros([examplar.size(0)]).cuda()
        # for i in range(examplar.size(0)):
        #     pred_cls, pred_loc = self.rpn(examplar0, search[i][None, :, :, :])
        #     pred_cls = self.log_softmax(pred_cls)
        #     cls_loss = select_cross_entropy_loss(pred_cls, gt_cls[i][None, :, :, :])
        #     loc_loss = weight_l1_loss(pred_loc, gt_loc[i][None, :, :, :], gt_loc_weight[i][None, :, :, :])
        #     total_loss = cfg.TRAIN.CLS_WEIGHT * cls_loss + cfg.TRAIN.LOC_WEIGHT * loc_loss
        #     cls_losses[i], loc_losses[i], total_losses[i] = cls_loss, loc_loss, total_loss
        # cls_losses, loc_losses, total_losses = cls_losses.mean(), loc_losses.mean(), total_losses.mean()
        # return {
        #     'cls_loss': cls_losses,
        #     'loc_loss': loc_losses,
        #     'total_loss': total_losses,
        #     'init_cls_loss': init_cls_losses,
        #     'init_loc_loss': init_loc_losses,
        #     'init_total_loss': init_total_losses,
        #     'examplar_grad': examplar_grad
        # }
        examplar.requires_grad_(True)
        pred_cls, pred_loc = self.rpn(examplar, search)
        pred_cls = self.log_softmax(pred_cls)
        init_cls_loss = select_cross_entropy_loss(pred_cls, gt_cls)
        init_loc_loss = weight_l1_loss(pred_loc, gt_loc, gt_loc_weight)
        init_total_loss = cfg.TRAIN.CLS_WEIGHT * init_cls_loss + cfg.TRAIN.LOC_WEIGHT * init_loc_loss
        examplar_grad = torch.autograd.grad(init_total_loss, examplar)[0] * 1000
        examplar = examplar + self.grad_layer(examplar_grad)
        pred_cls, pred_loc = self.rpn(examplar, search)
        pred_cls = self.log_softmax(pred_cls)
        cls_loss = select_cross_entropy_loss(pred_cls, gt_cls)
        loc_loss = weight_l1_loss(pred_loc, gt_loc, gt_loc_weight)
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
        self.examplar.requires_grad_(True)
        pred_cls, pred_loc = self.rpn(self.examplar, search)
        pred_cls = self.log_softmax(pred_cls)
        cls_loss = select_cross_entropy_loss(pred_cls, gt_cls)
        loc_loss = weight_l1_loss(pred_loc, gt_loc, gt_loc_weight)
        total_loss = cfg.TRAIN.CLS_WEIGHT * cls_loss + cfg.TRAIN.LOC_WEIGHT * loc_loss
        # backward for the grad
        # examplar_grad = torch.autograd.grad(total_loss, self.examplar)[0] * 1000
        examplar_grad = torch.autograd.grad(total_loss, self.examplar)[0] * 1000
        self.examplar = self.examplar + self.grad_layer(examplar_grad)
