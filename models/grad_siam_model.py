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

        # freeze the backbone,neck, only the grad_layer can be trained.
        freeze_module(self.backbone)
        if cfg.ADJUST.USE:
            freeze_module(self.neck)

    def forward(self, examplars,
                train_searchs, train_gt_cls, train_gt_loc, train_gt_loc_weight,
                test_searchs, test_gt_cls, test_gt_loc, test_gt_loc_weight):
        batch_cls_loss = torch.zeros([examplars.size(0)]).cuda()
        batch_loc_loss = torch.zeros([examplars.size(0)]).cuda()
        batch_total_loss = torch.zeros([examplars.size(0)]).cuda()
        for idx in range(examplars.size(0)):
            examplar = self.backbone(examplars[idx][None, :, :, :])
            train_search = self.backbone(train_searchs[idx])
            test_search = self.backbone(test_searchs[idx])
            if cfg.ADJUST.USE:
                examplar = self.neck(examplar)
                train_search = self.neck(train_search)
                test_search = self.neck(test_search)
            examplar.requires_grad_(True)
            init_cls_losses = torch.zeros([train_search.size(0)]).cuda()
            init_loc_losses = torch.zeros([train_search.size(0)]).cuda()
            init_total_losses = torch.zeros([train_search.size(0)]).cuda()
            for i in range(train_search.size(0)):
                pred_cls, pred_loc = self.rpn(examplar, train_search[i][None, :, :, :])
                pred_cls = self.log_softmax(pred_cls)
                cls_loss = select_cross_entropy_loss(pred_cls, train_gt_cls[idx][i][None, :, :, :])
                loc_loss = weight_l1_loss(pred_loc, train_gt_loc[idx][i][None, :, :, :],
                                          train_gt_loc_weight[idx][i][None, :, :, :])
                total_loss = cfg.TRAIN.CLS_WEIGHT * cls_loss + cfg.TRAIN.LOC_WEIGHT * loc_loss
                init_cls_losses[i], init_loc_losses[i], init_total_losses[i] = cls_loss, loc_loss, total_loss
            # backward for the grad
            init_cls_losses, init_loc_losses, init_total_losses = init_cls_losses.mean(), init_loc_losses.mean(), init_total_losses.mean()
            examplar_grad = torch.autograd.grad(init_total_losses, examplar)[0] * 1000
            examplar = examplar + self.grad_layer(examplar_grad)
            # use the new examplar to get the final loss
            cls_losses = torch.zeros([test_search.size(0)]).cuda()
            loc_losses = torch.zeros([test_search.size(0)]).cuda()
            total_losses = torch.zeros([test_search.size(0)]).cuda()
            for i in range(test_search.size(0)):
                pred_cls, pred_loc = self.rpn(examplar, test_search[i][None, :, :, :])
                pred_cls = self.log_softmax(pred_cls)
                cls_loss = select_cross_entropy_loss(pred_cls, test_gt_cls[idx][i][None, :, :, :])
                loc_loss = weight_l1_loss(pred_loc, test_gt_loc[idx][i][None, :, :, :],
                                          test_gt_loc_weight[idx][i][None, :, :, :])
                total_loss = cfg.TRAIN.CLS_WEIGHT * cls_loss + cfg.TRAIN.LOC_WEIGHT * loc_loss
                cls_losses[i], loc_losses[i], total_losses[i] = cls_loss, loc_loss, total_loss
            cls_losses, loc_losses, total_losses = cls_losses.mean(), loc_losses.mean(), total_losses.mean()
            batch_cls_loss[idx], batch_loc_loss[idx], batch_total_loss[idx] = cls_losses, loc_losses, total_losses
        return {
            "cls_loss": batch_cls_loss.mean(),
            'loc_loss': batch_loc_loss.mean(),
            'total_loss': batch_total_loss.mean()
        }

    def set_examplar(self, examplar, searches, gt_cls, gt_loc, gt_loc_weight):
        examplar = self.backbone(examplar)
        init_cls_losses = torch.zeros([searches.size(0)]).cuda()
        init_loc_losses = torch.zeros([searches.size(0)]).cuda()
        init_total_losses = torch.zeros([searches.size(0)]).cuda()
        for i in range(searches.size(0)):
            search = self.backbone(searches[i][None, :, :, :])
            if cfg.ADJUST.USE:
                examplar = self.neck(examplar)
                search = self.neck(search)
            pred_cls, pred_loc = self.rpn(examplar, search)
            pred_cls = self.log_softmax(pred_cls)
            cls_loss = select_cross_entropy_loss(pred_cls, gt_cls[i][None, :, :, :])
            loc_loss = weight_l1_loss(pred_loc, gt_loc[i][None, :, :, :], gt_loc_weight[i][None, :, :, :])
            total_loss = cfg.TRAIN.CLS_WEIGHT * cls_loss + cfg.TRAIN.LOC_WEIGHT * loc_loss
            init_cls_losses[i], init_loc_losses[i], init_total_losses[i] = cls_loss, loc_loss, total_loss
        # backward for the grad
        init_cls_losses, init_loc_losses, init_total_losses = init_cls_losses.mean(), init_loc_losses.mean(), init_total_losses.mean()
        examplar_grad = torch.autograd.grad(init_total_losses, examplar)[0] * 1000
        self.examplar = examplar + self.grad_layer(examplar_grad)
