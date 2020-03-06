from collections import OrderedDict

import torch
from torch import nn

from configs.config import cfg
from models.base_siam_model import BaseSiamModel
from utils.loss import select_cross_entropy_loss, weight_l1_loss


class MetaSiamModel(BaseSiamModel):
    # meta track
    def set_examplar(self, examplars, searches, gt_cls, gt_loc, gt_loc_weight):
        self.examplar = self.backbone(examplars[0][None, ...])
        self.weight = self.meta_train(examplars, searches, gt_cls, gt_loc, gt_loc_weight)

    def track(self, search):
        search = self.backbone(search)
        examplar = self.examplar
        if cfg.ADJUST.USE:
            examplar = self.neck(self.examplar)
            search = self.neck(search)
        pred_cls, pred_loc = self.rpn(examplar, search, self.weight, self.bn_weight)
        return pred_cls, pred_loc

    # meta train
    def _fix_module(self, module):
        for param in module.parameters():
            param.requires_grad = False
        for m in module.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def meta_train_init(self):
        self._fix_module(self.backbone)
        if cfg.ADJUST.USE:
            self._fix_module(self.neck)
        self._fix_module(self.rpn)
        self.init_weight, self.bn_weight = self.get_rpn_state()
        self.alpha = OrderedDict()
        for k, v in self.init_weight.items():
            a = v.clone().detach().requires_grad_(True)
            a.data.fill_(cfg.META.INIT_ALPHA)
            self.alpha[k] = a

    def meta_train(self, examplar, search, gt_cls, gt_loc, gt_loc_weight):
        examplar = self.backbone(examplar)
        search = self.backbone(search)
        if cfg.ADJUST.USE:
            examplar = self.neck(examplar)
            search = self.neck(search)
        # first iter
        pred_cls, pred_loc = self.rpn(examplar, search, self.init_weight, self.bn_weight)
        pred_cls = self.log_softmax(pred_cls)
        cls_loss = select_cross_entropy_loss(pred_cls, gt_cls)
        loc_loss = weight_l1_loss(pred_loc, gt_loc, gt_loc_weight)
        total_loss = cfg.TRAIN.CLS_WEIGHT * cls_loss + cfg.TRAIN.LOC_WEIGHT * loc_loss

        grads = torch.autograd.grad(total_loss, self.init_weight.values(), retain_graph=True, create_graph=True)
        new_init_weight = OrderedDict((k, iw-a*g)
                                      for (k, iw), a, g in zip(self.init_weight.items(), self.alpha.values(), grads))
        # second iter
        pred_cls, pred_loc = self.rpn(examplar, search, new_init_weight, self.bn_weight)
        pred_cls = self.log_softmax(pred_cls)
        cls_loss = select_cross_entropy_loss(pred_cls, gt_cls)
        loc_loss = weight_l1_loss(pred_loc, gt_loc, gt_loc_weight)
        total_loss = cfg.TRAIN.CLS_WEIGHT * cls_loss + cfg.TRAIN.LOC_WEIGHT * loc_loss
        grads = torch.autograd.grad(total_loss, new_init_weight.values(), create_graph=True)
        new_init_weight = OrderedDict((k, iw-a*g)
                                      for (k, iw), a, g in zip(new_init_weight.items(), self.alpha.values(), grads))
        return new_init_weight

    def meta_eval(self, new_init_weight, examplar, search, gt_cls, gt_loc, gt_loc_weight):
        examplar = self.backbone(examplar)
        search = self.backbone(search)
        if cfg.ADJUST.USE:
            examplar = self.neck(examplar)
            search = self.neck(search)
        pred_cls, pred_loc = self.rpn(examplar, search, new_init_weight, self.bn_weight)
        pred_cls = self.log_softmax(pred_cls)
        cls_loss = select_cross_entropy_loss(pred_cls, gt_cls)
        loc_loss = weight_l1_loss(pred_loc, gt_loc, gt_loc_weight)
        total_loss = cfg.TRAIN.CLS_WEIGHT * cls_loss + cfg.TRAIN.LOC_WEIGHT * loc_loss
        # compute the loss of init examplar
        init_grad_vals = torch.autograd.grad(total_loss, self.init_weight.values(), retain_graph=True)
        alpha_grad_vals = torch.autograd.grad(total_loss, self.alpha.values(), retain_graph=True)
        # generate ordered dict
        init_grads = OrderedDict((k, g) for k, g in zip(self.init_weight.keys(), init_grad_vals))
        alpha_grads = OrderedDict((k, g) for k, g in zip(self.alpha.keys(), alpha_grad_vals))
        return init_grads, alpha_grads, total_loss

    def get_rpn_state(self):
        weight = OrderedDict()
        bn_weight = OrderedDict()
        for k, v in self.rpn.state_dict().items():
            if k.split('.')[-1].startswith('num'):  # skip bn running num
                continue

            if k.split('.')[-1].startswith('running'):  # skip bn mean and var
                bn_weight[k] = v.clone().detach()
                continue
            weight[k] = v.clone().detach().requires_grad_(True)
        return weight, bn_weight

    def set_rpn_state(self):
        for k, v in self.rpn.state_dict().items():
            if k.split('.')[-1].startswith('num'):  # skip bn running num
                continue
            if k.split('.')[-1].startswith('running'):  # skip bn mean and var
                continue
            v.data.copy_(self.init_weight[k])


