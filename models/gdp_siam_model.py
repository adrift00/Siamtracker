import torch
from configs.config import cfg

from models.base_siam_model import BaseSiamModel
from utils.loss import select_cross_entropy_loss, weight_l1_loss


class GDPSiamModel(BaseSiamModel):
    def __init__(self):
        super().__init__()
        self.mask = {}
        self.mask_scores = {}
        self.create_mask()

    def forward(self, examplar, search, gt_cls, gt_loc, gt_loc_weight):
        # compute the weight*mask
        model_params = dict(model.named_parameters())
        for k, mask in self.mask.items():
            if k in model_params:
                if mask.size() == model_params[k].size():
                    # todo: can't use the mm
                    model_params[k].data.mm(mask)
                else:
                    model_params[k].data.mm(mask[:, None, None, None])
        # normal forward
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
        return {
            'cls_loss': cls_loss,
            'loc_loss': loc_loss,
            'total_loss': total_loss
        }

    def create_mask(self):
        repeat_times = [1, 2, 3, 4, 3, 3, 1]
        backbone_params = dict(self.backbone.named_parameters())
        backbone_param_keys = list(backbone_params.keys())
        backbone_param_values = list(backbone_params.values())
        # for bottleneck
        # layer0
        for i in range(3):
            self.mask[backbone_param_keys[i]] = torch.ones(backbone_param_values[i].size(0))
        # layer1-7
        idx = 3
        for i in range(7):  # 6 layers
            for j in range(repeat_times[i]):  # for n bottlenecks
                for k in range(6):
                    # the 6 params will be mask, because the last layer of every bottleneck will be added.
                    self.mask[backbone_param_keys[idx + k]] = torch.ones(backbone_param_values[idx + k].size(0))
                idx += 9
        # for neck
        for k, v in self.neck.named_parameters():
            self.mask[k] = torch.ones(v.size(0))

        # for rpn
        for k, v in self.rpn.named_parameters():
            if 'head.0' in k or 'head.1' in k:  # now only prune the first layer of the head
                self.mask[k] = torch.ones(v.size(0))

    def update_mask(self, *args):
        loss = self(*args)  # self.forward
        total_loss = loss['total_loss']
        model_params = dict(self.named_parameters())
        pruned_params = {k: model_params[k] for k in self.mask.keys()}
        pruned_grads = torch.autograd.grad(total_loss, pruned_params.values())
        for i, (k, v) in enumerate(self.mask.items()):
            self.mask_scores[k] = (pruned_grads[i] * model_params[k]).sum().item()
        sorted_scores = dict(sorted(self.mask_scores.items(), key=lambda item: item[1], reverse=True))
        # get the first n layers
        pruning_num = len(sorted_scores) * cfg.PRUNING_RATE
        for i, (k, v) in enumerate(sorted_scores.items()):
            if i < pruning_num:
                self.mask[k] = 1
            else:
                self.mask[k] = 0


if __name__ == '__main__':
    cfg.merge_from_file('../configs/mobilenetv2_finetune.yaml')
    model = GDPSiamModel()
    for k, v in model.mask.items():
        print(k, v.size())
