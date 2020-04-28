import torch
import numpy as np
from configs.config import cfg

from models.base_siam_model import BaseSiamModel
from utils.loss import select_cross_entropy_loss, weight_l1_loss


class PruningSiamModel(BaseSiamModel):
    def __init__(self):
        super().__init__()
        self.mask = {}
        self.mask_scores = {}
        self.create_mask()

    def forward(self, examplar, search, gt_cls, gt_loc, gt_loc_weight):
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

    # @torch.no_grad()
    # def forward(self, examplar):
    #     # np.set_printoptions(threshold=np.inf)
    #     # print(examplar.detach().cpu().numpy()[0,1,:,:],)
    #     # examplar = self.backbone(examplar)
    #     # if cfg.ADJUST.USE:
    #     #     examplar=self.neck(examplar)
    #     # print(examplar[0].detach().cpu().numpy())
    #     # return examplar[0],examplar[1],examplar[2]
    #     examplar=self.backbone(examplar)
    #     np.set_printoptions(threshold=np.inf)
    #     print(examplar[0,0:10,:,:].detach().cpu().numpy())
    #     return examplar

    def create_mask(self):
        repeat_times = [1, 2, 3, 4, 3, 3, 1]
        backbone_params = dict(self.backbone.named_parameters())
        backbone_param_keys = list(backbone_params.keys())
        backbone_param_values = list(backbone_params.values())
        # for bottleneck
        # layer0
        for i in range(3):
            if len(backbone_param_values[i].size()) == 1:  # skip the batchnorm
                continue
            self.mask['backbone.' + backbone_param_keys[i]] = torch.ones(backbone_param_values[i].size(0)).cuda()
        # layer1-7
        idx = 3
        for i in range(7):  # 6 layers
            for j in range(repeat_times[i]):  # for n bottlenecks
                for k in range(6):
                    # the 6 params will be mask, because the last layer of every bottleneck will be added, so don't pruning them..
                    if len(backbone_param_values[idx + k].size()) == 1:  # skip the batchnorm
                        continue
                    if len(backbone_param_values[idx + k].size()) == 4 \
                            and backbone_param_values[idx + k].size(1) == 1:  # skip the depth-wise conv
                        continue
                    self.mask['backbone.' + backbone_param_keys[idx + k]] = torch.ones(
                        backbone_param_values[idx + k].size(0)).cuda()
                idx += 9
        # for neck
        for k, v in self.neck.named_parameters():
            if len(v.size()) == 1:  # skip the batchnorm
                continue
            self.mask['neck.' + k] = torch.ones(v.size(0)).cuda()

        # for rpn
        for k, v in self.rpn.named_parameters():
            if 'head.0' in k or 'head.1' in k:  # now only prune the first layer of the head
                if len(v.size()) == 1:  # skip the batchnorm
                    continue
                self.mask['rpn.' + k] = torch.ones(v.size(0)).cuda()

    def update_mask(self):
        # sfp
        model_params = dict(self.named_parameters())
        pruned_params = {k: model_params[k] for k in self.mask.keys()}
        for i, (k, v) in enumerate(self.mask.items()):
            self.mask_scores[k] = torch.sqrt(torch.pow(pruned_params[k], 2).sum((1, 2, 3))).detach().cpu().numpy()

        for key, mask_score in self.mask_scores.items():
            keep_num = int(len(mask_score) * cfg.PRUNING.KEEP_RATE)
            sorted_idx = np.argsort(-mask_score)  # reserve order, so times -1
            self.mask[key][sorted_idx[:keep_num]] = 1
            self.mask[key][sorted_idx[keep_num:]] = 0

        # gm pruning
        # model_params=dict(self.named_parameters())
        # pruned_params={k:model_params[k] for k in self.mask.keys()}
        # for k in self.mask.keys():
        #     param=pruned_params[k]
        #     self.mask_scores[k]=np.zeros(param.size(0))
        #     for i in range(param.size(0)):
        #         filter=param[i]
        #         distance=((param-filter)**2).sum((1,2,3)).sqrt().sum()
        #         # distance=((param-filter)**2).sum()
        #         self.mask_scores[k][i]=distance
        # for key, mask_score in self.mask_scores.items():
        #     keep_num = int(len(mask_score) * cfg.PRUNING.KEEP_RATE)
        #     sorted_idx = np.argsort(-mask_score)  # reserve order, so times -1
        #     self.mask[key][sorted_idx[:keep_num]] = 1
        #     self.mask[key][sorted_idx[keep_num:]] = 0

    def apply_mask(self):
        model_params = dict(self.named_parameters())
        for key, mask in self.mask.items():
            model_params[key].data.mul_(mask[:, None, None, None])
            # bn mask
            key_prefix = key.split('.')[:-1]
            key_prefix[-1] = str(int(key_prefix[-1]) + 1)
            key_prefix = '.'.join(key_prefix)
            self._apply_bn_mask(model_params, key_prefix, mask)
            if key.startswith('backbone'):
                # for deepwise
                key_prefix = key.split('.')[:-1]
                key_prefix[-1] = str(int(key_prefix[-1]) + 3)
                key_prefix = '.'.join(key_prefix)
                self._apply_deepwise_mask(model_params, key_prefix, mask)
                # for deepwise bn
                key_prefix = key.split('.')[:-1]
                key_prefix[-1] = str(int(key_prefix[-1]) + 4)
                key_prefix = '.'.join(key_prefix)
                self._apply_bn_mask(model_params, key_prefix, mask)

    def _apply_bn_mask(self, model_params, key_prefix, mask):
        new_k = key_prefix + '.weight'
        if new_k in model_params.keys():
            model_params[new_k].data.mul_(mask)
        new_k = key_prefix + '.bias'
        if new_k in model_params.keys():
            model_params[new_k].data.mul_(mask)
        new_k = key_prefix + '.running_mean'
        if new_k in model_params.keys():
            model_params[new_k].data.mul_(mask)
        new_k = key_prefix + '.running_var'
        if new_k in model_params.keys():
            model_params[new_k].data.mul_(mask)

    def _apply_deepwise_mask(self, model_params, key_prefix, mask):
        new_k = key_prefix + '.weight'
        if new_k in model_params.keys():
            model_params[new_k].data.mul_(mask[:, None, None, None])


if __name__ == '__main__':
    cfg.merge_from_file('../configs/mobilenetv2_finetune.yaml')
    model = PruningSiamModel()
    for k, v in model.mask.items():
        print(k, v.size())
