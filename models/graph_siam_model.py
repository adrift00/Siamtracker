from configs.config import cfg
from models.base_siam_model import BaseSiamModel
from models.gcn.similar_gcn import SimilarGCN
from utils.loss import select_cross_entropy_loss, weight_l1_loss


class GraphSiamModel(BaseSiamModel):
    def __init__(self):
        super(GraphSiamModel, self).__init__()
        self.gcn = SimilarGCN(**cfg.GRAPH.KWARGS)

    def set_examplar(self, examplars):
        examplars = self.backbone(examplars)
        if cfg.ADJUST.USE:
            examplars = self.neck(examplars)
        self.examplar = self.gcn(examplars)

    def forward(self, examplars, search, gt_cls, gt_loc, gt_loc_weight):
        examplars = self.backbone(examplars)
        search = self.backbone(search)
        if cfg.ADJUST.USE:
            examplars = self.neck(examplars)
            search = self.neck(search)

        examplar = self.gcn(examplars)
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

    def track(self, search):
        search = self.backbone(search)
        if cfg.ADJUST.USE:
            search = self.neck(search)
        examplar = self.examplar
        pred_cls, pred_loc = self.rpn(examplar, search)
        return pred_cls, pred_loc




