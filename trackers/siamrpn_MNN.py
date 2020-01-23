import cv2
import torch
import MNN
import numpy as np
from utils.bbox import delta2bbox, corner2center
from utils.anchor import AnchorGenerator
from trackers.base_tracker import BaseTracker
from configs.config import cfg


class SiamRPN_MNN(BaseTracker):
    def __init__(self):
        super(SiamRPN_MNN, self).__init__()
        self.anchor_generator = AnchorGenerator(cfg.ANCHOR.SCALES,
                                                cfg.ANCHOR.RATIOS,
                                                cfg.ANCHOR.STRIDE)
        self.score_size = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXAMPLAR_SIZE) // \
                          cfg.ANCHOR.STRIDE + 1 + cfg.TRACK.BASE_SIZE
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.window = np.tile(window.flatten(), self.anchor_generator.anchor_num)
        self.all_anchor = self.anchor_generator.generate_all_anchors(cfg.TRACK.INSTANCE_SIZE // 2, self.score_size)

        self.exam_interp = MNN.Interpreter("./pretrained_models/siamrpn_examplar_sim.mnn")
        self.exam_sess = self.exam_interp.createSession()
        self.exam_input = self.exam_interp.getSessionInput(self.exam_sess)

        self.search_interp = MNN.Interpreter("./pretrained_models/siamrpn_search_sim.mnn")
        self.search_sess = self.search_interp.createSession()
        self.search_input = []
        self.search_input.append(self.search_interp.getSessionInput(self.search_sess, 'examplar'))
        self.search_input.append(self.search_interp.getSessionInput(self.search_sess, 'search'))

    def init(self, img, bbox):
        bbox_pos = bbox[0:2]  # cx,cy
        bbox_size = bbox[2:4]  # w,h
        size_z = self._size_z(bbox_size)
        self.channel_average = img.mean((0, 1))
        examplar = self.get_subwindow(img, bbox_pos, cfg.TRACK.EXAMPLAR_SIZE, size_z, self.channel_average) \
            .transpose((2, 0, 1))
        examplar = examplar[np.newaxis, :]
        examplar_data = MNN.Tensor((1, 3, 127, 127), MNN.Halide_Type_Float, examplar, MNN.Tensor_DimensionType_Caffe)
        self.exam_input.copyFrom(examplar_data)
        self.exam_interp.runSession(self.exam_sess)
        self.examplar = self.exam_interp.getSessionOutput(self.exam_sess)  # the feature map of examplar
        out = np.zeros((1, 256, 6, 6))
        out_data = MNN.Tensor((1, 256, 6, 6), MNN.Halide_Type_Float, out, MNN.Tensor_DimensionType_Caffe)
        self.examplar.copyToHostTensor(out_data)
        self.search_input[0].copyFrom(out_data)
        # import ipdb;
        # ipdb.set_trace()
        self.bbox_pos = bbox_pos
        self.bbox_size = bbox_size

    def track(self, img):
        bbox_size = self.bbox_size
        size_z = self._size_z(bbox_size)
        scale_z = cfg.TRACK.EXAMPLAR_SIZE / size_z
        size_x = self._size_x(bbox_size)
        search = self.get_subwindow(img, self.bbox_pos, cfg.TRACK.INSTANCE_SIZE, size_x, self.channel_average) \
            .transpose((2, 0, 1))
        search = search[np.newaxis, :]
        search_data = MNN.Tensor((1, 3, 287, 287), MNN.Halide_Type_Float, search, MNN.Tensor_DimensionType_Caffe)
        self.search_input[1].copyFrom(search_data)
        self.search_interp.runSession(self.search_sess)
        cls = self.search_interp.getSessionOutput(self.search_sess, 'cls')
        loc = self.search_interp.getSessionOutput(self.search_sess, 'loc')
        # import ipdb; ipdb.set_trace()
        # cls convert to torch tensor
        cls_array = np.zeros((1, 10, 21, 21))
        cls_data = MNN.Tensor((1, 10, 21, 21), MNN.Halide_Type_Float, cls_array, MNN.Tensor_DimensionType_Caffe)
        cls.copyToHostTensor(cls_data)
        cls_data = np.array(cls_data.getData())
        cls = torch.from_numpy(cls_data.reshape((1, 10, 21, 21)).astype(np.float32))
        # import ipdb;ipdb.set_trace()
        # loc convert to tensor
        loc_array = np.zeros((1, 20, 21, 21))
        loc_data = MNN.Tensor((1, 20, 21, 21), MNN.Halide_Type_Float, loc_array, MNN.Tensor_DimensionType_Caffe)
        loc.copyToHostTensor(loc_data)
        loc_data = np.array(loc_data.getData())
        loc = torch.from_numpy(loc_data.reshape((1, 20, 21, 21)).astype(np.float32))
        score = self._convert_score(cls)
        import ipdb;ipdb.set_trace()
        loc = loc.reshape(4, self.anchor_generator.anchor_num, loc.size()[2], loc.size()[3])
        pred_bbox = delta2bbox(self.all_anchor, loc)
        pred_bbox = pred_bbox.transpose((1, 2, 3, 0)).reshape((-1, 4))  # x1,y1,x2,y2
        pred_bbox = corner2center(pred_bbox)  # cx,cy,w,h

        def change(r):
            return np.maximum(r, 1 / r)

        def s_z(w, h):
            w_z = w + 0.5 * (w + h)
            h_z = h + 0.5 * (w + h)
            size_z = np.sqrt(w_z * h_z)
            return size_z

        rc = change((bbox_size[0] / bbox_size[1]) / (pred_bbox[:, 2] / pred_bbox[:, 3]))
        sc = change(
            s_z(self.bbox_size[0] * scale_z, self.bbox_size[1] * scale_z) / s_z(pred_bbox[:, 2], pred_bbox[:, 3]))
        penalty = np.exp(-(rc * sc - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
                 self.window * cfg.TRACK.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore)
        best_bbox = pred_bbox[best_idx, :]
        # show_single_bbox(search,best_bbox.tolist())
        best_bbox[0] -= cfg.TRACK.INSTANCE_SIZE // 2
        best_bbox[1] -= cfg.TRACK.INSTANCE_SIZE // 2
        best_bbox = best_bbox / scale_z
        cx = best_bbox[0] + self.bbox_pos[0]
        cy = best_bbox[1] + self.bbox_pos[1]
        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR
        w = self.bbox_size[0] * (1 - lr) + lr * best_bbox[2]
        h = self.bbox_size[1] * (1 - lr) + lr * best_bbox[3]
        pred_bbox = self._clip_bbox(cx, cy, w, h, img.shape[1], img.shape[0])
        # update
        self.bbox_pos = pred_bbox[0:2]
        self.bbox_size = pred_bbox[2:4]

        return {
            'bbox': pred_bbox,
            'score': score[best_idx]
        }
