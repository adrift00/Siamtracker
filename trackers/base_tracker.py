import cv2
import numpy as np
import torch
import torch.nn.functional as F
from configs.config import cfg
from utils.bbox import Corner


class BaseTracker(object):
    def init(self, img, bbox):
        raise NotImplementedError

    def track(self, img):
        raise NotImplementedError

    def get_subwindow(self, img, pos, dst_size, ori_size, padding):
        # scale = (dst_size) / ori_size
        # shift_x = -scale * (pos[0] - ori_size / 2)
        # shift_y = -scale * (pos[1] - ori_size / 2)
        # mapping = np.array([[scale, 0, shift_x],
        #                     [0, scale, shift_y]], dtype=np.float)
        # patch = cv2.warpAffine(img, mapping, (dst_size, dst_size), borderMode=cv2.BORDER_CONSTANT,
        #                        borderValue=padding)
        # return patch
        ori_size = int(ori_size)
        img_h, img_w, img_c = img.shape
        x1, y1 = round(pos[0] - ori_size / 2), round(pos[1] - ori_size / 2) # round or floor???
        x2, y2 = x1 + ori_size, y1 + ori_size
        cx1, cy1, cx2, cy2 = int(max(x1, 0)), int(max(y1, 0)), int(min(x2, img_w)), int(min(y2, img_h))
        left_pad, top_pad, right_pad, bottom_pad = map(lambda x: int(max(x, 0)), [-x1, -y1, x2-img_w, y2-img_h])
        if any([left_pad, top_pad, right_pad, bottom_pad]):
            patch = np.zeros((ori_size, ori_size, img_c), dtype=np.uint8) # use dtype=np.uint8???
            patch[top_pad:ori_size-bottom_pad, left_pad:ori_size-right_pad, :] = img[cy1:cy2, cx1:cx2, :]
            if left_pad:
                patch[:, 0:left_pad, :] = padding
            if top_pad:
                patch[0:top_pad, :, :] = padding
            if right_pad:
                patch[:, ori_size-right_pad:ori_size, :] = padding
            if bottom_pad:
                patch[ori_size-bottom_pad:ori_size, :, :] = padding
        else:
            patch = img[cy1:cy2, cx1:cx2, :]
        patch = cv2.resize(patch, (dst_size, dst_size))
        return patch

    def _convert_score(self, cls):
        cls = cls.reshape(2, -1).permute(1, 0)
        score = F.softmax(cls, dim=1)
        score = score.data[:, 1].cpu().numpy()
        return score

    def _clip_bbox(self, cx, cy, w, h, img_w, img_h):
        cx = np.clip(cx, 0, img_w)
        cy = np.clip(cy, 0, img_h)
        w = np.clip(w, 10, img_w)
        h = np.clip(h, 10, img_h)
        return [cx, cy, w, h]

    def _get_bbox(self, img_c, bbox_size, scale):
        o_w, o_h = bbox_size[0] * scale, bbox_size[1] * scale
        return Corner(img_c - o_w / 2, img_c - o_h / 2, img_c + o_w / 2, img_c + o_h / 2)

    def _size_z(self, bbox_size):
        context_amount = 0.5
        w_z = bbox_size[0] + context_amount * sum(bbox_size)
        h_z = bbox_size[1] + context_amount * sum(bbox_size)
        size_z = np.sqrt(w_z * h_z)
        return round(size_z)

    def _size_x(self, bbox_size):
        context_amount = 0.5
        w_z = bbox_size[0] + context_amount * sum(bbox_size)
        h_z = bbox_size[1] + context_amount * sum(bbox_size)
        size_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXAMPLAR_SIZE / size_z
        d_search = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXAMPLAR_SIZE) / 2
        pad = d_search / scale_z
        size_x = size_z + 2 * pad
        return round(size_x)
