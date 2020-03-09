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
        ori_size = int(ori_size)
        img_h, img_w, img_c = img.shape
        x1, y1 = np.floor(pos[0] - (ori_size + 1) / 2 + 0.5), np.floor(
            pos[1] - (ori_size + 1) / 2 + 0.5)  # round or floor???
        x2, y2 = x1 + ori_size - 1, y1 + ori_size - 1
        cx1, cy1, cx2, cy2 = int(max(x1, 0)), int(max(y1, 0)), int(min(x2, img_w)), int(min(y2, img_h))
        left_pad, top_pad, right_pad, bottom_pad = map(lambda x: int(max(x, 0)),
                                                       [-x1, -y1, x2 - img_w + 1, y2 - img_h + 1])
        if any([left_pad, top_pad, right_pad, bottom_pad]):
            patch = np.zeros((ori_size, ori_size, img_c), dtype=np.uint8)  # use dtype=np.uint8???
            patch[top_pad:ori_size - bottom_pad, left_pad:ori_size - right_pad, :] = img[cy1:cy2 + 1, cx1:cx2 + 1, :]
            if left_pad:
                patch[:, 0:left_pad, :] = padding
            if top_pad:
                patch[0:top_pad, :, :] = padding
            if right_pad:
                patch[:, ori_size - right_pad:ori_size, :] = padding
            if bottom_pad:
                patch[ori_size - bottom_pad:ori_size, :, :] = padding
        else:
            patch = img[cy1:cy2 + 1, cx1:cx2 + 1, :]
        patch = cv2.resize(patch, (dst_size, dst_size))
        return patch

        # ori_size = int(ori_size)
        # sz = ori_size
        # dst_size = int(dst_size)
        # im_sz = img.shape
        # c = (ori_size + 1) / 2
        # # context_xmin = round(pos[0] - c) # py2 and py3 round
        # context_xmin = np.floor(pos[0] - c + 0.5)
        # context_xmax = context_xmin + sz - 1
        # # context_ymin = round(pos[1] - c)
        # context_ymin = np.floor(pos[1] - c + 0.5)
        # context_ymax = context_ymin + sz - 1
        # left_pad = int(max(0., -context_xmin))
        # top_pad = int(max(0., -context_ymin))
        # right_pad = int(max(0., context_xmax - im_sz[1] + 1))
        # bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))
        #
        # context_xmin = context_xmin + left_pad
        # context_xmax = context_xmax + left_pad
        # context_ymin = context_ymin + top_pad
        # context_ymax = context_ymax + top_pad
        #
        # r, c, k = img.shape
        # if any([top_pad, bottom_pad, left_pad, right_pad]):
        #     size = (r + top_pad + bottom_pad, c + left_pad + right_pad, k)
        #     te_im = np.zeros(size, np.uint8)
        #     te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = img
        #     if top_pad:
        #         te_im[0:top_pad, left_pad:left_pad + c, :] = padding
        #     if bottom_pad:
        #         te_im[r + top_pad:, left_pad:left_pad + c, :] = padding
        #     if left_pad:
        #         te_im[:, 0:left_pad, :] = padding
        #     if right_pad:
        #         te_im[:, c + left_pad:, :] = padding
        #     im_patch = te_im[int(context_ymin):int(context_ymax + 1),
        #                int(context_xmin):int(context_xmax + 1), :]
        # else:
        #     im_patch = img[int(context_ymin):int(context_ymax + 1),
        #                int(context_xmin):int(context_xmax + 1), :]
        #
        # if not np.array_equal(dst_size, ori_size):
        #     im_patch = cv2.resize(im_patch, (dst_size, dst_size))
        # return im_patch

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
        return size_z

    def _size_x(self, bbox_size):
        context_amount = 0.5
        w_z = bbox_size[0] + context_amount * sum(bbox_size)
        h_z = bbox_size[1] + context_amount * sum(bbox_size)
        size_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXAMPLAR_SIZE / size_z
        d_search = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXAMPLAR_SIZE) / 2
        pad = d_search / scale_z
        size_x = size_z + 2 * pad
        return size_x
