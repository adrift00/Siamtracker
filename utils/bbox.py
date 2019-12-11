import numpy as np
import torch
from collections import namedtuple


Corner = namedtuple('Corner', 'x1 y1 x2 y2')
# alias
BBox = Corner
Center = namedtuple('Center', 'x y w h')
def center2corner(center):
    if isinstance(center,list):
        cx,cy,w,h=center
        return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
    elif isinstance(center, Center):
        x, y, w, h = center
        return Corner(x - w * 0.5, y - h * 0.5, x + w * 0.5, y + h * 0.5)
    elif isinstance(center,np.ndarray) and len(center.shape)==2:
        cx,cy,w,h=center[:,0],center[:,1],center[:,2],center[:,3]
        x1,y1,x2,y2=cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        corner=np.hstack((x1[:,np.newaxis],y1[:,np.newaxis],x2[:,np.newaxis],y2[:,np.newaxis]))
        return corner
    elif isinstance(center,np.ndarray) and len(center.shape)==4:
        cx, cy, w, h = center[0], center[1], center[2], center[3]
        x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        return np.stack((x1,y1,x2,y2))
    else:
        raise Exception('invalid center!',center)

def corner2center(corner):
    if isinstance(corner,list):
        x1,y1,x2,y2=corner
        return [(x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1), (y2 - y1)]
    elif isinstance(corner, Corner):
        x1, y1, x2, y2 = corner
        return Center((x1 + x2) * 0.5, (y1 + y2) * 0.5, (x2 - x1), (y2 - y1))
    elif isinstance(corner,np.ndarray) and len(corner.shape)==2:
        x1,y1,x2,y2=corner[:,0],corner[:,1],corner[:,2],corner[:,3]
        cx,cy,w,h=(x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1), (y2 - y1)
        center = np.hstack((cx[:, np.newaxis], cy[:, np.newaxis], w[:, np.newaxis], h[:, np.newaxis]))
        return center
    elif isinstance(corner,np.ndarray) and len(corner.shape)==4:
        x1, y1, x2, y2=corner[0],corner[1],corner[2],corner[3]
        cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1), (y2 - y1)
        return np.stack((cx,cy,w,h))
    else:
        raise Exception('invalid corner!',corner)



def delta2bbox(src_bbox, delta):
    if isinstance(delta,torch.Tensor):
        delta=delta.data.cpu().numpy()
    src_width = src_bbox[2] - src_bbox[0]
    src_height = src_bbox[3] - src_bbox[1]
    src_ctr_x = src_bbox[0] + src_width * 0.5
    src_ctr_y = src_bbox[1] + src_height * 0.5

    dx, dy, dw, dh = delta

    dst_ctr_x = dx * src_width + src_ctr_x
    dst_ctr_y = dy * src_height + src_ctr_y
    dst_width = src_width * np.exp(dw)
    dst_height = src_height * np.exp(dh)

    x1 = dst_ctr_x - dst_width * 0.5
    y1 = dst_ctr_y - dst_height * 0.5
    x2 = dst_ctr_x + dst_width * 0.5
    y2 = dst_ctr_y + dst_height * 0.5

    dst_boxes = np.stack((x1, y1, x2, y2))
    return dst_boxes


def bbox2delta(src_bbox, dst_bbox):
    src_width = src_bbox[2] - src_bbox[0]
    src_height = src_bbox[3] - src_bbox[1]
    src_ctr_x = src_bbox[0] + src_width * 0.5
    src_ctr_y = src_bbox[1] + src_height * 0.5

    dst_width = dst_bbox[2] - dst_bbox[0]
    dst_height = dst_bbox[3] - dst_bbox[1]
    dst_ctr_x = dst_bbox[0] + dst_width * 0.5
    dst_ctr_y = dst_bbox[1] + dst_height * 0.5

    # NOTE: eps is needed to avoid divide 0
    eps = np.finfo(src_height.dtype).eps
    src_width = np.maximum(src_width, eps)
    src_height = np.maximum(src_height, eps)

    dx = (dst_ctr_x - src_ctr_x) / src_width
    dy = (dst_ctr_y - src_ctr_y) / src_height
    dw = np.log(dst_width / src_width)
    dh = np.log(dst_height / src_height)

    delta = np.stack((dx, dy, dw, dh))
    return delta


def calc_iou(rect1, rect2):
    """ caculate interection over union
    Args:
        rect1: (x1, y1, x2, y2)
        rect2: (x1, y1, x2, y2)
    Returns:
        iou
    """
    # overlap
    x1, y1, x2, y2 = rect1[0], rect1[1], rect1[2], rect1[3]
    tx1, ty1, tx2, ty2 = rect2[0], rect2[1], rect2[2], rect2[3]

    xx1 = np.maximum(tx1, x1)
    yy1 = np.maximum(ty1, y1)
    xx2 = np.minimum(tx2, x2)
    yy2 = np.minimum(ty2, y2)

    ww = np.maximum(0, xx2 - xx1)
    hh = np.maximum(0, yy2 - yy1)

    area = (x2 - x1) * (y2 - y1)
    target_a = (tx2 - tx1) * (ty2 - ty1)
    inter = ww * hh
    iou = inter / (area + target_a - inter)
    return iou
