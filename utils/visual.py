import cv2
import numpy as np
from utils.bbox import center2corner


def show_double_bbox(img, pred_bbox, gt_bbox,idx,lost_number):
    """
    :param img: (h,w,c)
    :param pred_bbox: cx,cy,w,h
    :param gt_bbox:
    :param idx: the frame idx
    :param lost_number: the lost number in the video
    :return:
    """
    pred_bbox = center2corner(pred_bbox)
    gt_bbox = center2corner(gt_bbox)
    pred_bbox = list(map(lambda x: int(x), pred_bbox))
    gt_bbox = list(map(lambda x: int(x), gt_bbox))
    cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]), (pred_bbox[2], pred_bbox[3]), (0, 0, 255), 2)
    cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]), (gt_bbox[2], gt_bbox[3]), (255, 255, 0), 2)
    cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(img, str(lost_number), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.namedWindow('double bbox',cv2.WINDOW_NORMAL| cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow('double bbox',600,400)
    cv2.imshow('double bbox', img)
    cv2.waitKey(100)


def show_single_bbox(img, bbox):
    pred_bbox = center2corner(bbox)
    pred_bbox = list(map(lambda x: int(x), pred_bbox))
    cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]), (pred_bbox[2], pred_bbox[3]), (0, 0, 255), 2)
    # cv2.namedWindow('single bbox', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('single bbox', img)
    cv2.waitKey(1)


def show_img(img):
    cv2.namedWindow('img', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('img', img)
    cv2.waitKey(1)


#def show_bboxes(img,bboxes):
#    for bbox in bboxes:
#        if isinstance(bbox,np.ndarray):
#            bbox=bbox.tolist()
#        pred_bbox = center2corner(bbox)
#        pred_bbox = list(map(lambda x: int(x), pred_bbox))
#        cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]), (pred_bbox[2], pred_bbox[3]), (0, 0, 255), 2)
#    # cv2.namedWindow('single bbox', cv2.WINDOW_AUTOSIZE)
#    cv2.imshow('bboxes', img)
#    cv2.waitKey(1)
