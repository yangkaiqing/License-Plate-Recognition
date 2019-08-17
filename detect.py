#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#import _init_paths
from lib.model.config import cfg
from lib.model.test import im_detect
from lib.model.nms_wrapper import nms

from lib.utils.timer import Timer
import tensorflow as tf
import numpy as np
import os, cv2

#from lib.nets.vgg16 import vgg16
from lib.nets.resnet_v1 import resnetv1


CLASSES = ('__background__',
           'blue plate', 'black plate', 'yellow plate', 'white plate', 'green plate')
def vis_detections(im, class_name, dets, path_list,result,out,color, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    # flag = 0
    if len(inds) == 0:
        # imgout=np.zeros((46, 147),dtype=np.uint8)
        # class_name ='blue plate'
       # out = np.zeros((224, 224), dtype=np.uint8)
        return result,out,color
       # return result, color
    for i in inds:
        # flag = 1
        bbox = dets[i, :4]
        score = dets[i, -1]
        imgout = im[ int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
        # outputdir = 'E:/finall\out/'
        # cv2.imencode('.jpg', imgout)[1].tofile(outputdir + path_list)
      #  bbox = dets[i, :4]
      #  score = dets[i, -1]
        w = int(bbox[2]) - int(bbox[0]) + 1
        h = int(bbox[3]) - int(bbox[1]) + 1
        halfup = 0
        halfdown = 0
        halfl = 0
        halfr = 0
        if w >= h:
            cha = w - h
            halfup = cha // 2
            halfdown = cha - halfup
            halfl = 0
            halfr = 0
        else:
            cha = h - w
            halfl = cha // 2
            halfr = cha - halfl
            halfdown = 0
            halfup = 0


        x1 = int(bbox[0]) - halfl
        y1 = int(bbox[1]) - halfup
        x2 = int(bbox[2]) + halfr
        y2 = int(bbox[3]) + halfdown

        x11 = int(bbox[0]) - halfl
        y11 = int(bbox[1]) - halfup
        x22 = int(bbox[2]) + halfr
        y22 = int(bbox[3]) + halfdown

        if x11 < 0:
            x1 = 0
            x2 = h
        if x22 >= im.shape[1]:
            x2 = im.shape[1] - 1
            x1 = im.shape[1] - 1 - h
        if y11 < 0:
            y1 = 0
            y2 = w
        if y22 >= im.shape[0]:
            y2 = im.shape[0] - 1
            y1 = im.shape[0] - 1 - w

        if (x1 < 0) | (y1 < 0) | (x2 >= im.shape[1]) | (y2 >= im.shape[0]) | (x1 >= x2) | (y1 >= y2):
            x1 = int(bbox[0]) - halfl
            y1 = int(bbox[1]) - halfup
            x2 = int(bbox[2]) + halfr
            y2 = int(bbox[3]) + halfdown

        cv2.rectangle(im, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0,0,255), 2)
        cv2.putText(im, str(score), (int(bbox[0]), int(bbox[1] - 2)), 0, 0.6, (0, 0, 255), 1)
        out = im[y1:y2 + 1, x1:x2 + 1]
        # print('---------------------------------')
        # print(out.shape)
        # print('out',out)

        return imgout,out,class_name

def demo(sess, net, image_name, path_list):
    """Detect object classes in an image using pre-computed object proposals."""
    # Load the demo image

    im = cv2.imdecode(np.fromfile(image_name, dtype=np.uint8), -1)
    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    #print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.5
    NMS_THRESH = 0.3

    result = np.zeros((46, 147,3), dtype=np.uint8)
    outout = np.zeros((224, 224,3), dtype=np.uint8)
    color = 'blue plate'

    for cls_ind, cls in enumerate(CLASSES[1:]):
    #    print(cls_ind, cls)
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        #print(cls_boxes)

     #   print('scores',scores)
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
    #    print(dets)

        result,outout,color = vis_detections(im, cls, dets, path_list, result,outout,color,thresh=CONF_THRESH)

    return result,outout,color






