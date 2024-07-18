#%% Analysis(region) ##############################################
import sys
import warnings 
#warnings.filterwarnings(action='ignore')
import numpy as np
import pandas as pd
np.set_printoptions(threshold=sys.maxsize)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.patches as patches
import os, glob
#from val import process_batch
from skimage import io, measure, filters
from skimage.feature import hog
#from utils.metrics import  ap_per_class
import pdb
from tqdm import tqdm
import pickle
import time
import cv2
import csv
import imageio
import math
import torch
import re
import copy
#%matplotlib inline
from sklearn.linear_model import LinearRegression
import argparse
import json
from mvextractor.videocap import VideoCap
from mvt.loaders import load_detections_orig, load_detections_YOLO, detections_to_YOLO, save_to_detections_YOLO
from analyzers_server import analyzer 
from fb_constants import *

def calculate_accuracy_list(preds_list, gt_list, names, metric='f1', imgHeight= 720, imgWidth=1280, niou=10):
    """caluclate accuracy by comparing 2 predictions"""
    def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir='.', names=(), eps=1e-16, prefix=''):
        """ Compute the average precision, given the recall and precision curves.
        Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
        # Arguments
            tp:  True positives (nparray, nx1 or nx10).
            conf:  Objectness value from 0-1 (nparray).
            pred_cls:  Predicted object classes (nparray).
            target_cls:  True object classes (nparray).
            plot:  Plot precision-recall curve at mAP@0.5
            save_dir:  Plot save directory
        # Returns
            The average precision as computed in py-faster-rcnn.
        """

        # Sort by objectness
        i = np.argsort(-conf)
        tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
        # Find unique classes
        unique_classes, nt = np.unique(target_cls, return_counts=True)
        nc = unique_classes.shape[0]  # number of classes, number of detections

        # Create Precision-Recall curve and compute AP for each class
        px, py = np.linspace(0, 1, 1000), []  # for plotting
        ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
        for ci, c in enumerate(unique_classes):
            i = pred_cls == c
            n_l = nt[ci]  # number of labels
            n_p = i.sum()  # number of predictions
            if n_p == 0 or n_l == 0:
                continue

            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_l + eps)  # recall curve
            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score
            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
                if plot and j == 0:
                    py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5
        # Compute F1 (harmonic mean of precision and recall)
        f1 = 2 * p * r / (p + r + eps)
        names = [v for k, v in names.items() if k in unique_classes]  # list: only classes that have data
        names = dict(enumerate(names))  # to dict

        i = smooth(f1.mean(0), 0.1).argmax()  # max F1 index
        p, r, f1 = p[:, i], r[:, i], f1[:, i]
        tp = (r * nt).round()  # true positives
        fp = (tp / (p + eps) - tp).round()  # false positives
        return tp, fp, p, r, f1, ap, unique_classes.astype(int)

    def process_batch(detections, labels, iouv):
        """
        Return correct prediction matrix
        Arguments:
            detections (array[N, 6]), x1, y1, x2, y2, conf, class
            labels (array[M, 5]), class, x1, y1, x2, y2
        Returns:
            correct (array[N, 10]), for 10 IoU levels
        """
        correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
        iou = box_iou(labels[:, 1:], detections[:, :4])
        correct_class = labels[:, 0:1] == detections[:, 5]
        for i in range(len(iouv)):
            x = torch.where((iou >= iouv[i]))# & correct_class)  # IoU > threshold and classes match #must consider class match
            if x[0].shape[0]:
                matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
                if x[0].shape[0] > 1:
                    matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    # matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                correct[matches[:, 1].astype(int), i] = True
        return torch.tensor(correct, dtype=torch.bool, device=iouv.device)

    def compute_ap(recall, precision):
        """ Compute the average precision, given the recall and precision curves
        # Arguments
            recall:    The recall curve (list)
            precision: The precision curve (list)
        # Returns
            Average precision, precision curve, recall curve
        """

        # Append sentinel values to beginning and end
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))

        # Compute the precision envelope
        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

        # Integrate area under curve
        method = 'interp'  # methods: 'continuous', 'interp'
        if method == 'interp':
            x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
            ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
        else:  # 'continuous'
            i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

        return ap, mpre, mrec

    def smooth(y, f=0.05):
        # Box filter of fraction f
        nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
        p = np.ones(nf // 2)  # ones padding
        yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
        return np.convolve(yp, np.ones(nf) / nf, mode='valid')  # y-smoothed

    # imgHeight, imgWidth, _ = image.shape
    acc_list = []
    p_list = []
    r_list = []
    correct_list = []
    assert len(preds_list) == len(gt_list), "Length of predictions and ground truth should be the same"

    for predBoxes,gtBoxes in zip(preds_list,gt_list):
        if len(predBoxes) == 0 or len(gtBoxes) == 0:
            continue
        # Read prediction labels
        predBoxes = predBoxes[predBoxes[:,-1]>=0.2]
        gtBoxes = gtBoxes[gtBoxes[:,-1]>=0.2]
        stats = []
        nc = 10 
        boundingBoxes = []
        # Read bounding boxes from pre-detection(x_min, y_min, x_max, y_max)
        preds = []
        iouv = torch.linspace(0.5, 0.95, niou)
        preds = torch.tensor(predBoxes)
        labels = gtBoxes.clone()#torch.tensor(gtBoxes)
        # Metrics
        npr = preds.shape[0]
        correct = torch.zeros(npr, niou, dtype=torch.bool)  # init

        # Evaluate
        labels[:, 1:5] *= torch.tensor([imgWidth, imgHeight, imgWidth, imgHeight])
        preds[:, 1:5] *= torch.tensor([imgWidth, imgHeight, imgWidth, imgHeight])

        predbox = xywh2xyxy(preds[:, 1:5])
        gtbox = xywh2xyxy(labels[:, 1:5])

        predn = torch.cat((predbox, preds[:, 5:6],preds[:, :1]), dim=1)
        labelsn = torch.cat((labels[:, 0:1], gtbox), 1)  # native-space labels
        correct = process_batch(predn, labelsn, iouv)
        stats.append((correct, predn[:, 4], predn[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)
        mp = 0 
        mr = 0 
        map50 = 0 
        map = 0
        f1 = 0
        f1_mean = 0
        # Compute metrics
        stats = [torch.cat(x, 0).numpy() for x in zip(*stats)]  # to numpy
        if len(stats) and stats[0].any():
            #tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, save_dir=save_dir, names={0: 'pedestrian', 1: 'people', 2: 'bicycle', 3: 'car', 4: 'van', 5: 'truck', 6: 'tricycle', 7: 'awning-tricycle', 8: 'bus', 9: 'motor'}) 
            tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, names=names) 
            ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
            mp, mr, map50, map, f1_mean = p.mean(), r.mean(), ap50.mean(), ap.mean(), f1.mean()
            nt = np.bincount(stats[3].astype(int), minlength=nc)
            result =  {
            'mp': mp,
            'mr': mr,
            'map50': map50,
            'map': map,
            # 'nt': nt,
            'f1': f1[0] if isinstance(f1,np.ndarray) else f1, # f1_mean
            'f1_mean': f1_mean
             }
             # acc_list.append(f1)
            acc_list.append(result[metric])
            p_list.append(mp)
            r_list.append(mr)
            correct_list.append(correct)
        else:
            acc_list.append(0)
            p_list.append(0)
            r_list.append(0)
            correct_list.append(0)
        # else:
        #     breakpoint()
              # number of targets per class
  
    return np.mean(acc_list)#, acc_list, p_list, r_list, correct_list

def calculate_accuracy(pred, gt, names, metric='f1', imgHeight= 720, imgWidth=1280, niou=10):
    """caluclate accuracy by comparing 2 predictions"""
    
    def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir='.', names=(), eps=1e-16, prefix=''):
        """ Compute the average precision, given the recall and precision curves.
        Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
        # Arguments
            tp:  True positives (nparray, nx1 or nx10).
            conf:  Objectness value from 0-1 (nparray).
            pred_cls:  Predicted object classes (nparray).
            target_cls:  True object classes (nparray).
            plot:  Plot precision-recall curve at mAP@0.5
            save_dir:  Plot save directory
        # Returns
            The average precision as computed in py-faster-rcnn.
        """

        # Sort by objectness
        i = np.argsort(-conf)
        tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
        # Find unique classes
        unique_classes, nt = np.unique(target_cls, return_counts=True)
        nc = unique_classes.shape[0]  # number of classes, number of detections

        # Create Precision-Recall curve and compute AP for each class
        px, py = np.linspace(0, 1, 1000), []  # for plotting
        ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
        for ci, c in enumerate(unique_classes):
            i = pred_cls == c
            n_l = nt[ci]  # number of labels
            n_p = i.sum()  # number of predictions
            if n_p == 0 or n_l == 0:
                continue

            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_l + eps)  # recall curve
            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score
            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
                if plot and j == 0:
                    py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5
        # Compute F1 (harmonic mean of precision and recall)
        f1 = 2 * p * r / (p + r + eps)
        names = [v for k, v in names.items() if k in unique_classes]  # list: only classes that have data
        names = dict(enumerate(names))  # to dict

        i = smooth(f1.mean(0), 0.1).argmax()  # max F1 index
        p, r, f1 = p[:, i], r[:, i], f1[:, i]
        tp = (r * nt).round()  # true positives
        fp = (tp / (p + eps) - tp).round()  # false positives
        return tp, fp, p, r, f1, ap, unique_classes.astype(int)

    def process_batch(detections, labels, iouv):
        """
        Return correct prediction matrix
        Arguments:
            detections (array[N, 6]), x1, y1, x2, y2, conf, class
            labels (array[M, 5]), class, x1, y1, x2, y2
        Returns:
            correct (array[N, 10]), for 10 IoU levels
        """
        correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
        iou = box_iou(labels[:, 1:], detections[:, :4])
        correct_class = labels[:, 0:1] == detections[:, 5]
        for i in range(len(iouv)):
            x = torch.where((iou >= iouv[i]))# & correct_class)  # IoU > threshold and classes match #must consider class match
            if x[0].shape[0]:
                matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
                if x[0].shape[0] > 1:
                    matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    # matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                correct[matches[:, 1].astype(int), i] = True
        return torch.tensor(correct, dtype=torch.bool, device=iouv.device)

    def compute_ap(recall, precision):
        """ Compute the average precision, given the recall and precision curves
        # Arguments
            recall:    The recall curve (list)
            precision: The precision curve (list)
        # Returns
            Average precision, precision curve, recall curve
        """

        # Append sentinel values to beginning and end
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))

        # Compute the precision envelope
        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

        # Integrate area under curve
        method = 'interp'  # methods: 'continuous', 'interp'
        if method == 'interp':
            x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
            ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
        else:  # 'continuous'
            i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

        return ap, mpre, mrec

    def smooth(y, f=0.05):
        # Box filter of fraction f
        nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
        p = np.ones(nf // 2)  # ones padding
        yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
        return np.convolve(yp, np.ones(nf) / nf, mode='valid')  # y-smoothed

    # imgHeight, imgWidth, _ = image.shape
    idx = 0

    # Read prediction labels
    
    predBoxes = np.copy(pred)
    gtBoxes = np.copy(gt)
    predBoxes = predBoxes[predBoxes[:,-1]>=0.3]
    gtBoxes = gtBoxes[gtBoxes[:,-1]>=0.3]
    stats = []
    nc = 10 
    # Read bounding boxes from pre-detection(x_min, y_min, x_max, y_max)
    preds = []
    iouv = torch.linspace(0.5, 0.95, niou)
    preds = torch.tensor(predBoxes)
    labels = torch.tensor(gtBoxes)
    # breakpoint()
    # Metrics
    npr = preds.shape[0]
    correct = torch.zeros(npr, niou, dtype=torch.bool)  # init

    # Evaluate
    labels[:, 1:5] *= torch.tensor([imgWidth, imgHeight, imgWidth, imgHeight])
    preds[:, 1:5] *= torch.tensor([imgWidth, imgHeight, imgWidth, imgHeight])

    predbox = xywh2xyxy(preds[:, 1:5])
    gtbox = xywh2xyxy(labels[:, 1:5])

    predn = torch.cat((predbox, preds[:, 5:6],preds[:, :1]), dim=1)
    labelsn = torch.cat((labels[:, 0:1], gtbox), 1)  # native-space labels
    correct = process_batch(predn, labelsn, iouv)
    
    stats.append((correct, predn[:, 4], predn[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)
    mp = 0 
    mr = 0 
    map50 = 0 
    map = 0
    f1 = 0
    # Compute metrics
    stats = [torch.cat(x, 0).numpy() for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        #tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, save_dir=save_dir, names={0: 'pedestrian', 1: 'people', 2: 'bicycle', 3: 'car', 4: 'van', 5: 'truck', 6: 'tricycle', 7: 'awning-tricycle', 8: 'bus', 9: 'motor'}) 
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, names=names) 
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map, f1_mean = p.mean(), r.mean(), ap50.mean(), ap.mean(), f1.mean()
    nt = np.bincount(stats[3].astype(int), minlength=nc)  # number of targets per class
    # if idx == 78:
        # breakpoint()
    result =  {
        'mp': mp,
        'mr': mr,
        'map50': map50,
        'map': map,
        'nt': nt,
        'f1': f1[0] #f1_mean
    }
    return result[metric], correct

def box_iou(box1, box2, eps=1e-7):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def xcycwh2xmymxMyMN(x, imgHeight, imgWidth):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    x[..., 0] = x[..., 0] * imgWidth
    x[..., 1] = x[..., 1] * imgHeight
    x[..., 2] = x[..., 2] * imgWidth
    x[..., 3] = x[..., 3] * imgHeight
    
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def xcycwh2xmymwhN(x, imgHeight, imgWidth):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, w, h] where xy1=top-left, in native coordinate
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    x[..., 0] = x[..., 0] * imgWidth
    x[..., 1] = x[..., 1] * imgHeight
    x[..., 2] = x[..., 2] * imgWidth
    x[..., 3] = x[..., 3] * imgHeight

    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 2]  # width
    y[..., 3] = x[..., 3]  # height
    
    return y

def loadBox2Patches(image, preds, box_color, corrects=[], fill=True, letter=False):
    # Calculate the coordinates of the box's corners
    imgHeight, imgWidth, _ = image.shape
    list_patch = []
    for idx, pred in enumerate(preds):
        x_c = pred[1] * imgWidth
        y_c = pred[2] * imgHeight
        w = pred[3] * imgWidth
        h = pred[4] * imgHeight
        x_min = x_c - w//2 
        y_min = y_c - h//2
 
        patch = patches.Rectangle((x_min, y_min), w, h, linewidth=1, edgecolor=box_color, facecolor=box_color,alpha =0.5)
        list_patch.append(patch)

    return list_patch

def visualize(image, name, preds= []):
    img_height, img_width = image.shape[:2]
    # DPI (dots per inch) for the figure
    dpi = 100  # You can adjust this value to your preference
    # Create a figure with the same pixel size as the image
    fig, ax = plt.subplots(figsize=(img_width/dpi, img_height/dpi), dpi=dpi)
    ax.imshow(image)
    assert len(preds)>0, "No predictions"
    det_patches = loadBox2Patches(image,preds,'green')
    for patch in det_patches:
        ax.add_patch(patch)
    # Remove axes and padding
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    # Save the figure to a file

    plt.savefig(f'./visualize_tmp/{name}.jpg', dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)  # Close the figure to free up memory

def load_preds_benchmark(preds_archive,frame_idx,frameHeight,frameWidth,conf_thr=0):
    pred = preds_archive[preds_archive[:,0]==frame_idx+1]
    pred = pred[:,1:] 
    col_order = [4,0,1,2,3,5]
    pred = pred[:,col_order]
    
    cls = pred[:,0]
    conf = pred[:,-1]
    
    # only leave columns with confidence higher than the threshold
    pred = pred[conf>conf_thr]
    cls = cls[conf>conf_thr]
    conf = conf[conf>conf_thr]
    boxes = pred[:,1:5]
    width = (boxes[:,2] - boxes[:,0])
    height = (boxes[:,3] - boxes[:,1])
    assert any(width > 0), f"at frame:{frame_idx}"
    assert any(height > 0), f"at frame:{frame_idx}"
    # width = (boxes[:,2] - boxes[:,0])
    # height = (boxes[:,3] - boxes[:,1])

    boxes[:,0] = boxes[:,0] + width//2
    boxes[:,1] = boxes[:,1] + height//2
    boxes[:,2] = width
    boxes[:,3] = height

    boxes[:,0] /= frameWidth
    boxes[:,1] /= frameHeight
    boxes[:,2] /= frameWidth
    boxes[:,3] /= frameHeight
    
    cls_reshaped = cls.reshape(-1, 1)
    conf_reshaped = conf.reshape(-1, 1)

    # Vertically concatenate the arrays

    if torch.is_tensor(cls_reshaped) and torch.is_tensor(boxes) and torch.is_tensor(conf_reshaped):
        det_boxes_cat = torch.cat((cls_reshaped, boxes, conf_reshaped), dim=1)
    else:
        det_boxes_cat = np.hstack((cls_reshaped, boxes, conf_reshaped))
    return det_boxes_cat

def calculate_custom_feature(regressioner, image, box_list, time_list= [], box_type = 'yolo', feature_threshold=[0.4, 0.6], feature_config = ['region','edge']):
    """
    calculate custom feature.
    mainly set the target region for feature calculation
    actual feature calculation is done in calculate_feature function
    """
    assert feature_config[0] in ['region','obj_region','box','small_box'], "Feature type not supported"
    start_t = time.time()
    def extract_bbox_region(image, box, box_type= box_type):
        # Extract box region of interest 
        # roi = np.zeros_like(image)
        imgHeight, imgWidth, _ = image.shape
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        if box_type == 'yolo':
            x_c = np.copy(box[1])
            y_c = np.copy(box[2])    
            x_min = int((x_c-box[3]/2)*imgWidth)
            y_min = int((y_c-box[4]/2)*imgHeight)
            x_max = int((x_c+box[3]/2)*imgWidth)
            y_max = int((y_c+box[4]/2)*imgHeight)    
            # roi[y_min:y_max,x_min:x_max]= image[y_min:y_max,x_min:x_max] 
            roi= image[y_min:y_max,x_min:x_max] 
            mask[y_min:y_max,x_min:x_max] = 1   
        else:
            x_min = max(0,int(box[1]))
            y_min = max(0,int(box[2]))
            x_max = min(imgWidth,int(box[1]+box[3]))
            y_max = min(imgHeight, int(box[2]+box[4]))
            # roi[y_min:y_max,x_min:x_max]= image[y_min:y_max,x_min:x_max] 
            roi= image[y_min:y_max,x_min:x_max] 
            mask[y_min:y_max,x_min:x_max] = 1           
        
        total_area = np.sum(mask)
        return roi, total_area

    def extract_tile_region(image, box, box_type= box_type):
        # Extract box region of interest 
        imgHeight, imgWidth, _ = image.shape
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        if box_type == 'yolo':
            x_c = np.copy(box[1])
            y_c = np.copy(box[2])    
            x_min = int((x_c-box[3]/2)*imgWidth)
            y_min = int((y_c-box[4]/2)*imgHeight)
            x_max = int((x_c+box[3]/2)*imgWidth)
            y_max = int((y_c+box[4]/2)*imgHeight)    
            # roi[y_min:y_max,x_min:x_max]= image[y_min:y_max,x_min:x_max] 
            roi= image[y_min:y_max,x_min:x_max] 
            mask[y_min:y_max,x_min:x_max] = 1   
        else:
            x_min = max(0,int(box[1]))
            y_min = max(0,int(box[2]))
            x_max = min(imgWidth,int(box[1]+box[3]))
            y_max = min(imgHeight, int(box[2]+box[4]))
            # roi[y_min:y_max,x_min:x_max]= image[y_min:y_max,x_min:x_max] 
            roi= image[y_min:y_max,x_min:x_max] 
            mask[y_min:y_max,x_min:x_max] = 1           
        
        total_area = np.sum(mask)
        return roi, total_area

    # def extract_target_tile(image,box, box_type= box_type):
        
    if 'region' in feature_config[0]:
        if feature_config[0] == 'region':
            target_region = image
        elif feature_config[0] == 'obj_region':
            target_region = image[60:240, :]
        feature = calculate_feature(target_region, feature_config[1])
        
    elif 'box' in feature_config[0]: 
        det_box_list = box_list[box_list[:,5] >= 0.3]
        for box in det_box_list:
            box_feature_list = []
            region, box_size = extract_bbox_region(image,box,box_type=box_type)
            if region.shape[0] == 0 or region.shape[1] == 0:
                continue
            if 'small' in feature_config[0] and box_size>4000:
                continue
            
            feature = calculate_feature(region,feature_config[1],regressioner = regressioner)
                
            box_feature_list.append(feature)

        box_feature_list = np.array(box_feature_list)

        feature = np.sum(box_feature_list) if len(box_feature_list)>0 else 0 
        
    end_t = time.time()
    time_list.append(end_t - start_t)
    return feature

def calculate_feature(image, feature_type, regressioner = None):
    # assert feature_type in   
    if image.dtype != np.uint8:
        image = np.uint8(image * 255)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if feature_type == 'blur':
        # Calculate blurriness (Laplacian variance)
        blur_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return blur_var

    if 'edge' in feature_type:
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_img = np.sqrt(sobelx**2 + sobely**2)#.mean()
        edge_count = np.count_nonzero(edge_img)
        imgHeight,imgWidth = gray.shape
        area = imgHeight * imgWidth
        if 'sum' in feature_type:
            return edge_img.sum()
        if 'mean' in feature_type:
            return edge_img.mean()
        if 'var' in feature_type:
            return edge_img.var()
        if 'rel1' in feature_type:
            return edge_img.sum()/area
        if 'rel2' in feature_type:
            return edge_img.sum()/edge_count
        if 'rel3' in feature_type:
            assert regressioner is not None, "Regressioner not provided"
            ref_feature = regressioner.predict([[area]])
            return edge_img.sum()/ref_feature
        
    elif feature_type == 'corner':
        gray = np.float32(gray)
        corner = cv2.cornerHarris(gray, 5, 3, 0.05)
        corner = cv2.dilate(corner, None)
        # if corner.any():
        corner_count = np.sum(corner > 0.01 * corner.max())
        return corner_count
        
    elif feature_type =='hog':
        hog_val = hog(gray, orientations=10,
                        pixels_per_cell=(7,7),
                        cells_per_block=(3,3)
                        ).astype('float32')
        hog_magnitude = np.linalg.norm(hog_val)
        return hog_magnitude
    
    elif feature_type =='sift':
        sift = cv2.SIFT_create()
        keypoints, des = sift.detectAndCompute(gray, None)
        return len(keypoints)
    
    elif feature_type =='surf':
        # Create SURF Object
        hessian_threshold = 100
        surf = cv2.xfeatures2d.SURF_create(hessian_threshold)

        # Detect keypoints and descriptors
        keypoints_surf, descriptors_surf = surf.detectAndCompute(gray, None)
        return len(keypoints_surf)

def numpy_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: numpy_to_list(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_list(item) for item in obj]
    return obj

def check_existing_result(json_dir,vid_name, res, det_model_name, criterion, target_acc, tracker_type):
    """Check if a result JSON file already exists for the given parameters."""
    filename = f"{json_dir}/{vid_name}_{res}_{det_model_name}_{criterion}_{target_acc}_{tracker_type}.json"
    return os.path.exists(filename)

# vid_name = 'highway'
# res = '720p'
# det_model_name = 'yolov5m'  #'yolov5s' 'frcnn_resnet50'
# target_acc = '0.9'
# tracker_type = 'mvt'

#Parse command line arguments
parser = argparse.ArgumentParser(description='Run video analysis with different settings.')
parser.add_argument('--det_model', type=str, default='yolov5s', help='Detection model name')
parser.add_argument('--vid_name', type=str, default='highway', help='Video name')
parser.add_argument('--tracker', type=str, default='mvt', help='Tracker type')
parser.add_argument('--target_acc', type=str, default='0.8', help='Target accuracy')

args = parser.parse_args()
print(f"Received arguments: {sys.argv}")
vid_name = args.vid_name
det_model_name = args.det_model
target_acc = float(args.target_acc)
tracker_type = args.tracker

#Environmental setting
res = f'{video_res[vid_name][1]}p'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
metric = 'f1'
print(f'evaluation metric: {metric}')
det_model_type = 'yolo'

# Load data
# Use the parsed arguments
curr_path = os.getcwd()
data_dir_Root = f"{curr_path}/../datasets/yoda/{vid_name}/{res}"
video_path_LR = f"{data_dir_Root}/{vid_name}_{res}.mp4"
# gt_file_path = f'{data_dir_Root}/{vid_name}/ground_truth/gt_yolov56l_coco.csv'
frameWidth = video_res[vid_name][0]
frameHeight = video_res[vid_name][1]
vid_fps = int(video_fps[vid_name])

# trigger_criterion = 'reducto'
pred_list = []
gt_list = []
accuracy_list = []

gt_length = 20*vid_fps
tot_range = [0,gt_length]

frame_idx = list(range(gt_length)) 
for idx in frame_idx:
    pred = torch.load(f'{data_dir_Root}/detection_data/{det_model_name}/frame{idx}.pkl')
    pred_list.append(pred)
    
    gt = torch.load(f'{data_dir_Root}/smoothed_detection_data/{det_model_name}/frame{idx}.pkl')
    gt_list.append(gt)

video_data=[]
for idx in range(0,gt_length):
    #load pkl within range
    with open(f'{data_dir_Root}/video_data/frame{idx}.pkl', 'rb') as f:
        frame_data = torch.load(f)
        video_data.append(frame_data)     
    

segment = True
setting1 = {
            'criterion':'fixed',
            'diff_feat':'pixel',
            'segment': segment
            }  
setting2 = {
            'criterion':'glimpse',
            'diff_feat':'pixel',
            'segment': segment
            }
setting3 = {
            'criterion':'reducto',
            'diff_feat':'pixel',
            'segment': segment
            }
setting4 = {
            'criterion':'frameboost',
            'diff_feat':'area',
            'segment': segment
            }
setting5 = {
            'criterion':'frameboost_v1',
            'diff_feat':'area',
            'segment': segment
            }
setting6 = {
            'criterion':'frameboost_v3',
            'diff_feat':'area',
            'segment': segment
            }

settings_list = [setting1,setting2,setting3,setting4,setting6]
# Create a directory to store results if it doesn't exist

plt_list = []

for eval_setting in settings_list:
    scheme = eval_setting['criterion']
    json_dir = f'./results/json_server/{scheme}'
    os.makedirs(json_dir, exist_ok=True)
    
    setting_name = f"{vid_name}_{res}_{det_model_name}_{eval_setting['criterion']}_{target_acc}_{tracker_type}"
    json_filename = f"{json_dir}/{setting_name}.json"

    if check_existing_result(json_dir,vid_name, res, det_model_name, eval_setting['criterion'], target_acc, tracker_type):
        print(f"Result for {setting_name} already exists. Skipping evaluation.")
        continue
    warmup_end = 10*vid_fps
    dryrun_start = 5*vid_fps
    test_range_start = warmup_end
    test_range_end = gt_length

    warmup_range = [0,warmup_end]
    dryrun_range = [dryrun_start,test_range_end] if eval_setting['criterion'] =='reducto' else [test_range_start,test_range_end]
    test_range = [test_range_start,test_range_end]
    short_video_length= video_fps[vid_name] if eval_setting['segment'] else test_range[1]-test_range[0]
    
    video_analyzer1 = analyzer(vid_name, video_path_LR, tot_range, video_data, trigger_criterion=eval_setting['criterion'], 
                               target_acc=target_acc, det_model_type=det_model_type, 
                               preds_list=copy.deepcopy(pred_list), tracker_type=tracker_type)
    
    video_analyzer1.warmup(warmup_range, keyframe_setting=eval_setting)
    tot_profile_time = video_analyzer1.dryrun(dryrun_range, short_video_length=short_video_length, keyframe_setting=eval_setting)
    result_dict = video_analyzer1.run(test_range, short_video_length=short_video_length, visualize=False)  

    # Process and save results
    plt_dict = {'x': [], 'y': [], 'boxes': [], 'num_det': [], 'det_frames': [], 'tracking_time': [], 'profile_time':[]}
    for key, value in result_dict.items():
        range_start = test_range[0] + key * short_video_length
        range_end = range_start + short_video_length
        box_list_baseline = value['boxes']
        pred_list_test = pred_list[range_start:range_end]
        gt_list_test = gt_list[range_start:range_end]
        accuracy1 = calculate_accuracy_list(box_list_baseline, copy.deepcopy(gt_list_test), COCO_names, 
                                            metric=metric, imgWidth=frameWidth, imgHeight=frameHeight)
        plt_dict['y'].append(accuracy1)
        plt_dict['boxes'].append(value['boxes'])
        plt_dict['num_det'].append(value['num_det'])
        plt_dict['det_frames'].append(value['det_frames'])
        plt_dict['tracking_time'].append(value['tracking_time'])
        plt_dict['profile_time'].append(tot_profile_time/len(result_dict))

    inference_latency = np.mean(server_inference_dict[res][det_model_name])/1000
    eps = 1e-7
    plt_dict['x'] = [short_video_length/(d*inference_latency + t_t + t_p) for d,t_t,t_p in 
                     zip(plt_dict['num_det'], plt_dict['tracking_time'], plt_dict['profile_time'])]

    # Save results to JSON
    results = {
        'setting': setting_name,
        'target_acc': video_analyzer1.target_acc,
        'warmup_range': warmup_range,
        'dryrun_range': dryrun_range,
        'test_range': test_range,
        'tracker_type': video_analyzer1.tracker_type,
        'eval_setting': eval_setting,
        'x': plt_dict['x'],
        'y': plt_dict['y'],
        # 'boxes': plt_dict['boxes'],
        'num_det': plt_dict['num_det'],
        'det_frames': plt_dict['det_frames'],
        'tracking_time': plt_dict['tracking_time'],
        'profile_time': tot_profile_time
    }
    
    results = numpy_to_list(results)  # Ensure all numpy arrays are converted to lists
    
    with open(json_filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {json_filename}")

    plt_list.append(plt_dict)
# %%
