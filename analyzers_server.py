#%% Analysis(region) ##############################################
import sys
import numpy as np
import copy
# np.set_printoptions(threshold=sys.maxsize)
import os, glob
#from val import process_batch
from skimage import io, measure, filters
from skimage.feature import hog
#from utils.metrics import  ap_per_class
import cv2
import torch
import torchvision
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import math
from differencer import Differencer
from mvextractor.videocap import VideoCap
from trackers import videoTracker
from mvt.tracker import MotionVectorTracker
from mvt import trackerlib
#from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from mvt.loaders import load_detections_YOLO, YOLO_to_tracker, detections_to_YOLO, tracker_to_YOLO, detections_to_tracker, detections_to_tracker
from benchmarking.glimpse.Glimpse import Glimpse
from benchmarking.reducto.Reducto import Reducto
from benchmarking.videos.custom import CustomVideo
from fb_constants import video_fps
from tqdm import tqdm
from fb_constants import COCO_names
from sklearn.cluster import DBSCAN

def calculate_accuracy_list(preds_list, gt_list, names=COCO_names, metric='f1', niou=10):
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
        stats = []
        nc = 10 
        boundingBoxes = []
        # Read bounding boxes from pre-detection(x_min, y_min, x_max, y_max)
        preds = []
        iouv = torch.linspace(0.5, 0.95, niou)
        preds = torch.tensor(predBoxes)
        gtBoxes = torch.tensor(gtBoxes)
        labels = gtBoxes.clone()#torch.tensor(gtBoxes)
        # Metrics
        npr = preds.shape[0]
        correct = torch.zeros(npr, niou, dtype=torch.bool)  # init

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

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def box2native_tensor(image,box_list):
    """
    Retrun native scale (x1,y1,x2,y2) boxes (tensor).
    Arguments:
        box_list(np array): array of boxes 
    """
    box_list = np.array(box_list)
    imgHeight, imgWidth, _ = image.shape
    scale= [imgWidth,imgHeight,imgWidth,imgHeight]
    box_list[:,1:5] *= scale
    w = box_list[:,3]
    h = box_list[:,4]
    x_min = box_list[:,1] - w//2
    y_min = box_list[:,1] - h//2
    x_max = box_list[:,1] + w//2
    y_max = box_list[:,1] + h//2
     
    boxN_list = np.transpose([x_min, y_min, x_max, y_max])
    return torch.from_numpy(boxN_list)

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

def xcycwh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def xmymwh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] #- x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] #- x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] #/ 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] #/ 2  # bottom right y
    return y

def xyxy2xcycwh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y

def xyxy2xmymwh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] #+ x[..., 2]) / 2  # x center
    y[..., 1] = x[..., 1] #+ x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y

def frame_difference(old_frame, new_frame, thresh=35):
    old_frame_gray = cv2.cvtColor(old_frame,
                                      cv2.COLOR_BGR2GRAY)
    new_frame_gray = cv2.cvtColor(new_frame,
                                      cv2.COLOR_BGR2GRAY)
    """Compute the sum of pixel differences which are greater than thresh."""
    # thresh = 35 is used in Glimpse paper
    # pdb.set_trace()
    diff = np.absolute(new_frame_gray.astype(int) - old_frame_gray.astype(int))
    mask = np.greater(diff, thresh)
    pix_change = np.sum(mask)

    return pix_change

def remove_outliers(data, m=0.01):
    """
    Remove outliers using the Interquartile Range (IQR) method.
    
    :param data: List of values
    :param m: Multiplier for IQR (default is 1.5)
    :return: List with outliers removed
    """
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (m * iqr)
    upper_bound = q3 + (m * iqr)
    return [x for x in data if lower_bound <= x <= upper_bound]

def detect_regions(events, eps=50, min_samples=5):
    """
    Detect multiple regions using DBSCAN clustering.
    
    :param events: List of (frame, x, y) tuples
    :param eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    :param min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
    :return: List of regions, each represented as (x1, y1, x2, y2)
    """
    if not events:
        return []
    
    _, x_coords, y_coords = zip(*events)
    
    # Remove outliers while keeping x and y paired
    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)
    x_inliers = remove_outliers(x_coords)
    y_inliers = remove_outliers(y_coords)
    
    # Keep only points where both x and y are inliers
    inlier_mask = np.isin(x_coords, x_inliers) & np.isin(y_coords, y_inliers)
    x_coords = x_coords[inlier_mask]
    y_coords = y_coords[inlier_mask]
    
    points = np.column_stack((x_coords, y_coords))
    
    # Apply DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    
    regions = []
    for cluster in set(clustering.labels_):
        if cluster == -1:  # Skip noise points
            continue
        cluster_points = points[clustering.labels_ == cluster]
        x_min, y_min = np.min(cluster_points, axis=0)
        x_max, y_max = np.max(cluster_points, axis=0)
        regions.append((x_min, y_min, x_max, y_max))
    
    return regions

def detect_exit_regions(detection_list, frame_width, frame_height, border_threshold=0.05):
    """
    Detect regions where objects exit the frame, using a statistical approach to remove outliers.
    """
    exit_events = []
    previous_objects = set()
    
    border_x = frame_width * border_threshold
    border_y = frame_height * border_threshold
    
    for frame_idx, frame_detections in enumerate(detection_list):
        current_objects = set(tuple(obj) for obj in frame_detections)
        
        # Check for exiting objects
        for obj_id in previous_objects - current_objects:
            x1, y1, w, h = obj_id[1:5]
            center_x = x1 + w// 2
            center_y = y1 + h// 2
            x2 = x1 + w
            y2 = y1 + h
            
            
            if (x1 <= border_x or x2 >= frame_width - border_x or 
                y1 <= border_y or y2 >= frame_height - border_y):
                exit_events.append((frame_idx - 1, center_x, center_y))
        
        previous_objects = current_objects
    
    return detect_regions(exit_events)

def detect_emerge_regions(detection_list, frame_width, frame_height, border_threshold=0.05):
    """
    Detect regions where objects emerge in the frame, using a statistical approach to remove outliers.
    """
    emerge_events = []
    previous_objects = set()
    
    border_x = frame_width * border_threshold
    border_y = frame_height * border_threshold
    
    for frame_idx, frame_detections in enumerate(detection_list):
        current_objects = set()
        
        for obj in frame_detections:
            x1, y1, w, h = obj[1:5]
            center_x = x1 + w// 2
            center_y = y1 + h// 2
            x2 = x1 + w
            y2 = y1 + h
            
            obj_id = tuple(obj)
            current_objects.add(obj_id)
            
            # Check for emerging objects
            if obj_id not in previous_objects:
                if (x1 <= border_x or x2 >= frame_width - border_x or 
                    y1 <= border_y or y2 >= frame_height - border_y):
                    emerge_events.append((frame_idx, center_x, center_y))
        
        previous_objects = current_objects
    
    return detect_regions(emerge_events)


class analyzer:
    def __init__(self, vid_name, video_data_path, video_range, video_data ,trigger_criterion = 'fixed', target_acc =0.8, det_model_type='yolo',  preds_list=[], tracker_type = 'mvs', verbose= False):
        # load video data & propoerty 
        self.vid_name = vid_name
        self.video_path = video_data_path
        
        self.video_data = video_data
        
        self.frameHeight = self.video_data[0]['frame'].shape[0]
        self.frameWidth = self.video_data[0]['frame'].shape[1]
        self.fps =  video_fps[self.vid_name]
        self.target_acc = float(target_acc)
        # print('video fps:')
        print('video frame size:',self.frameWidth, self.frameHeight)
        self.warmup_result = {}
        # load detection data
        self.det_model_type = det_model_type
        self.preds_list = preds_list
        
        self.tracker_type = tracker_type  
        self.trigger_criterion = trigger_criterion
        self.benchmark_video = CustomVideo([self.frameWidth,self.frameHeight],self.fps,self.video_data, copy.deepcopy(self.preds_list))
        self.tracker = videoTracker(tracker_type=self.tracker_type)
    
        if self.trigger_criterion == 'fixed':
            self.pipeline = []
        elif self.trigger_criterion == 'glimpse':
            self.pipeline = Glimpse([20, 15, 10, 8, 5, 2],[1], self.tracker, 'glimpse.csv', target_f1= self.target_acc)
        elif self.trigger_criterion == 'reducto':
            self.pipeline = Reducto(self.benchmark_video,self.tracker,target_f1=self.target_acc)
        elif 'frameboost' in self.trigger_criterion:
            self.pipeline = []
        self.verbose = verbose
                
    def load_preds(self,frame_idx):
        """
        secretly load detection
        """
        pred = self.preds_list[frame_idx]
        if len(pred):
            pred_boxes = YOLO_to_tracker(np.copy(pred),self.frameHeight, self.frameWidth)
            #concat obj id, keep last max id and make obj id unique
            obj_ids = np.arange(self.tracker.next_id, self.tracker.next_id+len(pred_boxes)).reshape(-1,1)
            self.tracker.next_id += len(pred_boxes)
                            
            pred_boxes = np.concatenate((pred_boxes, obj_ids), axis=1)
        else:
            pred_boxes = np.empty(shape=(0, 7))
        
        return pred_boxes 
    
    def warmup(self, video_range, keyframe_setting ={}):
        print('warmup start')
        #profile detection keyframe trigger system
        start_idx = video_range[0]
        cutoff = video_range[1]
        if cutoff > 0:
            assert start_idx < cutoff
        
        frame_idx = 0
        
        #empty np array
        X = np.empty((0,2)) #np.empty((0,3))
        Y = np.empty((0,1))
        # differencer = Differencer.str2class(diff_feature_type)() 
        if self.trigger_criterion == 'fixed':
            for det_int in (range(self.fps, 0, -1)):
                det_idx_list = list(range(start_idx, cutoff, det_int))
                preds_list = []
                for frame_idx in range(video_range[0],video_range[1]):
                    frame = self.video_data[frame_idx]['frame']
                    motion_vectors = self.video_data[frame_idx]['motion_vectors']
                    frame_type = self.video_data[frame_idx]['frame_type']
                    if frame_idx in det_idx_list:
                        self.tracker.init_tracker(frame_idx, frame, self.load_preds(frame_idx))
                    else:
                        self.tracker.predict(frame, motion_vectors, frame_type)
                    preds_list.append(np.copy(self.tracker.get_boxes()))
                gt_list = [self.load_preds(i) for i in range(video_range[0],video_range[1])]
                f1 = calculate_accuracy_list(preds_list, gt_list,metric='f1')
                if f1 >= self.target_acc:
                    break
            self.warmup_result['det_int'] = det_int
                
        elif self.trigger_criterion == 'glimpse':
            best_frame_difference_threshold_divisor, \
                    best_tracking_error_threshold = self.pipeline.profile(
                        [], self.benchmark_video, video_range[0], video_range[1])
            self.warmup_result['thr'] = best_frame_difference_threshold_divisor
        
        elif self.trigger_criterion == 'reducto':
            best_feat_type, best_thresh, thresholds = self.pipeline.profile((video_range[0],video_range[1]))
            self.warmup_result['diff_feat'] = best_feat_type
            # self.warmup_result['thresholds'] = thresholds
        elif self.trigger_criterion == 'frameboost':
            redet_thr_list = np.arange(0, 20, 0.5).tolist()
            num_det_dict = {}
            accuracy_dict = {}
            
            X_dict = {}
            Y_dict = {}
            
            best_redet_thr = None
            for redet_thr in redet_thr_list:
                num_det_dict[redet_thr] = 0
                preds_list = []
                for frame_idx in range(video_range[0],video_range[1]):
                    frame = self.video_data[frame_idx]['frame']
                    motion_vectors = self.video_data[frame_idx]['motion_vectors']
                    frame_type = self.video_data[frame_idx]['frame_type']
                    
                    if frame_idx==start_idx:
                        num_det_dict[redet_thr]+=1 
                        self.tracker.init_tracker(frame_idx, frame, self.load_preds(frame_idx))
                        bSize_list = [(pred[-1], pred[3]*pred[4]) for pred in self.tracker.get_boxes()]  # (obj_id, size)
                        cum_shift_mags = {obj_id: 0 for obj_id, _ in bSize_list}
                        # prev_error_list = np.zeros(len(bSize_list))
                    else:
                        shifts = self.tracker.predict(frame, motion_vectors, frame_type)
                        shift_mags = np.linalg.norm(shifts, axis=1) if len(shifts) > 0 else [0]
                        
                        current_boxes = self.tracker.get_boxes()
                        bSize_list = [(pred[-1], pred[3]*pred[4]) for pred in current_boxes]  # (obj_id, size)
                        
                        # Update cumulative shift magnitudes
                        for (obj_id, _), shift_mag in zip(bSize_list, shift_mags):
                            if obj_id in cum_shift_mags:
                                cum_shift_mags[obj_id] += shift_mag
                            else:
                                cum_shift_mags[obj_id] = shift_mag

                        IoU_list, obj_id_list = self.tracker.get_iou_list(self.load_preds(frame_idx))
                        IoU_error_list = [1 - iou for iou in IoU_list]
                        tot_eIoU = np.sum(IoU_error_list)

                        iou_dict = dict(zip(obj_id_list, IoU_list))
                        
                        new_X_data = []
                        new_Y_data = []
                        for (obj_id, size) in bSize_list:
                            if obj_id in cum_shift_mags and obj_id in iou_dict:
                                new_X_data.append([size, cum_shift_mags[obj_id]])
                                new_Y_data.append([iou_dict[obj_id]])

                        if new_X_data and new_Y_data:
                            X = np.concatenate((X, np.array(new_X_data)), axis=0)
                            Y = np.concatenate((Y, np.array(new_Y_data)), axis=0)
                        
                        if tot_eIoU > redet_thr:
                            num_det_dict[redet_thr]+=1 
                            self.tracker.init_tracker(frame_idx, frame, self.load_preds(frame_idx))
                            bSize_list = [(pred[-1], pred[3]*pred[4]) for pred in self.tracker.get_boxes()]  # (obj_id, size)
                            cum_shift_mags = {obj_id: 0 for obj_id, _ in bSize_list}
                
                    current_boxes = self.tracker.get_boxes()
                    preds_list.append(np.copy(current_boxes))
                    
                preds_list = [preds[:,:-1] for preds in preds_list]
                gt_list = [self.load_preds(i) for i in range(video_range[0],video_range[1])]
                
                f1 = calculate_accuracy_list(preds_list, gt_list,metric='f1')
                accuracy_dict[redet_thr] = f1
                X_dict[redet_thr] = X
                Y_dict[redet_thr] = Y
            # mIoU_list=np.array(mIoU_list).reshape(-1,1)
            
            # get the redet_thr with maximum accuracy
            # traverse redet_thr and find accuracy over self.target_accurayc and among them, lowest number of detections
            min_det = video_range[1]-video_range[0]
            
            totX = np.empty((0,X.shape[1]))
            totY = np.empty((0,1))
            
            for redet_thr in redet_thr_list:
                totX = np.concatenate((totX, X_dict[redet_thr]), axis=0)
                totY = np.concatenate((totY, Y_dict[redet_thr]), axis=0)
                print(f'redet thr:{redet_thr} accuracy:{accuracy_dict[redet_thr]} numdet:{num_det_dict[redet_thr]}')
                if (accuracy_dict[redet_thr] >= self.target_acc and num_det_dict[redet_thr] < min_det):
                    min_det = num_det_dict[redet_thr]
                    best_redet_thr = redet_thr
            
            if best_redet_thr is None:
                best_redet_thr = max(accuracy_dict, key=accuracy_dict.get)
            print(best_redet_thr)
            polyreg = make_pipeline(PolynomialFeatures(2), LinearRegression())
            polyreg.fit(totX,totY)
            self.warmup_result['redet_thr'] = best_redet_thr
            self.warmup_result['predictor'] = polyreg

        elif self.trigger_criterion == 'frameboost_v3':
            redet_thr_list = np.arange(0, 20, 0.5).tolist()
            num_det_dict = {}
            accuracy_dict = {}
            
            X_dict = {}
            Y_dict = {}
            
            tmp_pred_list = []
            for frame_idx in range(video_range[0],video_range[1]):
                tmp_pred_list.append(self.load_preds(frame_idx))
            exit_regions = detect_exit_regions(tmp_pred_list, self.frameWidth, self.frameHeight)
            self.warmup_result = {'exit_regions': exit_regions}
            
            best_redet_thr = None
            for redet_thr in redet_thr_list:
                num_det_dict[redet_thr] = 0
                preds_list = []
                for frame_idx in range(video_range[0],video_range[1]):
                    frame = self.video_data[frame_idx]['frame']
                    motion_vectors = self.video_data[frame_idx]['motion_vectors']
                    frame_type = self.video_data[frame_idx]['frame_type']
                    
                    if frame_idx==start_idx:
                        num_det_dict[redet_thr]+=1 
                        self.tracker.init_tracker(frame_idx, frame, self.load_preds(frame_idx))
                        bSize_list = [(pred[-1], pred[3]*pred[4]) for pred in self.tracker.get_boxes()]  # (obj_id, size)
                        cum_shift_mags = {obj_id: 0 for obj_id, _ in bSize_list}
                        # prev_error_list = np.zeros(len(bSize_list))
                    else:
                        shifts = self.tracker.predict(frame, motion_vectors, frame_type)
                        shift_mags = np.linalg.norm(shifts, axis=1) if len(shifts) > 0 else [0]
                        
                        current_boxes = self.tracker.get_boxes()
                        bSize_list = [(pred[-1], pred[3]*pred[4]) for pred in current_boxes]  # (obj_id, size)
                        
                        # Update cumulative shift magnitudes
                        for (obj_id, _), shift_mag in zip(bSize_list, shift_mags):
                            if obj_id in cum_shift_mags:
                                cum_shift_mags[obj_id] += shift_mag
                            else:
                                cum_shift_mags[obj_id] = shift_mag

                        IoU_list, obj_id_list = self.tracker.get_iou_list(self.load_preds(frame_idx))      
                        IoU_error_list = [1 - iou for iou in IoU_list]
                        tot_eIoU = np.sum(IoU_error_list)

                        iou_dict = dict(zip(obj_id_list, IoU_list))
                        
                        new_X_data = []
                        new_Y_data = []
                        for (obj_id, size) in bSize_list:
                            if obj_id in cum_shift_mags and obj_id in iou_dict:
                                new_X_data.append([size, cum_shift_mags[obj_id]])
                                new_Y_data.append([iou_dict[obj_id]])
                        if new_X_data and new_Y_data:
                            X = np.concatenate((X, np.array(new_X_data)), axis=0)
                            Y = np.concatenate((Y, np.array(new_Y_data)), axis=0)
                        
                        if tot_eIoU > redet_thr:
                            num_det_dict[redet_thr]+=1 
                            self.tracker.init_tracker(frame_idx, frame, self.load_preds(frame_idx))
                            bSize_list = [(pred[-1], pred[3]*pred[4]) for pred in self.tracker.get_boxes()]  # (obj_id, size)
                            cum_shift_mags = {obj_id: 0 for obj_id, _ in bSize_list}
                
                    current_boxes = self.tracker.get_boxes()
                    preds_list.append(np.copy(current_boxes))
                    
                preds_list = [preds[:,:-1] for preds in preds_list]
                gt_list = [self.load_preds(i) for i in range(video_range[0],video_range[1])]
                
                f1 = calculate_accuracy_list(preds_list, gt_list,metric='f1')
                accuracy_dict[redet_thr] = f1
                X_dict[redet_thr] = X
                Y_dict[redet_thr] = Y

            min_det = video_range[1]-video_range[0]
            
            totX = np.empty((0,X.shape[1]))
            totY = np.empty((0,1))
            
            for redet_thr in redet_thr_list:
                totX = np.concatenate((totX, X_dict[redet_thr]), axis=0)
                totY = np.concatenate((totY, Y_dict[redet_thr]), axis=0)
                print(f'redet thr:{redet_thr} accuracy:{accuracy_dict[redet_thr]} numdet:{num_det_dict[redet_thr]}')
                if (accuracy_dict[redet_thr] >= self.target_acc and num_det_dict[redet_thr] < min_det):
                    min_det = num_det_dict[redet_thr]
                    best_redet_thr = redet_thr
            
            if best_redet_thr is None:
                best_redet_thr = max(accuracy_dict, key=accuracy_dict.get)
            print(best_redet_thr)
            polyreg = make_pipeline(PolynomialFeatures(2), LinearRegression())
            polyreg.fit(totX,totY)
            self.warmup_result['redet_thr'] = best_redet_thr
            self.warmup_result['predictor'] = polyreg


    def dryrun(self, video_range, short_video_length=30, keyframe_setting = {}): 
        print('dryrun start')
        self.key_frames = []
        self.thr_list = []
        tot_profile_t = 0
        start_idx = video_range[0]

        if self.trigger_criterion == 'reducto':
            _, perfs, selected_frames, thr_list, diff_t, predict_t  = self.pipeline.evaluate(self.warmup_result['diff_feat'], (video_range[0], video_range[1]))
            print(f'diff_t: {diff_t}, predict_t: {predict_t}')
            self.key_frames += selected_frames
            self.thr_list += thr_list
            print(thr_list)
            return diff_t + predict_t
        # elif self.trigger_criterion == 'glimpse':
        
        #iterate through the video using dryrun_one_segment // short_video_length
        else:
            for seg_idx in tqdm(range((video_range[1]-video_range[0])//short_video_length)):
                seg_start = start_idx+seg_idx*short_video_length
                seg_end = seg_start + short_video_length
                seg_prof_t = self.dryrun_one_segment((seg_start, seg_end), keyframe_setting)
                tot_profile_t += seg_prof_t
            return tot_profile_t

    def dryrun_one_segment(self, video_range, keyframe_setting = {}): 
        start_idx = video_range[0]            
        # trigger_criterion = keyframe_setting['criterion']
        
        # assert trigger_criterion in ['fixed','glimpse','reducto','frameboost']
        
        if self.trigger_criterion == 'fixed':
            diff_feature_type = 'pixel'
            det_interval = self.warmup_result['det_int']
            self.key_frames += list(range(start_idx,video_range[1],det_interval))
            return 0 #no profile time needed
        if self.trigger_criterion == 'glimpse':
            diff_feature_type = 'pixel'
            thr_list = self.warmup_result['thr']
        if self.trigger_criterion == 'reducto':
            diff_feature_type = self.warmup_result['diff_feat']
            # thr_list = self.warmup_result['thr_list']

        if 'frameboost' in self.trigger_criterion:
            diff_feature_type = keyframe_setting['diff_feat']
            iouPredictor = self.warmup_result['predictor'] 
            redet_thr = self.warmup_result['redet_thr']        

        diff_feature_list = []
        differencer = Differencer.str2class(diff_feature_type)()  
        # profile
        tot_profile_t= 0
        pred_error_list = []
        for frame_idx in range(video_range[0],video_range[1]):
            frame = self.video_data[frame_idx]['frame']
            motion_vectors = self.video_data[frame_idx]['motion_vectors']
            frame_type = self.video_data[frame_idx]['frame_type']

            frameHeight, frameWidth, _ = frame.shape
            
            if frame_idx==start_idx: #% fps  == 0: #frame_type=='I': #  
                self.key_frames.append(frame_idx)
                self.tracker.init_tracker(frame_idx, frame, self.load_preds(frame_idx))
                
                diff_feature_list.append(frame) 
                
                bSize_list = [(pred[-1], pred[3]*pred[4]) for pred in self.tracker.get_boxes()]  # (obj_id, size)
                cum_shifts = {obj_id: (0,0) for obj_id, _ in bSize_list}
                cum_shift_mags = {obj_id: 0 for obj_id, _ in bSize_list}
                # diff_feature = differencer.get_frame_feature(frame)   
            else:
                #track
                shifts = self.tracker.predict(frame, motion_vectors, frame_type)
                shift_mags = np.linalg.norm(shifts, axis=1) if len(shifts) > 0 else [0]
                
                #profile start
                diff_feature_list.append(frame)
                profile_start_t = time.time()
                if self.trigger_criterion == 'glimpse' and len(diff_feature_list)>1: 
                    thr = thr_list
                    diff = frame_difference(diff_feature_list[0],diff_feature_list[-1])
                    if diff > (frameHeight*frameWidth)//thr: #trigger threshold
                        diff_feature_list = []
                        diff_feature_list.append(frame)  
                        self.key_frames.append(frame_idx)
                    profile_end_t = time.time()
                
                elif self.trigger_criterion == 'frameboost':
                    current_boxes = self.tracker.get_boxes()
                    bSize_list = [(pred[-1], pred[3]*pred[4]) for pred in current_boxes]  # (obj_id, size)

                    # Update cumulative shift magnitudes
                    for (obj_id, _), shift_mag in zip(bSize_list, shift_mags):
                        if obj_id in cum_shift_mags:
                            cum_shift_mags[obj_id] += shift_mag
                        else:
                            cum_shift_mags[obj_id] = shift_mag
                    
                    # Prepare data for X and Y, ensuring object IDs match   
                    X_data = []        
                    for (obj_id, size) in bSize_list:
                        if obj_id in cum_shift_mags:
                            X_data.append([size, cum_shift_mags[obj_id]])

                    if not X_data:
                        breakpoint()
                        
                    IoU_pred_list = iouPredictor.predict(np.array(X_data))
                    IoU_error_list = [1 - iou for iou in IoU_pred_list]
                    tot_pred_error = np.sum(IoU_error_list)
                    # print(tot_pred_error)
                    
                    pred_error_list.append(tot_pred_error)
                    profile_end_t = time.time()

                    if tot_pred_error > redet_thr: #trigger_thr: #trigger threshold
                        self.key_frames.append(frame_idx)
                        self.tracker.init_tracker(frame_idx, frame, self.load_preds(frame_idx))
                        bSize_list = [(pred[-1], pred[3]*pred[4]) for pred in self.tracker.get_boxes()]  # (obj_id, size)
                        cum_shift_mags = {obj_id: 0 for obj_id, _ in bSize_list}
                        
                elif self.trigger_criterion == 'frameboost_v3':
                    exit_regions = self.warmup_result['exit_regions']
                    current_boxes = self.tracker.get_boxes()
                    bSize_list = [(pred[-1], pred[3]*pred[4]) for pred in current_boxes]  # (obj_id, size)

                    # Update cumulative shift magnitudes
                    for (obj_id, _), shift_mag in zip(bSize_list, shift_mags):
                        if obj_id in cum_shift_mags:
                            cum_shift_mags[obj_id] += shift_mag
                        else:
                            cum_shift_mags[obj_id] = shift_mag
                    
                    # Prepare data for X and Y, ensuring object IDs match   
                    X_data = []        
                    obj_id_list = []
                    for (obj_id, size) in bSize_list:
                        if obj_id in cum_shift_mags:
                            obj_id_list.append(obj_id)
                            X_data.append([size, cum_shift_mags[obj_id]])

                    if not X_data:
                        breakpoint()
                        
                    IoU_pred_list = iouPredictor.predict(np.array(X_data))
                    
                    clean_IoU_list = []
                    margin = 15
                    for i, (IoU, obj_id) in enumerate(zip(IoU_pred_list, obj_id_list)):
                        x1, y1, w, h = current_boxes[i][1:5]
                        # for exit in exit_regions: #check if the object is in the exit region, exit region xyxy
                        #     if x1 >= exit[0] and x1 <= exit[2] and y1 >= exit[1] and y1 <= exit[3]:
                        #         exit = True
                        #         breakpoint()
                        #         break
                        x2 = x1 + w
                        y2 = y1 + h
                        if self.vid_name == 'highway':
                            exit = (y1 <= 0 + margin or y2 >= self.frameHeight-margin)
                        elif self.vid_name == 'crossroad':
                            exit = (x1 <= 0 + margin or x2 >= self.frameWidth - margin)
                        elif self.vid_name == 'motorway':
                            exit = (y1 <= 0 + margin or y2 >= self.frameHeight-margin)
                        # exit = (y2 >= self.frameHeight-margin)
                        # print(x1,y1,w,h)
                        # print(exit)
                        # breakpoint()
                        if exit:
                            continue
                        clean_IoU_list.append(IoU)

                    IoU_error_list = [1 - iou for iou in clean_IoU_list]
                    tot_pred_error = np.sum(IoU_error_list)
                    # print(tot_pred_error)
                    
                    pred_error_list.append(tot_pred_error)
                    profile_end_t = time.time()

                    if tot_pred_error > redet_thr: #trigger_thr: #trigger threshold
                        self.key_frames.append(frame_idx)
                        self.tracker.init_tracker(frame_idx, frame, self.load_preds(frame_idx))
                        bSize_list = [(pred[-1], pred[3]*pred[4]) for pred in self.tracker.get_boxes()]  # (obj_id, size)
                        cum_shift_mags = {obj_id: 0 for obj_id, _ in bSize_list}
                           
                tot_profile_t += profile_end_t - profile_start_t
            
        return tot_profile_t
                                  
    def run(self, video_range, short_video_length = 30, visualize=False):    
        print('run start')
        start_idx = video_range[0]
        result_dict = {}
        #iterate through the video using dryrun_one_segment // short_video_length
        for seg_idx in tqdm(range((video_range[1]-video_range[0])//short_video_length)):
            seg_start = start_idx+seg_idx*short_video_length
            seg_end = seg_start + short_video_length
            box_list_det, num_det, det_frames, tracking_time = self.run_one_segment((seg_start, seg_end),visualize=visualize)
            result_dict[seg_idx] = {'boxes':box_list_det, 'num_det':num_det, 'det_frames':det_frames, 'tracking_time':tracking_time}
        return result_dict

    def run_one_segment(self, video_range, visualize=False):
        self.tracker.reset_tracking_time()
        num_det = 0
        frame_list = []
        det_frames = []
        
        box_size_list = []
        obj_id_list = []
        iou_list = []
        
        seg_key_frames = [frame for frame in self.key_frames if frame in range(video_range[0], video_range[1])]
        # self.key_frames[self.key_frames in range(video_range[0],video_range[1])]
        frame_idx = 0
        box_list_tracker = []
        box_list_det = []
        for frame_idx in range(video_range[0],video_range[1]):
            frame = self.video_data[frame_idx]['frame']
            frame_list.append(frame)
            motion_vectors = self.video_data[frame_idx]['motion_vectors']
            frame_type = self.video_data[frame_idx]['frame_type']
            frameHeight, frameWidth, _ = frame.shape
            
            if frame_idx in seg_key_frames:
                pred_boxes = self.load_preds(frame_idx)
                #detection
                num_det += 1
                det_frames.append(frame_idx) 
                self.tracker.init_tracker(frame_idx, frame, pred_boxes)
                box_list_tracker.append((np.copy(pred_boxes)))
            else:
                #track
                _ = self.tracker.predict(frame, motion_vectors, frame_type)
                box_list_tracker.append(np.copy(self.tracker.get_boxes()))
        
        box_list_tracker = [box_list[:,:6] for box_list in box_list_tracker]
        box_list_tmp = []        
        for box_list in box_list_tracker:
            if len(box_list) > 0:
                box_list_tmp.append(box_list[:,:6])
            else:
                box_list_tmp.append([])
        box_list_tracker = box_list_tmp
        
        if visualize: 
            for fidx in range(video_range[1]-video_range[0]):
                self.visualize(frame_list[fidx], 'baseline',fidx+video_range[0], preds= box_list_tracker[fidx], threshold=0.3)
        #post process
        for boxes in box_list_tracker:
            box_list_det.append(tracker_to_YOLO(np.copy(boxes), frameHeight, frameWidth))
        return box_list_det, num_det, det_frames, self.tracker.get_total_tracking_time()

    def study(self, video_range, visualize=False):
        self.tracker.reset_tracking_time()
        num_det = 0
        frame_list = []
        det_frames = []
        
        box_size_list = []
        obj_id_list = []
        trigger_iou_list = []
        
        seg_key_frames = [frame for frame in self.key_frames if frame in range(video_range[0], video_range[1])]
        # self.key_frames[self.key_frames in range(video_range[0],video_range[1])]
        frame_idx = 0
        box_list_tracker = []
        box_list_det = []
        for frame_idx in range(video_range[0],video_range[1]):
            frame = self.video_data[frame_idx]['frame']
            frame_list.append(frame)
            motion_vectors = self.video_data[frame_idx]['motion_vectors']
            frame_type = self.video_data[frame_idx]['frame_type']
            frameHeight, frameWidth, _ = frame.shape
            if frame_idx in seg_key_frames:
                pred_boxes = self.load_preds(frame_idx)
                #detection
                num_det += 1
                det_frames.append(frame_idx) 
                self.tracker.init_tracker(frame_idx, frame, pred_boxes)
                box_list_tracker.append((np.copy(pred_boxes)))
            else:
                #track 
                _ = self.tracker.predict(frame, motion_vectors, frame_type)
                box_list_tracker.append(np.copy(self.tracker.get_boxes()))
                if frame_idx in np.array(seg_key_frames) -1:
                    #get trigger iou
                    IoU_list, obj_id=_list = self.tracker.get_iou_list(self.load_preds(frame_idx))
                    trigger_iou_list+=IoU_list
        box_list_tracker = [box_list[:,:6] for box_list in box_list_tracker]        
        if visualize: 
            for fidx in range(video_range[1]-video_range[0]):
                self.visualize(frame_list[fidx], 'baseline',fidx+video_range[0], preds= box_list_tracker[fidx], threshold=0.3)
        
        return trigger_iou_list

    def study2(self, video_range, visualize=False):
        boxes_dict = {}
        self.tracker.reset_tracking_time()
        num_det = 0
        frame_list = []
        det_frames = []

        trigger_iou_list = []
        
        seg_key_frames = [frame for frame in self.key_frames if frame in range(video_range[0], video_range[1])]
        # self.key_frames[self.key_frames in range(video_range[0],video_range[1])]
        frame_idx = 0
        box_list_tracker = []
        box_list_det = []
        for frame_idx in range(video_range[0],video_range[1]):
            frame = self.video_data[frame_idx]['frame']
            frame_list.append(frame)
            motion_vectors = self.video_data[frame_idx]['motion_vectors']
            frame_type = self.video_data[frame_idx]['frame_type']
            frameHeight, frameWidth, _ = frame.shape
            if frame_idx in seg_key_frames:
                pred_boxes = self.load_preds(frame_idx)
                for box in pred_boxes:
                    if box[-1] not in boxes_dict:
                        boxes_dict[box[-1]] = {}
                        boxes_dict[box[-1]]['size'] =  [box[3]*box[4]]
                        boxes_dict[box[-1]]['iou'] = [1]
                #detection
                num_det += 1
                det_frames.append(frame_idx) 
                self.tracker.init_tracker(frame_idx, frame, pred_boxes)
                box_list_tracker.append((np.copy(pred_boxes)))
            else:
                #track 
                _ = self.tracker.predict(frame, motion_vectors, frame_type)
                box_list_tracker.append(np.copy(self.tracker.get_boxes()))
                IoU_list, obj_id=_list = self.tracker.get_iou_list(self.load_preds(frame_idx))
                current_boxes = self.tracker.get_boxes()
                for box in current_boxes:
                    if box[-1] not in boxes_dict:
                        if len(box):
                            boxes_dict[box[-1]] = {}
                            boxes_dict[box[-1]]['size'] = [box[3]*box[4]]
                            boxes_dict[box[-1]]['iou'] = [1]
                    else:
                        if len(box):
                            boxes_dict[box[-1]]['size'].append(box[3]*box[4])
                            boxes_dict[box[-1]]['iou'].append(IoU_list[obj_id.index(box[-1])])
                        
                trigger_iou_list+=IoU_list
        box_list_tracker = [box_list[:,:6] for box_list in box_list_tracker]        
        if visualize: 
            for fidx in range(video_range[1]-video_range[0]):
                self.visualize(frame_list[fidx], 'baseline',fidx+video_range[0], preds= box_list_tracker[fidx], threshold=0.3)
        
        return boxes_dict

    def visualize(self,image, dir, idx, preds= [], threshold=0.3): 
        def loadBox2Patches(image, preds, box_color):
            # Calculate the coordinates of the box's corners
            assert len(preds)>0, "No detection"
            imgHeight, imgWidth, _ = image.shape
            list_patch = []
            for idx, box in enumerate(preds):
                x_min = max(0,int(box[1]))
                y_min = max(0,int(box[2]))
                w = int(box[3])
                h = int(box[4])

                # patch = patches.Rectangle((x_min, y_min), w, h, linewidth=1.5, edgecolor=box_color, facecolor='none')
                patch = patches.Rectangle((x_min, y_min), w, h, linewidth=1.5, edgecolor=box_color, facecolor=box_color, alpha =0.5)
                list_patch.append(patch)
            return list_patch

        img_height, img_width = image.shape[:2]
        # DPI (dots per inch) for the figure
        dpi = 100  # You can adjust this value to your preference
        # Create a figure with the same pixel size as the image
        fig, ax = plt.subplots(figsize=(img_width/dpi, img_height/dpi), dpi=dpi)
        ax.imshow(image)
        det_box_list = preds[preds[:,-1]>=threshold]
        if len(det_box_list)>0:
            det_patches = loadBox2Patches(image,det_box_list,'green')
            for patch in det_patches:
                ax.add_patch(patch)
        # Remove axes and padding
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        # Save the figure to a file
        if not os.path.exists(f'./visualize_{dir}'):
            os.makedirs(f'./visualize_{dir}')
        plt.savefig(f'./visualize_{dir}/{idx}.jpg', dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close(fig)  # Close the figure to free up memory


    __call__ = run
##########################################################################################################################################
