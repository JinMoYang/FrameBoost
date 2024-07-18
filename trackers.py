import cv2
import numpy as np
import time
from mvt.tracker import MotionVectorTracker
import scipy
from scipy import optimize
class videoTracker:
    def __init__(self, tracker_type):
        self.tracker_type = tracker_type
        self.tracker_initialized = False
        self.tracker = None
        self.total_tracking_time = 0
        
        if tracker_type == 'mvt':
            self.tracker = MotionVectorTracker()
        elif tracker_type == 'kcf':
            self.tracker = kcfTracker()
        elif tracker_type == 'lk':
            self.tracker = LucasKanadeTracker()
        self.next_id = 0
        
    def init_tracker(self, frame_idx, frame, boxes):
        if self.tracker_type == 'mvt':
            # self.tracker.init_tracker(boxes)
            self.tracker.boxes = boxes
        elif self.tracker_type == 'kcf':
            self.tracker.init_trackers(frame_idx, frame, boxes)
        elif self.tracker_type == 'lk':
            self.tracker.init_tracker(frame, boxes)
        self.tracker_initialized = True
    
    def init_tracker_benchmark(self, frame_idx, frame, boxes):
        #different form of box. match box first from (x,y,x,y,cls,conf,obj_id) to (cls,xc,yc,w,h,score,obj_id)
        converted_boxes = []
        for box in boxes:
            x1, y1, x2, y2, cls, conf, obj_id = box
            xc = (x1 + x2) / 2
            yc = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1
            new_box = [cls, xc, yc, w, h, conf, obj_id]
            converted_boxes.append(new_box)
        
        
        if self.tracker_type == 'mvt':
            # self.tracker.init_tracker(boxes)
            self.tracker.boxes = converted_boxes
        elif self.tracker_type == 'kcf':
            self.tracker.init_trackers(frame_idx, frame, converted_boxes)
        elif self.tracker_type == 'lk':
            self.tracker.init_tracker(frame, converted_boxes)
        self.tracker_initialized = True

    def predict(self, frame, motion_vectors=[], frame_type='I'):
        if not self.tracker_initialized:
            raise ValueError("Tracker is not initialized. Call init_tracker first.")

        start_time = time.time()
        shifts = self.tracker.predict(frame, motion_vectors, frame_type)
        self.total_tracking_time += time.time() - start_time
        return shifts

    def get_boxes(self):
        return self.tracker.get_boxes()
    
    def get_benchmark_boxes(self):
        #different form of box. match box first from (cls,xc,yc,w,h,score,obj_id) back to (x,y,x,y,cls,conf,obj_id)  
        boxes = self.tracker.get_boxes()
        converted_boxes = []
        for box in boxes:
            cls, xc, yc, w, h, score, obj_id = box
            x1 = xc - w / 2
            x2 = xc + w / 2
            y1 = yc - h / 2
            y2 = yc + h / 2
            new_box = [x1, y1, x2, y2, cls, score, obj_id]
            converted_boxes.append(new_box)
        return converted_boxes
    
    def get_iou_list(self, detected_boxes):
        return self.tracker.compute_iou_with_detections(detected_boxes)

    def reset_tracking_time(self):
        self.total_tracking_time = 0

    def get_total_tracking_time(self):
        return self.total_tracking_time

class kcfTracker:
    def __init__(self):
        return
    
    def init_trackers(self, frame_idx, frame, boxes):
        self.trackers_dict = {}
        self.boxes = None
        self.boxes = boxes
        resolution = (frame.shape[1], frame.shape[0])
        frame_copy = cv2.resize(frame, (640, 480))
        for box in self.boxes:
            t, xmin, ymin, w, h, score, obj_id = box
            tracker = cv2.TrackerKCF_create()
            x_min = int(xmin * 640 / resolution[0])
            y_min = int(ymin * 480 / resolution[1])
            w = int(w * 640 / resolution[0])
            h = int(h * 480 / resolution[1])
            tracker.init(frame_copy, (x_min, y_min, w, h))
            key = f"{int(t)}_{int(obj_id)}_{int(t)}"
            self.trackers_dict[key] = tracker

    def predict(self, frame, motion_vectors=None, frame_type=None):
        # print(f"Number of boxes: {len(self.boxes)}")
        # print(f"Number of trackers: {len(self.trackers_dict)}")
        resolution = (frame.shape[1], frame.shape[0])
        frame_copy = cv2.resize(frame, (640, 480))
        new_boxes = []
        shift_list = []
        # breakpoint()
        for box in self.boxes:
            t, x, y, w, h, score, obj_id = box
            key = f"{int(t)}_{int(obj_id)}_{int(t)}"
            tracker = self.trackers_dict.get(key)
            
            if tracker is None:
                raise ValueError(f"No tracker found for box {box}. This should not happen.")
            
            ok, bbox = tracker.update(frame_copy)
            
            if ok:
                new_x, new_y, new_w, new_h = bbox
                new_x = int(new_x * resolution[0] / 640)
                new_y = int(new_y * resolution[1] / 480)
                new_w = int(new_w * resolution[0] / 640)
                new_h = int(new_h * resolution[1] / 480)
                
                dx = new_x - x
                dy = new_y - y
                
                new_boxes.append([t, new_x, new_y, new_w, new_h, score, obj_id])
                shift_list.append([dx, dy])
            else:
                # If tracking fails, keep the old box and add zero shift
                # new_boxes.append(box)
                shift_list.append([0, 0])
                # print('i am deleting trackers dict')
                # breakpoint()
                del self.trackers_dict[key]

        self.boxes = np.array(new_boxes)
        return shift_list

    def compute_iou_with_detections(self, detected_boxes):
            return get_IoU_list(self.boxes, detected_boxes)

    def get_boxes(self):
            return self.boxes

class LucasKanadeTracker:
    def __init__(self, lk_params=None, feature_params=None):
        self.lk_params = lk_params if lk_params is not None else dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        self.feature_params = feature_params if feature_params is not None else dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )
        
    def init_tracker(self, frame, bounding_boxes):
        self.old_gray = None
        self.points = None
        self.boxes = None

        self.old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.boxes = bounding_boxes
        self.points = self._initialize_points(frame, bounding_boxes)

    def _initialize_points(self, frame, bounding_boxes):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        points = []
        for box in bounding_boxes:
            x, y, w, h = map(int, box[1:5])
            roi = gray[y:y+h, x:x+w]
            p = cv2.goodFeaturesToTrack(roi, mask=None, **self.feature_params)
            if p is not None:
                p[:, 0, 0] += x
                p[:, 0, 1] += y
                points.append(p)
        return np.concatenate(points) if points else np.array([])

    def predict(self, frame, motion_vectors=None, frame_type=None):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.points is None or len(self.points) == 0:
            print("No points to track. Reinitializing points.")
            self.points = self._initialize_points(frame, self.boxes)
            if len(self.points) == 0:
                print("Failed to initialize points. Returning original boxes.")
                return self.boxes, [[0, 0]] * len(self.boxes)
        
        new_points, status, _ = cv2.calcOpticalFlowPyrLK(self.old_gray, frame_gray, self.points, None, **self.lk_params)
        
        if new_points is None:
            print("Optical flow failed. Returning original boxes.")
            return self.boxes, [[0, 0]] * len(self.boxes)
        
        good_new = new_points[status == 1]
        good_old = self.points[status == 1]
        
        new_boxes = []
        shift_list = []
        
        for i, box in enumerate(self.boxes):
            t, x, y, w, h, score, obj_id = box
            x, y, w, h = map(int, [x, y, w, h])
            
            mask = (good_old[:, 0] >= x) & (good_old[:, 0] < x + w) & \
                   (good_old[:, 1] >= y) & (good_old[:, 1] < y + h)
            
            if np.any(mask):
                dx = np.median(good_new[mask, 0] - good_old[mask, 0])
                dy = np.median(good_new[mask, 1] - good_old[mask, 1])
                
                new_x = x + dx
                new_y = y + dy
                
                new_boxes.append([t, new_x, new_y, w, h, score, obj_id])
                shift_list.append([dx, dy])
            else:
                new_boxes.append(box)
                shift_list.append([0, 0])
        
        self.boxes = np.array(new_boxes)
        self.old_gray = frame_gray.copy()
        self.points = good_new.reshape(-1, 1, 2)
        
        return shift_list

    def compute_iou_with_detections(self, detected_boxes):
            return get_IoU_list(self.boxes, detected_boxes)

    def get_boxes(self):
        return self.boxes

def get_IoU_list(tracked_boxes, detected_boxes):
    """Matches detection boxes with tracked boxes based on IoU.

    Args:
        tracked_boxes (`numpy.ndarray` or None): Array of shape (T, 7) and dtype float of the
            T tracked bounding boxes in the format [frame, xmin, ymin, width, height, ..., object_id].
            Can be None if no tracked boxes exist.

        detected_boxes (`numpy.ndarray`): Array of shape (D, 6) and dtype float of the
            D detected bounding boxes in the format [frame, xmin, ymin, width, height, ...].

    Returns:
        tuple: (IoU_list, object_id_list)
            IoU_list (`list`): List of IoU values for each tracked box. If a tracked box has no match,
                its IoU value is 0. If there are no tracked boxes, an empty list is returned.
            object_id_list (`list`): List of object IDs corresponding to the IoU list.
    """
    if tracked_boxes is None or len(tracked_boxes) == 0:
        return [], []

    IoU_list = []
    object_id_list = []
    t_boxes = tracked_boxes[:, 1:5]
    d_boxes = detected_boxes[:, 1:5]
    object_ids = tracked_boxes[:, -1]  # Last column contains object IDs

    # compute IoU matrix for all possible matches of tracking and detection boxes
    iou_matrix = np.zeros([len(t_boxes), len(d_boxes)])
    for t, t_box in enumerate(t_boxes):
        for d, d_box in enumerate(d_boxes):    
            iou_matrix[t, d] = compute_iou(t_box, d_box)

    # find matches between detection and tracking boxes that lead to maximum total IoU
    t_idx, d_idx = scipy.optimize.linear_sum_assignment(-iou_matrix)

    for tracker_idx in range(t_boxes.shape[0]):
        if tracker_idx in t_idx:
            match_idx = np.where(t_idx == tracker_idx)[0][0]
            IoU_list.append(iou_matrix[t_idx[match_idx], d_idx[match_idx]])
        else:
            IoU_list.append(0)
        object_id_list.append(object_ids[tracker_idx])

    #if IoU list has NaN
    for i in range(len(IoU_list)):
        if np.isnan(IoU_list[i]):
            breakpoint()
    
    assert len(IoU_list) == tracked_boxes.shape[0], 'Error: IoU_list length does not match the number of tracked boxes'
    assert len(object_id_list) == tracked_boxes.shape[0], 'Error: object_id_list length does not match the number of tracked boxes'
    
    return IoU_list, object_id_list

def compute_iou(boxA, boxB):
    """Computes the Intersection over Union (IoU) for two bounding boxes.

    Args:
        boxA, boxB (`numpy.ndarray`): Bounding boxes [xmin, ymin, width, height]
            as arrays with shape (4,) and dtype float.

    Returns:
        IoU (`float`): The IoU of the two boxes. It is within the range [0, 1],
        0 meaning no overlap and 1 meaning full overlap of the two boxes.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    boxAArea = abs(boxA[2] * boxA[3])
    boxBArea = abs(boxB[2] * boxB[3])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou