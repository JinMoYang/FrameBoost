import time
import uuid

import numpy as np
import cv2
import pickle
import math 
from mvt import trackerlib

class MotionVectorTracker:
    def __init__(self, iou_threshold = 0.3, det_conf_threshold = 0.3,
        state_thresholds = (0,1,10), use_only_p_vectors=False, use_kalman=False,
        use_numeric_ids=False, measure_timing=False):
        self.iou_threshold = iou_threshold
        self.det_conf_threshold = det_conf_threshold
        self.use_only_p_vectors = use_only_p_vectors
        self.use_kalman = use_kalman
        self.boxes = np.empty(shape=(0, 6))
        self.box_ids = []
        self.last_motion_vectors = np.empty(shape=(0, 10))
        self.next_id = 1
        self.track_count = 0
        if self.use_kalman:
            self.filters = []
        self.use_numeric_ids = use_numeric_ids

        self.state_counters = {"missed": [], "redetected": []}
        self.target_states = []

        # target state transition thresholds
        self.pending_to_confirmed_thres = state_thresholds[0]
        self.confirmed_to_pending_thres = state_thresholds[1]
        self.pending_to_deleted_thres = state_thresholds[2]

        # for timing analaysis
        self.measure_timing = measure_timing
        self.last_predict_dt = np.inf
        self.last_update_dt = np.inf

    def _filter_low_confidence_detections(self, detection_boxes):
        idx = np.nonzero(detection_boxes[:,-1] >= self.det_conf_threshold)
        detection_boxes[idx]
        return detection_boxes[idx]

    def update(self, motion_vectors, frame_type, detection_boxes):
        if self.measure_timing:
            start_update = time.perf_counter()

        # remove detections with confidence lower than det_conf_threshold
        if self.det_conf_threshold is not None:
            detection_boxes = self._filter_low_confidence_detections(detection_boxes)
        # bring boxes into next state
        self.predict([],motion_vectors, frame_type)

        # match predicted (tracked) boxes with detected boxes
        matches, unmatched_trackers, unmatched_detectors = trackerlib.match_bounding_boxes(self.boxes, detection_boxes, self.iou_threshold)

        # handle matches by incremeting the counter for redetection and resetting the one for lost
        for d, t in matches:
            self.state_counters["missed"][t] = 0  # reset lost counter
            self.state_counters["redetected"][t] += 1  # increment redetection counter
            if self.use_kalman:
                 self.filters[t].predict()
                 self.filters[t].update(detection_boxes[d])
                 self.boxes[t] = self.filters[t].get_box_from_state()
            else:
                self.boxes[t] = detection_boxes[d]
            # update target state based on counter values
            if self.state_counters["redetected"][t] >= self.pending_to_confirmed_thres:
                self.target_states[t] = "confirmed"


        # handle unmatched detections by spawning new trackers in pending state
        for d in unmatched_detectors:
            self.state_counters["missed"].append(0)
            self.state_counters["redetected"].append(0)
            if self.pending_to_confirmed_thres > 0:
                self.target_states.append("pending")
            elif self.pending_to_confirmed_thres == 0:
                self.target_states.append("confirmed")
            if self.use_numeric_ids:
                self.box_ids.append(self.next_id)
                self.next_id += 1
            else:
                uid = uuid.uuid4()
                self.box_ids.append(uid)
            self.boxes = np.vstack((self.boxes, detection_boxes[d]))
            if self.use_kalman:
                 filter = trackerlib.Kalman()
                 filter.set_initial_state(detection_boxes[d])
                 self.filters.append(filter)

        # handle unmatched tracker predictions by counting how often a target got lost subsequently
        for t in unmatched_trackers:
            self.state_counters["missed"][t] += 1
            self.state_counters["redetected"][t] = 0
            # if target is not redetected for confirmed_to_pending_thres cosecutive times set its state to pending
            if self.state_counters["missed"][t] > self.confirmed_to_pending_thres:
                self.target_states[t] = "pending"
            #   if target is not redetected for pending_to_deleted_thres cosecutive times delete it
            if self.state_counters["missed"][t] > self.pending_to_deleted_thres:
                self.boxes = np.delete(self.boxes, t, axis=0)
                self.box_ids.pop(t)
                self.state_counters["missed"].pop(t)
                self.state_counters["redetected"].pop(t)
                self.target_states.pop(t)
                if self.use_kalman:
                    self.filters.pop(t)

        if self.measure_timing:
            self.last_update_dt = time.perf_counter() - start_update

    def find_matches(self, motion_vectors, frame_type, detection_boxes):
        if self.measure_timing:
            start_update = time.perf_counter()

        # remove detections with confidence lower than det_conf_threshold
        # if self.det_conf_threshold is not None:
        #     detection_boxes = self._filter_low_confidence_detections(detection_boxes)
        # bring boxes into next state
        
        self.predict([], motion_vectors, frame_type)
        
        # match predicted (tracked) boxes with detected boxes
        matches, unmatched_trackers, unmatched_detectors = trackerlib.match_bounding_boxes(self.boxes, detection_boxes, self.iou_threshold)   
        if self.measure_timing:
            self.last_update_dt = time.perf_counter() - start_update
            
        return matches
    
    def compute_iou_with_detections(self, detection_boxes):
        IoU_list = trackerlib.match_bounding_boxes_IoU(self.boxes, detection_boxes)   
        return IoU_list

    def undetected_tracekd_box(self, motion_vectors, frame_type, detection_boxes):
        if self.measure_timing:
            start_update = time.perf_counter()

        # remove detections with confidence lower than det_conf_threshold
        # if self.det_conf_threshold is not None:
        #     detection_boxes = self._filter_low_confidence_detections(detection_boxes)
        # bring boxes into next state
        
        self.predict([],motion_vectors, frame_type)
        
        # match predicted (tracked) boxes with detected boxes
        matches, unmatched_trackers, unmatched_detectors = trackerlib.match_bounding_boxes(self.boxes, detection_boxes, self.iou_threshold)   

        if self.measure_timing:
            self.last_update_dt = time.perf_counter() - start_update
            
        return unmatched_trackers

    def unmatched_detected_box (self, motion_vectors, frame_type, detection_boxes):
        if self.measure_timing:
            start_update = time.perf_counter()

        # remove detections with confidence lower than det_conf_threshold
        # if self.det_conf_threshold is not None:
        #     detection_boxes = self._filter_low_confidence_detections(detection_boxes)
        # bring boxes into next state
        
        self.predict([],motion_vectors, frame_type)
        
        # match predicted (tracked) boxes with detected boxes
        matches, unmatched_trackers, unmatched_detectors = trackerlib.match_bounding_boxes(self.boxes, detection_boxes, self.iou_threshold)   

        if self.measure_timing:
            self.last_update_dt = time.perf_counter() - start_update
            
        return unmatched_detectors

    def predict(self, frame, motion_vectors, frame_type):
        if self.measure_timing:
            start_predict = time.perf_counter()
        # I frame has no motion vectors
        if frame_type != "I":
            if self.use_only_p_vectors:
                motion_vectors = trackerlib.get_vectors_by_source(motion_vectors, "past")
            # get non-zero motion vectors and normalize them to point to the past frame (source = -1)
            motion_vectors = trackerlib.get_nonzero_vectors(motion_vectors)
            motion_vectors = trackerlib.normalize_vectors(motion_vectors)

            self.last_motion_vectors = motion_vectors
        
        # shift the box edges based on the contained motion vectors
        motion_vector_subsets = trackerlib.get_vectors_in_boxes(self.last_motion_vectors, self.boxes)
        shifts = trackerlib.get_box_shifts(motion_vector_subsets, metric="mean")
        self.boxes = trackerlib.adjust_boxes(self.boxes, shifts)

        tot_shift_mag = 0
        tot_norm_mag = 0
        conf_thr = 0.3
        for idx,shift in enumerate(shifts):
            if self.boxes[idx,-1] > conf_thr:
                shift_mag = math.sqrt(shift[0]*shift[0] + shift[1]*shift[1])
                tot_shift_mag += shift_mag
                tot_norm_mag += shift_mag / self.boxes[idx,3]*self.boxes[idx,4]
        #shift_mag =  np.linalg.norm(shifts, axis=1)
        #if shift_mag < 1:
        # Sum up the magnitudes
        #tot_shift_mag = np.sum(shift_mag)
        self.track_count += 1
        
        if self.measure_timing:
            self.last_predict_dt = time.perf_counter() - start_predict
        return shifts

    # def get_boxes(self):
    #     # get only those boxes with state "confirmed"
    #     mask = [target_state == "confirmed" for target_state in self.target_states]
    #     boxes_filtered = self.boxes[mask]
    #     return boxes_filtered

    def get_boxes(self):
        return self.boxes

    def get_box_ids(self):
        box_ids_filtered = [box_id for box_id, target_state in zip(self.box_ids, self.target_states) if target_state == "confirmed"]
        return box_ids_filtered
