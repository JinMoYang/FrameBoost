import numpy as np
import torch

def load_detections_orig(det_file, num_frames):
    det_boxes = []
    det_scores = []
    raw_data = np.genfromtxt(det_file, delimiter=',')
    for frame_idx in range(num_frames):
        idx = np.where(raw_data[:, 0] == frame_idx+1)
        if idx[0].size:
            det_boxes.append(np.stack(raw_data[idx], axis=0)[:, 2:6])
            det_scores.append(np.stack(raw_data[idx], axis=0)[:, 6])
        else:
            det_boxes.append(np.empty(shape=(0, 4)))
            det_scores.append(np.empty(shape=(0,)))
    return det_boxes, det_scores

def load_detections_YOLO(preds, frameHeight, frameWidth):
    if len(preds)==0:
        return np.empty(shape=(0, 6))
    else:
        cls = preds[:,0]
        conf = preds[:,-1]
        boxes = preds[:,1:5]
        scale = [frameWidth, frameHeight, frameWidth, frameHeight]
        boxes *= scale
        x_center = np.copy(boxes[:,0])
        y_center = np.copy(boxes[:,1])
        width = np.copy(boxes[:,2])
        height = np.copy(boxes[:,3])
        boxes[:,0] = x_center - width/2
        boxes[:,1] = y_center - height/2
        # boxes[:,2] = x_center + width/2
        # boxes[:,3] = y_center + height/2
        boxes[:,2] = width
        boxes[:,3] = height
        
        cls_reshaped = cls.reshape(-1, 1)
        conf_reshaped = conf.reshape(-1, 1)
        det_boxes_cat = np.hstack((cls_reshaped, boxes, conf_reshaped))
        return det_boxes_cat

def load_detections_txt_YOLO(det_file, frameHeight, frameWidth):
    det_boxes = []
    #det_scores = []
    det_boxes = np.genfromtxt(det_file)#, dtype = object)
    #pdb.set_trace()
    # for data in raw_data:
    #     det_boxes.append(data)
        #det_scores.append(data[5])
    # else:
    #     det_boxes.append(np.empty(shape=(0, 4)))
    #     det_scores.append(np.empty(shape=(0,)))
    cls = det_boxes[:,0]
    conf = det_boxes[:,-1]
    boxes = det_boxes[:,1:5]
    scale = [frameWidth, frameHeight, frameWidth, frameHeight]
    boxes *= scale
    
    x_center = np.copy(boxes[:,0])
    y_center = np.copy(boxes[:,1])
    width = np.copy(boxes[:,2])
    height = np.copy(boxes[:,3])

    boxes[:,0] = x_center - width/2
    boxes[:,1] = y_center - height/2
    # boxes[:,2] = x_center + width/2
    # boxes[:,3] = y_center + height/2
    boxes[:,2] = width
    boxes[:,3] = height
        
    cls_reshaped = cls.reshape(-1, 1)
    conf_reshaped = conf.reshape(-1, 1)

    det_boxes_cat = np.hstack((cls_reshaped, boxes, conf_reshaped))
    return det_boxes_cat

def detections_to_YOLO(det_boxes, frameHeight, frameWidth):
    cls = det_boxes[:,0]
    conf = det_boxes[:,-1]
    boxes = det_boxes[:,1:5]
    width = (boxes[:,2] - boxes[:,0])
    height = (boxes[:,3] - boxes[:,1])
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
    # Reshape array1 and array2 to have shape (300, 1)
    cls_reshaped = cls.reshape(-1, 1)
    conf_reshaped = conf.reshape(-1, 1)

    # Vertically concatenate the arrays

    if torch.is_tensor(cls_reshaped) and torch.is_tensor(boxes) and torch.is_tensor(conf_reshaped):
        det_boxes_cat = torch.cat((cls_reshaped, boxes, conf_reshaped), dim=1)
    else:
        det_boxes_cat = np.hstack((cls_reshaped, boxes, conf_reshaped))
    return det_boxes_cat

def detections_to_tracker(det_boxes, frameHeight= 0, frameWidth=0):
    cls = det_boxes[:,0]
    conf = det_boxes[:,-1]
    boxes = det_boxes[:,1:5]
    width = (boxes[:,2] - boxes[:,0])
    height = (boxes[:,3] - boxes[:,1])
    # width = (boxes[:,2] - boxes[:,0])
    # height = (boxes[:,3] - boxes[:,1])

    boxes[:,2] = width
    boxes[:,3] = height

    # Reshape array1 and array2 to have shape (300, 1)
    cls_reshaped = cls.reshape(-1, 1)
    conf_reshaped = conf.reshape(-1, 1)

    # Vertically concatenate the arrays
    if torch.is_tensor(cls_reshaped) and torch.is_tensor(boxes) and torch.is_tensor(conf_reshaped):
        det_boxes_cat = torch.cat((cls_reshaped, boxes, conf_reshaped), dim=1)
    else:
        det_boxes_cat = np.hstack((cls_reshaped, boxes, conf_reshaped))
    return det_boxes_cat

def tracker_to_YOLO(det_boxes, frameHeight, frameWidth):
    cls = det_boxes[:,0]
    conf = det_boxes[:,-1]
    boxes = det_boxes[:,1:5]
    width = boxes[:,2]#(det_boxes[:,2] - det_boxes[:,0])
    height = boxes[:,3]#(det_boxes[:,3] - det_boxes[:,1])
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
    # Reshape array1 and array2 to have shape (300, 1)
    cls_reshaped = cls.reshape(-1, 1)
    conf_reshaped = conf.reshape(-1, 1)

    # Vertically concatenate the arrays

    if torch.is_tensor(cls_reshaped) and torch.is_tensor(boxes) and torch.is_tensor(conf_reshaped):
        det_boxes_cat = torch.cat((cls_reshaped, boxes, conf_reshaped), dim=1)
    else:
        det_boxes_cat = np.hstack((cls_reshaped, boxes, conf_reshaped))
    return det_boxes_cat

def YOLO_to_tracker(det_boxes, frameHeight, frameWidth):
    
    cls = det_boxes[:,0]
    conf = det_boxes[:,-1]
    boxes = det_boxes[:,1:5]
    
    boxes[:,0] *= frameWidth
    boxes[:,1] *= frameHeight
    boxes[:,2] *= frameWidth
    boxes[:,3] *= frameHeight
    
    # width = boxes[:,2]
    # height = boxes[:,3]
    # width = (boxes[:,2] - boxes[:,0])
    # height = (boxes[:,3] - boxes[:,1])
    boxes[:,0] = boxes[:,0] - boxes[:,2]//2
    boxes[:,1] = boxes[:,1] - boxes[:,3]//2
    # boxes[:,2] = width
    # boxes[:,3] = height

    # Reshape array1 and array2 to have shape (300, 1)
    cls_reshaped = cls.reshape(-1, 1)
    conf_reshaped = conf.reshape(-1, 1)

    # Vertically concatenate the arrays
    if torch.is_tensor(cls_reshaped) and torch.is_tensor(boxes) and torch.is_tensor(conf_reshaped):
        det_boxes_cat = torch.cat((cls_reshaped, boxes, conf_reshaped), dim=1)
    else:
        det_boxes_cat = np.hstack((cls_reshaped, boxes, conf_reshaped))
    return det_boxes_cat

def save_to_detections_YOLO(preds_tracked_file, det_boxes, frameHeight, frameWidth):
    cls = det_boxes[:,0]
    conf = det_boxes[:,-1]
    boxes = det_boxes[:,1:5]
    
    width = boxes[:,2]#(det_boxes[:,2] - det_boxes[:,0])
    height = boxes[:,3]#(det_boxes[:,3] - det_boxes[:,1])
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
    
    # Reshape array1 and array2 to have shape (300, 1)
    cls_reshaped = cls.reshape(-1, 1)
    conf_reshaped = conf.reshape(-1, 1)

    # Vertically concatenate the arrays
    det_boxes_cat = np.hstack((cls_reshaped, boxes, conf_reshaped))
    #det_boxes = np.array(det_boxes,dtype=object)
    np.savetxt(preds_tracked_file,det_boxes_cat)