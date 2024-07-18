COCO_names = {0: 'person',
 1: 'bicycle',
 2: 'car',
 3: 'motorcycle',
 4: 'airplane',
 5: 'bus',
 6: 'train',
 7: 'truck',
 8: 'boat',
 9: 'traffic light',
 10: 'fire hydrant',
 11: 'stop sign',
 12: 'parking meter',
 13: 'bench',
 14: 'bird',
 15: 'cat',
 16: 'dog',
 17: 'horse',
 18: 'sheep',
 19: 'cow',
 20: 'elephant',
 21: 'bear',
 22: 'zebra',
 23: 'giraffe',
 24: 'backpack',
 25: 'umbrella',
 26: 'handbag',
 27: 'tie',
 28: 'suitcase',
 29: 'frisbee',
 30: 'skis',
 31: 'snowboard',
 32: 'sports ball',
 33: 'kite',
 34: 'baseball bat',
 35: 'baseball glove',
 36: 'skateboard',
 37: 'surfboard',
 38: 'tennis racket',
 39: 'bottle',
 40: 'wine glass',
 41: 'cup',
 42: 'fork',
 43: 'knife',
 44: 'spoon',
 45: 'bowl',
 46: 'banana',
 47: 'apple',
 48: 'sandwich',
 49: 'orange',
 50: 'broccoli',
 51: 'carrot',
 52: 'hot dog',
 53: 'pizza',
 54: 'donut',
 55: 'cake',
 56: 'chair',
 57: 'couch',
 58: 'potted plant',
 59: 'bed',
 60: 'dining table',
 61: 'toilet',
 62: 'tv',
 63: 'laptop',
 64: 'mouse',
 65: 'remote',
 66: 'keyboard',
 67: 'cell phone',
 68: 'microwave',
 69: 'oven',
 70: 'toaster',
 71: 'sink',
 72: 'refrigerator',
 73: 'book',
 74: 'clock',
 75: 'vase',
 76: 'scissors',
 77: 'teddy bear',
 78: 'hair drier',
 79: 'toothbrush'}

video_fps = {
    'motorway': 24,
    'driving1': 30,
    'driving_downtown': 30,
    'nyc': 30,
    'park': 30,
    'crossroad': 30,
    'highway': 25
}
#server
video_res = {
   'highway':(1280,720),
   'motorway':(1280,720),
   'crossroad':(1920,1080)
}
server_inference_dict = {
    '240p': {
        'yolov5s': [9.095152521133423],
        'yolov5m': [],
        'frcnn_resnet50': [33.38260836283366]
    },
    '480p': {
        'yolov5s': [12.054502824147542],
        'yolov5m': [],
        'frcnn_resnet50': [33.93206071217855]
    },
    '720p': {
        'yolov5s': [14.636934591929117],
        'yolov5m': [16.697662848472596],
        'frcnn_resnet50': [34.88400363922119]
    },
    '1080p': {
        'yolov5s': [21.646084798177082],
        'yolov5m': [31.713447302712336],
        'frcnn_resnet50': [36.76578197479248]
    }
}

jetson_inference_dict = {
    '240p': {
        'yolov5s': [],
    },
    '480p': {
        'yolov5s': [],
    },
    '720p': {
        'yolov5s': [140.6583033970424],
        'yolov5m': [314.42537583618157]
    },
    '1080p': {
        'yolov5s': [146.7979452950614],
        'yolov5m': [319.52247297763875]
    }
}

jetson_allow_det = 3