# FrameBoost: Advanced Video Analytics with Keyframe Selection via IoU Prediction

## Dependency
Please set up a python virtual environment and install the packages listed in 
[requirements.txt](requirements.txt).
```bash
# set up a python3 virtualenv environment
pip install -r requirements.txt
```

## Object Detection, DNN Model and COCO Label
1. Models used in this project:
    * We used faster_rcnn_resnet50 and YOLO-v5m

2. The DNN models used in this project are accessible through the following links
    * PyTorch page for Faster-RCNN (https://pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html#torchvision.models.detection.fasterrcnn_resnet50_fpn)
    * YOLOv5 official Github (https://github.com/ultralytics/yolov5?tab=readme-ov-file)
format. Labels used in this project:
 
## Data Source & benchmark
We refer source of 3 (highway,crossroad,motorway) video dataset and benchmark to Yoda benchmark (https://yoda.cs.uchicago.edu/download.html)

### Data preparation
For efficient execution, we prepare and load each frame's contents and detection information beforehand.

1. Necessary contents in frame contents are frame, motion-vectors, and frame type (I,P,B frame) for each frame. 
We handle each frame as a dictionary with keys 'frame','motion_vectors','frame_type.'

2. For detection, information we prepare them in list of detection (np.array). 
Detection boxes are formed in YOLO format, 
(x_center,y_center,w,h) in ratio with respect to (frameWidth,frameHeight).
Once a prediction is loaded into the video analyzer, it is converted to native coorintes.