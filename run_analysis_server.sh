#!/bin/bash

# Array of video names
videos="highway crossroad motorway"


# Array of trackers (corrected from tracker to trackers)
trackers="kcf lk mvt"


# Array of target accuracies
target_acc="0.9"
model='yolov5m'

for acc in $target_acc
do
    python run_server.py --det_model "$model" --vid_name "crossroad" --tracker "mvt" --target_acc "$acc" 
    sleep 1
done

for acc in $target_acc
do
    python run_server.py --det_model "$model" --vid_name "highway" --tracker "mvt" --target_acc "$acc"  
    sleep 1
done

for acc in $target_acc
do
    python run_server.py --det_model "$model" --vid_name "motorway" --tracker "mvt" --target_acc "$acc"  
    sleep 1
done


for acc in $target_acc
do
    python run_server.py --det_model "yolov5m" --vid_name "highway" --tracker "lk" --target_acc "$acc" 
    python run_server.py --det_model "yolov5m" --vid_name "highway" --tracker "kcf" --target_acc "$acc" 
    sleep 1
done

for acc in $target_acc
do
    python run_server.py --det_model "yolov5m" --vid_name "crossroad" --tracker "lk" --target_acc "$acc" 
    python run_server.py --det_model "yolov5m" --vid_name "crossroad" --tracker "kcf" --target_acc "$acc" 
    sleep 1
done

for acc in $target_acc
do
    python run_server.py --det_model "yolov5m" --vid_name "motorway" --tracker "lk" --target_acc "$acc" 
    python run_server.py --det_model "yolov5m" --vid_name "motorway" --tracker "kcf" --target_acc "$acc" 
    sleep 1
done

model='frcnn_resnet50'
for acc in $target_acc
do
    python run_server.py --det_model "$model" --vid_name "crossroad" --tracker "mvt" --target_acc "$acc" 
    sleep 1
done

for acc in $target_acc
do
    python run_server.py --det_model "$model" --vid_name "highway" --tracker "mvt" --target_acc "$acc"  
    sleep 1
done

for acc in $target_acc
do
    python run_server.py --det_model "$model" --vid_name "motorway" --tracker "mvt" --target_acc "$acc"  
    sleep 1
done
