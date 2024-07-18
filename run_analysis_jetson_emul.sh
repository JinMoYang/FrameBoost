#!/bin/bash

# Array of video names
videos="highway crossroad motorway"

# Array of target accuracies
acc="0.8"

for video in $videos
do
    python run_jetson_emul.py --det_model "yolov5m" --vid_name "$video" --tracker "mvt" --target_acc "$acc" 
    sleep 1
done
