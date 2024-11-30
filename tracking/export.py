from ultralytics import YOLO
import sys
import os
# 获取当前脚本路径
script_path = os.path.dirname(os.path.abspath(__file__))

model = YOLO(os.path.join(script_path , "weights/yolov11m-pose.pt")) # Load a model
model.export(format="engine")

# Load the exported TensorRT model
trt_model = YOLO(os.path.join(script_path , "weights/yolov11m-pose.engine"))

# Run inference
results = trt_model("https://ultralytics.com/images/bus.jpg")