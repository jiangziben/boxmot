from ultralytics import YOLO

model = YOLO("/home/jiangziben/CodeProject/ultralytics/workspace/people_track/yolo11m-pose.pt")  # Load a model
model.export(format="engine", half=True)

# Load the exported TensorRT model
trt_model = YOLO("/home/jetson/01_project/boxmot/tracking/weights/yolov11m-pose.engine")

# Run inference
results = trt_model("https://ultralytics.com/images/bus.jpg")