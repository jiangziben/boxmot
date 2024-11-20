from ultralytics import YOLO

# Load a YOLO11n PyTorch model
model = YOLO("/home/jetson/01_project/boxmot/tracking/weights/yolov11m_best.pt")

# Export the model to TensorRT
model.export(format="engine",half=True)  # creates 'yolo11n.engine'

# Load the exported TensorRT model
trt_model = YOLO("/home/jetson/01_project/boxmot/tracking/weights/yolov11m_best.engine")

# Run inference
results = trt_model("https://ultralytics.com/images/bus.jpg")