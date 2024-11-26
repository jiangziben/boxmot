from ultralytics import YOLO

model = YOLO("/home/jiangziben/CodeProject/ultralytics/workspace/people_track/yolo11m-pose.pt")  # Load a model
model.export(format="engine", half=True)