from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(data="wm_barriers_data/data.yaml", epochs=1)
