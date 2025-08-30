from ultralytics import YOLO
model = YOLO("yolo11s.pt")
model.train(data = "../train.yaml",epochs = 100, patience = 10,save = True)