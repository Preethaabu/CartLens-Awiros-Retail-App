from ultralytics import YOLO
model=YOLO("E:\\Projects\\CV App-a-thon\\persondet(yolov8)\\kaggle\\working\\runs\\detect\\train\\weights\\person.pt")
model.predict(source="E:\\Projects\\CV App-a-thon\\person_vid.mp4",save=True,conf=0.4)