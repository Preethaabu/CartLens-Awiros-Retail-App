from ultralytics import YOLO
modelyolo = YOLO('E:\\Projects\\CV App-a-thon\\persondet(yolov8)\\kaggle\\working\\runs\\detect\\train\\weights\\shelf_monitoring.onnx')
results = modelyolo(source='E:\\Projects\\CV App-a-thon\\biscuitpredict\\Screenshot (69).png', save=True)