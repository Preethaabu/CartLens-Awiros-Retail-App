from ultralytics import YOLO
modelyolo = YOLO('E:\\Projects\\CV App-a-thon\\persondet(yolonas)\\kaggle\\working\\checkpoints\\my_first_yolonas_run\\average_model.pth')
results = modelyolo(source='E:\\Projects\\CV App-a-thon\\person_vid.mp4',save=True)