from ultralytics import YOLO
model=YOLO("E:\\Projects\\CV App-a-thon\\converted_model.xml")
model.predict(source="E:\\Projects\\CV App-a-thon\\inference\\0.rf.2c10014884bac2c66f7e56b0d61821d0.jpg",save=True)