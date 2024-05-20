from ultralytics import YOLO

# Load a model  # load an official model
model = YOLO('E:\\Projects\\CV App-a-thon\\personmodel\\personv8.pt')  # load a custom trained model

# Export the model
model.export(format='openvino')