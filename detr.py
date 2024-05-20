import torch
from transformers import DetrImageProcessor, DetrForObjectDetection

# Load the pretrained DETR model
checkpoint_path = 'E:\\Projects\\CV App-a-thon\\biscuitmodel\\DETR_model1.pth'

model = torch.load(checkpoint_path)

# Set the model to evaluation mode
#model.eval()

# Create a dummy input tensor
dummy_input = torch.randn(1, 3, 800, 1069)  # Adjust the shape and size according to your model's input requirements

# Export the model to ONNX
onnx_path = 'E:\\Projects\\CV App-a-thon\\biscuitmodel'
torch.onnx.export(model, dummy_input, onnx_path, verbose=True)