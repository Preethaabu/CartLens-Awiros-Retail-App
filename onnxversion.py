import onnx

# Load the ONNX model
model = onnx.load("E:\\Projects\\CV App-a-thon\\personmodel\\person.onnx")

# Get the opset version
opset_version = model.opset_import[0].version
print(f"Opset version: {opset_version}")