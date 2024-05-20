from openvino import runtime as ov 
import cv2
import numpy as np

img = cv2.imread("E:\\Projects\\CV App-a-thon\\inference\\0.rf.2c10014884bac2c66f7e56b0d61821d0.jpg")

# Resize the image to match the expected input shape
input_shape = (640, 640)  # Replace with the desired shape
resized_img = cv2.resize(img, input_shape)
resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
input_data = resized_img.transpose((2, 0, 1))  # Change data layout (HWC to CHW)
input_data = input_data[np.newaxis]

core = ov.Core()
model = core.read_model(model="converted_model.xml")
compiled_model = core.compile_model(model=model, device_name="CPU")
output_layer = compiled_model.outputs[0]

result = compiled_model(input_data)[output_layer]

# Write the confidence scores to a text file
output_file = "confidence_scores.txt"  # Replace with your desired file path

with open(output_file, 'w') as file:
    file.write("Confidence Scores:\n")
    for i, detection in enumerate(result[0]):
        confidence = detection[0]  # Assuming a single class model
        file.write(f"Object {i + 1}: Confidence: {confidence:.2f}\n")

print(f"Confidence scores written to {output_file}")
