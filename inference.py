from openvino import runtime as ov 
import cv2
import numpy as np

img = cv2.imread("E:\\Projects\\CV App-a-thon\\biscuitpredict\\Screenshot (69).png")

# Resize the image to match the expected input shape
input_shape = (640, 640)  # Replace with the desired shape
resized_img = cv2.resize(img, input_shape)
resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
input_data = resized_img.transpose((2, 0, 1))  # Change data layout (HWC to CHW)
input_data = input_data[np.newaxis]


core = ov.Core()
model = core.read_model(model="E:\\Projects\\CV App-a-thon\\biscuitmodel\\multi.xml")
compiled_model = core.compile_model(model=model, device_name="CPU")
output_layer = compiled_model.outputs[0]
# Write the numerical values in the result to a text file
output_file1 = "result3.txt"  # Replace with your desired file path

with open(output_file1, 'w') as file:
    file.write(str(output_layer))
result = compiled_model(input_data)[output_layer]

# Write the numerical values in the result to a text file
output_file = "result2.txt"  # Replace with your desired file path

with open(output_file, 'w') as file:
    file.write(str(result[0]))
   

print(f"Results written to {output_file}")
