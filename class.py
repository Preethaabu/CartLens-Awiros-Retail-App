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

result = compiled_model(input_data)[output_layer]

# Define your list of class labels (modify this with your actual labels)
class_labels = ['50 50 Biscuit', 'Biscafe', 'Bounce', 'Bourbon Dark fantasy', 'Bourbon', 'Bourn Vita Biscuit', 'Chocobakes', 'Coffee Joy', 'Creme', 'Dark Fantasy', 'Digestive', 'Elite', 'Ginger', 'Good Day', 'Happy Happy', 'Hide - Seek', 'Jim Jam', 'KrackJack', 'Malkist', 'Marie Gold', 'Marie Light', 'Milk Bikis', 'Milk Short Cake', 'Mom Magic', 'Monaco', 'Nice', 'Nutri Choice', 'Nutri Choice-Crackers-', 'Nutri Choice-Herbs-', 'Nutri Choice-Sugar Free-', 'Oreo', 'Parle G', 'Potazo', 'Sunfeast green', 'Super Millets', 'Supermilk', 'Tninz', 'Treat', 'Unibic', 'Unibic-box','all rounder']

# Write the numerical values and class labels to a text file
output_file = "result2.txt"  # Replace with your desired file path

with open(output_file, 'w') as file:
    file.write("Detected Objects:\n")
    for i, detection in enumerate(result[0]):
        class_id = np.argmax(detection)
        confidence = detection[class_id]
        label = class_labels[class_id]
        file.write(f"Object {i + 1}: Class: {label}, Confidence: {confidence:.2f}\n")

print(f"Results with labels written to {output_file}")
