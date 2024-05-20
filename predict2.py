from ultralytics import YOLO
import torch
import cv2
import numpy as np
import os

# Load a pretrained YOLOv8n model
class_names = {
    2: "Bounce",
    35: "Supermilk",
    18: "Malkist",
    9: "Dark Fantasy",
    20: "Marie Light",
    28: "Nutri Choice-Herbs-",
    38: "Unibic"
   
    # Add more class names as needed
}

modelyolo = YOLO('E:\\Projects\\CV App-a-thon\\biscuitmodel\\best.pt')
results = modelyolo(source='E:\\Projects\\CV App-a-thon\\biscuitpredict\\Screenshot (69).png', save=True)

# Extract class predictions
boxes_cls = []

for result in results:
    boxes = result.boxes.cls
    # Convert the PyTorch tensor to a Python list
    boxes_list = boxes.tolist()
    # Extend the 'boxes_cls' list with the elements from 'boxes_list'
    boxes_cls.extend(boxes_list)

# Now, 'boxes_cls' contains all the class indices from each iteration
print(boxes_cls)
boxes_cls_set = set(boxes_cls)
print(boxes_cls_set)
out_stock = []

for class_index, class_name in class_names.items():
    if class_index not in boxes_cls_set:
        out_stock.append(class_name)

if len(out_stock) > 0:
    print("Out of Stock Classes:")
    for item in out_stock:
        text=(f"{item} is out of stock.")
        print(text)
else:
    print("All classes are in stock.")
    
# Load the image
input_image_path = 'E:\\Projects\\CV App-a-thon\\biscuitpredict\\Screenshot (69).png'
output_folder = 'E:\\Projects\\CV App-a-thon\\stock_output'

# Make sure the output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

image = cv2.imread(input_image_path)

# Specify the text and its position
stat=text
position = (50, 50)  # (x, y) coordinates

# Choose a font
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 2  # Font scale factor
font_color = (255, 255, 255)  # White color
font_thickness = 5

# Get the size of the text
(text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)

# Calculate the position to center the text
image_height, image_width, _ = image.shape
x = (image_width - text_width) // 2+50
y = (image_height + text_height) // 2

# Add the text to the image
cv2.putText(image, text, position, font, font_scale, font_color, font_thickness)

# Save the modified image to the output folder
output_image_path = os.path.join(output_folder, 'output_image.jpg')
cv2.imwrite(output_image_path, image)

print(f"Image with text saved to {output_image_path}")