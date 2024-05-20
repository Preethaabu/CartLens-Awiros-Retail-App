import matplotlib.pyplot as plt
import numpy as np
import cv2
from ultralytics import YOLO

# Load a pretrained YOLOv8n model
class_names = {
    2: "Bounce",
    35: "Supermilk",
    18: "Malkist",
    9: "Dark Fantasy",
    20: "Marie Light",
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

# Perform class count without using Counter
class_count = {}
for class_index in boxes_cls:
    if class_index not in class_count:
        class_count[class_index] = 1
    else:
        class_count[class_index] += 1

# Convert the class_count to a dictionary
class_count_dict = dict(class_count)
print(class_count_dict)

# Get class names and counts
class_names_list = list(class_count_dict.keys())
class_counts = list(class_count_dict.values())
print(class_names_list)
print(class_counts)

# Create a list of unique colors for each class
# Make sure class_names and class_count_dict are aligned
class_names_list = list(class_names.keys())
class_counts = [class_count_dict.get(class_index, 0) for class_index in class_names_list]

# Create a list of unique colors for each class
num_classes = len(class_names_list)
colors = ["#702963", "#800020", "#9F2B68", "#702963", "#DA70D6", "#5D3FD3"]

# Define the shape of your image (height, width, and channels)
image_shape = (1080, 1930, 3)

# Create a white pixel image using OpenCV with the specified dimensions
white_pixel_image = np.ones(image_shape, dtype=np.uint8) * 255

# Create a single figure with two subplots (1 row, 2 columns)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Create a bar chart on the first subplot (ax1)
ax1.bar(class_names.values(), class_counts, color=colors, width=0.8)
ax1.set_xlabel('Product Names')
ax1.set_ylabel('Number of Detections')
ax1.set_title('Products In-Stock')
ax1.tick_params(axis='x', rotation=45)
ax1.set_ylim(0, 30)
legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors]
ax1.legend(legend_handles, class_names.values(), loc='upper right', title='Product Classes', prop={'size': 6})

# Create a pie chart on the second subplot (ax2)
second_pie_data = [15, 25, 30, 20]
second_labels = ['Area 1', 'Area 2', 'Area 3', 'Area 4']
colors2 = ["#9F2B68", "#800020", "#702963", "#DA70D6"]
ax2.pie(second_pie_data, autopct='%1.1f%%', shadow=True, startangle=90, colors=colors2)
ax2.set_title('Customer Purchasing Behaviour')
ax2.set_aspect('equal')  # Ensure the pie chart is circular
ax2.legend(second_labels, loc='lower right', title='Product Classes', prop={'size': 6}, bbox_to_anchor=(1.25, 0))

# Render the Matplotlib visualization on the white pixel image
figure_canvas = fig.canvas
figure_canvas.draw()
width, height = fig.get_size_inches() * fig.get_dpi()
image_from_canvas = np.frombuffer(figure_canvas.buffer_rgba(), dtype=np.uint8).reshape(int(height), int(width), 4)
image_from_canvas = cv2.cvtColor(image_from_canvas, cv2.COLOR_RGBA2BGR)

# Define the position for overlaying the Matplotlib visualization
start_x, start_y = 50, 50

# Overlay the Matplotlib visualization on the white pixel image
white_pixel_image[start_y:start_y + int(height), start_x:start_x + int(width)] = image_from_canvas

# Close the plots
plt.close()

cv2.imshow('Overlayed Image', white_pixel_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
