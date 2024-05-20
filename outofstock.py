from ultralytics import YOLO
import torch
import cv2
import numpy as np

# Load a pretrained YOLOv8n model
class_names = {
    2: "Bounce",
    35: "Supermilk",
    18: "Malkist",
    9: "Dark Fantasy",
    20: "Marie Light",
    28: "Nutri Choice-Herbs-",
    38:"Unibic"
    # Add more class names as needed
}
boxes_cls=[]

modelyolo = YOLO('E:\\Projects\\CV App-a-thon\\biscuitmodel\\best.pt')
results = modelyolo(source='E:\\Projects\\CV App-a-thon\\biscuitpredict\\Screenshot (69).png',save=True)
for result in results:
    boxes = result.boxes.cls
    boxes_cls.append(boxes)
    boxes_cls = boxes_cls[0].to(dtype=torch.int).tolist()

    
    #coord = boxes
    
      # Append coord to the list

# Now, 'all_coords' contains all the coordinates from each iteration
print(boxes_cls)
boxes_cls_set = set(boxes_cls)
print(boxes_cls_set)
out_stock=[]

for class_index, class_name in class_names.items():
    if class_index not in boxes_cls_set:
        out_stock.append(class_name)

if len(out_stock) > 0:
    print("Out of Stock Classes:")
    for item in out_stock:
        print(f"{item} is out of stock.")
else:
    print("All classes are in stock.")

image_path = 'E:\\Projects\\CV App-a-thon\\runs\\detect\\predict31\\Screenshot (69).png'
image = cv2.imread(image_path)

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (0, 0, 255)  # Red color
font_thickness = 2

if len(boxes)>0:
    # Iterate through the out-of-stock classes and draw text or bounding boxes
    for class_name in out_stock:
        for box in boxes:
            if len(box) >= 5 and box[4] == class_name:  # Check if the box corresponds to the out-of-stock class
                x1, y1, x2, y2 = map(int, box[:4])
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Draw a red bounding box
                cv2.putText(image, class_name, (x1, y1 - 10), font, font_scale, font_color, font_thickness)

# Save the modified image with out-of-stock classes highlighted
    output_image_path = 'E:\\Projects\\CV App-a-thon\\basketanalysis(yolov8)\\stock_output'
    cv2.imwrite(output_image_path, image)
else:
    print("No bounding boxes detected. Check your YOLO model's predictions.")