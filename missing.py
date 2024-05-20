from ultralytics import YOLO

# Load a pretrained YOLOv8n model
modelyolo = YOLO('E:\\Projects\\CV App-a-thon\\biscuitmodel\\best.pt')
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

modelyolo = YOLO('E:\\Projects\\CV App-a-thon\\biscuitmodel\\best.pt')
all_coords = []  # Initialize an empty list to store coordinates

results = modelyolo(source='E:\\Projects\\CV App-a-thon\\biscuitpredict\\Screenshot (69).png',save=True)



# Convert integer labels to class names

# Now, det_class
all_coords=[]
for result in results:
    boxes = result.boxes.xyxy
    coord = boxes
    all_coords.append(coord)  # Append coord to the list



    # Convert the tensor to a Python list
      # Append coord to the list

# Now, 'all_coords' contains all the coordinates as Python lists from each iteration

print(all_coords)