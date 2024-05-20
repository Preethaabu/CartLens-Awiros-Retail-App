import cv2
import os

# Load the image
input_image_path = 'E:\\Projects\\CV App-a-thon\\biscuitpredict\\Screenshot (69).png'
output_folder = 'E:\\Projects\\CV App-a-thon\\stock_output'

# Make sure the output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

image = cv2.imread(input_image_path)

# Specify the text and its position
text = "Hello, OpenCV!"
position = (50, 50)  # (x, y) coordinates

# Choose a font
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1  # Font scale factor
font_color = (255, 255, 255)  # White color
font_thickness = 2

# Add the text to the image
cv2.putText(image, text, position, font, font_scale, font_color, font_thickness)

# Save the modified image to the output folder
output_image_path = os.path.join(output_folder, 'output_image.jpg')
cv2.imwrite(output_image_path, image)

print(f"Image with text saved to {output_image_path}")
