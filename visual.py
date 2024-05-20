from PIL import Image, ImageDraw

# Create a blank white canvas
width, height = 800, 600  # Adjust the dimensions as needed
canvas = Image.new("RGB", (width, height), "white")
draw = ImageDraw.Draw(canvas)

# List of detected shelf locations (x, y, width, height)
# Replace this with the actual detected shelf coordinates from YOLO
detected_shelves = [(100, 100, 200, 50), (300, 250, 150, 40)]

# Draw shelves on the canvas
for shelf in detected_shelves:
    x, y, w, h = shelf
    draw.rectangle([x, y, x + w, y + h], outline="blue")

# Save or display the canvas
canvas.save("shelf_visualization.png")
canvas.show()
