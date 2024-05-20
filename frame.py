import cv2
import os

# Video file path
video_path = 'E:\\Projects\\CV App-a-thon\\biscuitpredict\\basketvid1.mp4'

# Output directory where frames will be saved
output_directory = 'E:\\Projects\\CV App-a-thon\\biscuit_frames'

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Open the video file
cap = cv2.VideoCapture(video_path)

# Initialize a frame counter
frame_count = 0

# Read frames from the video and save them as images
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Define the file name for the frame
    frame_filename = f'{frame_count:04d}.jpg'  # You can use different image formats like .jpg, .png, etc.

    # Write the frame to the output directory
    frame_path = os.path.join(output_directory, frame_filename)
    cv2.imwrite(frame_path, frame)

    frame_count += 1

# Release the video capture object
cap.release()

print(f'Frames extracted: {frame_count}')
