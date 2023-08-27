import cv2
import os

# Load the image
image_path = 'path_to_your_image.jpg'
image = cv2.imread(image_path)

# Define the coordinates of the region to be cropped (x, y, width, height)
x = 100
y = 150
width = 200
height = 250

# Crop the region of interest
cropped_image = image[y:y+height, x:x+width]

# Create a folder to save the cropped images if it doesn't exist
output_folder = 'cropped_images'
os.makedirs(output_folder, exist_ok=True)

# Save the cropped image
output_path = os.path.join(output_folder, 'bus.jpg')
cv2.imwrite(output_path, cropped_image)

print("Cropped image saved successfully.")
