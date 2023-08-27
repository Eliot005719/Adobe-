import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from P5 import *
from datetime import datetime
import os

# Load YOLO model
model = YOLO('yolov8n.pt')

# Load COCO class list
with open("coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

tracker = Tracker()
area = [(54, 436), (41, 449), (317, 494), (317, 470)]
area_c = set()


def imgwrite(crop, output_folder):
    now = datetime.now()
    current_time = now.strftime("%d_%m_%Y_%H_%M_%S")
    filename = f"{current_time}."
    img_path = os.path.join(output_folder, filename)
    cv2.imsave(img_path, crop)
    


# Load images from the "All_Images" folder
image_folder = "All_Images"
output_folder = "Cropped_Entities"  # Output folder for cropped entities
image_files = [f for f in os.listdir(image_folder) if f.endswith(".jpeg") or f.endswith(".jpg")]

for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    frame = cv2.imread(image_path)

    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame)
    