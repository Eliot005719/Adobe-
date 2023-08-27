import torch

# Model
from ultralytics import YOLO
model=YOLO('yolov8n.pt')
# Image
runs = r'C:\Users\yadav\HP\Help\SCHOOL\Adobe Round 2\Problem 4\runs\detect\predict'


# Inference
results = model(runs)
results.pandas().xyxy[0]

