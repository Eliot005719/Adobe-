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
    a = results[0].boxes.boxes
    px = pd.DataFrame(a).astype("float")
    list = []

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'person' in c:
            list.append([x1, y1, x2, y2])

    bbox_idx = tracker.update(list)

    for bbox in bbox_idx:
        x3, y3, x4, y4, id = bbox
        
        results = cv2.pointPolygonTest(np.array(area, np.int32), ((x4, y4)), False)
        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
        cv2.circle(frame, (x4, y4), 4, (255, 0, 255), -1)
        cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)
        
        if results >= 0:
            crop = frame[y3:y4, x3:x4]
            (crop, output_folder)  # Save cropped entity to output folder
            area_c.add(id)

    cv2.polylines(frame, [np.array(area, np.int32)], True, (255, 0, 0), 2)
    k = len(area_c)
    cv2.putText(frame, str(k), (50, 60), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 3)
    cv2.imshow("I hate you", frame)
    
    if cv2.waitKey(0) & 0xFF == 27:
        break

cv2.destroyAllWindows()
