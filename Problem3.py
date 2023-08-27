from ultralytics import YOLO
import os
import numpy as np
import csv
import shutil
from PIL import Image
from skimage.metrics import structural_similarity as ssim

# Load YOLO model
model = YOLO('yolov8n.pt')  # Using yolov8n pretrained model
folder_path = './All_Images'  # Path to folder where images are stored
output_folder = './Grouped_Images'  # Path to folder where grouped images will be saved

def getFolder_Path(folder_path):
    # Getting the path to all files in a list
    file_list = []

    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            file_list.append(file_name)
    return file_list

def detect_Segments():
    # Using YOLO to detect objects in every image
    files_in_folder = getFolder_Path(folder_path)
    for i in files_in_folder:
        path = f'./All_Images/{i}'
        results = model.predict(path, save=True)
        counts, total_objects = count_Entities(results)
        group_images(i, counts)
        write_to_CSV(i, counts, total_objects)

def count_Entities(results):
    # Counting entities detected in the results
    counts = {}
    total_objects = 0

    for result in results:
        boxes = result.boxes.cpu().numpy()
        total_objects += len(boxes)
        for box in boxes:
            cls = int(box.cls[0])
            if cls not in counts:
                counts[cls] = 1
            else:
                counts[cls] += 1

    return counts, total_objects

def write_to_CSV(img_name, counts, total_objects):
    # Writing the data to a CSV file
    name = img_name.split('.')[0]
    csv_path = f"./runs/detect/predict/{name}.csv"

    with open(csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Entity', 'Count'])
        for key in counts:
            writer.writerow([model.names[key], counts[key]])
        writer.writerow(['Total Objects', total_objects])

    print(f"Data saved to {csv_path}")

def calculate_image_similarity(image1_path, image2_path):
    # Calculate the structural similarity index between two images
    image1 = Image.open(image1_path).convert("L")
    image2 = Image.open(image2_path).convert("L")
    similarity = ssim(np.array(image1), np.array(image2))
    return similarity

def group_images(img_name, counts):
    # Creating folders based on detected entities and moving images
    for entity_id in counts:
        entity_name = model.names[entity_id]
        group_folder = f"{output_folder}/Folder-grp{entity_id}_{entity_name}"
        if not os.path.exists(group_folder):
            os.makedirs(group_folder)

        src_path = f"./All_Images/{img_name}"
        dst_path = f"{group_folder}/{img_name}"
        shutil.copy(src_path, dst_path)

    print(f"Images grouped for {img_name}")

if __name__ == "__main__":
    detect_Segments()
