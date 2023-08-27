from ultralytics import YOLO  # pip install ultralytics
import os
import csv
import cv2
import shutil
import glob

# Load YOLO model
model = YOLO('yolov8n.pt')  # Using yolov8n pretrained model

folder_path = './Ads_problem4'  # Path to the main folder containing subfolders
output_folder = './Grouped_Images'  # Path to folder where grouped images will be saved


def try1(folder_path):
    file_list = []  # Initialize a list to store file names

    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            file_list.append(file_name)  # Append file name to the list
    return file_list  # Return the list of file names

def detect_Segments(files_in_folder):
    for i in files_in_folder:
        path = f'{folder_path}/{i}'
        results = model.predict(path, save=True)
        counts, total_objects = count_Entities(results)
        write_to_CSV(i, counts, total_objects)

def count_Entities(results):
    counts = {}
    total_objects = 0

    for result in results.pred[0]:
        cls = int(result[5])
        if cls not in counts:
            counts[cls] = 1
        else:
            counts[cls] += 1
        total_objects += 1

    return counts, total_objects

def write_to_CSV(img_name, counts, total_objects):
    name = img_name.split('.')[0]
    csv_path = f"./runs/detect/predict/{name}.csv"

    with open(csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Entity', 'Count'])
        for key in counts:
            writer.writerow([model.names[key], counts[key]])
        writer.writerow(['Total Objects', total_objects])

    print(f"Data saved to {csv_path}")



if __name__ == "__main__":
    files_in_folder = try1(folder_path)
    detect_Segments(files_in_folder)
