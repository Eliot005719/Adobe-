from ultralytics import YOLO
import os
import numpy
import random
import csv
# Load YOLO model
model = YOLO('yolov8n.pt')  # Using yolov8n pretrained model
folder_path = './Ads_problem4'
# Function to select a folder from available options

 
def select_folder(parent_folder):
    folder_list = os.listdir(parent_folder)
    
    print("Available folders:")
    for i, folder_name in enumerate(folder_list, start=1):
        print(f"{i}. {folder_name}")
    
    while True:
        try:
            choice = int(input("Select a folder by entering its corresponding number: "))
            if 1 <= choice <= len(folder_list):
                selected_folder = folder_list[choice - 1]
                return os.path.join(parent_folder, selected_folder)
            else:
                print("Invalid choice. Please enter a valid number.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")
 
# Function to get paths of ideal images
def get_ideal_images(folder_path):
    ideal_folder = os.path.join(folder_path, "ideal")
    ideal_images = [os.path.join(ideal_folder, image) for image in os.listdir(ideal_folder)]
    return ideal_images
 
# Function to get paths of other images
def get_other_images(folder_path):
    other_folder = os.path.join(folder_path, "other")
    other_images = [os.path.join(other_folder, image) for image in os.listdir(other_folder)]
    return other_images
 
# Function to count detected entities in the prediction results
def count_Entities(results):
    counts = {}
    cords = []  # Initialize an empty list to store cords
    total_objects = 0
 
    for result in results:
        boxes = result[0].boxes.cpu().numpy()  # Use 'pred' attribute to access boxes
        cords.extend(boxes)  # Extend the cords list with xyxy values
        
        for box in boxes:
            cls = (box[0])  # Get the class index from the box
            if cls not in counts:
                counts[cls] = 1
            else:
                counts[cls] += 1
        total_objects +=1  # Increment total_objects count
    
    return counts, cords
 
 
def compare_entities(ideal_boxes, other_boxes):
    # Compare entities between ideal and other counts
    ideal_counts, _ = count_Entities(ideal_boxes)
    other_counts, _ = count_Entities(other_boxes)
    
    missing_entities = {}
    extra_entities = {}
    
    # Find missing entities
    for cls, count in ideal_counts.items():
        if cls not in other_counts:
            missing_entities[cls] = count
    
    # Find extra entities
    for cls, count in other_counts.items():
        if cls not in ideal_counts:
            extra_entities[cls] = count
    
    return missing_entities, extra_entities
 
# Function to process ideal images and predict using the model
def ideal_images(selected_folder_path):
    all_results = []
    # Get paths of images in "ideal" folder
    ideal_images = get_ideal_images(selected_folder_path)
    for path in ideal_images:
        results = model.predict(path, save=True)  # Run YOLO prediction
        all_results.append(results)
    return all_results
 
 
# Function to process other images and predict using the model
def other_images(selected_folder_path):
    all_results = []
    # Get paths of images in "other" folder
    other_images = get_other_images(selected_folder_path)
    for path in other_images:
        results = model.predict(path, save=True)  # Run YOLO prediction
        all_results.append(results)
    return all_results
 
def compare_entities(ideal_results, other_result):
    # Compare entities between ideal and other result
    ideal_counts, _ = count_Entities(ideal_results)
    other_counts, _ = count_Entities(other_result)
    
    missing_entities = {}
    extra_entities = {}
    
    # Find missing entities
    for cls, count in ideal_counts.items():
        if cls not in other_counts:
            missing_entities[cls] = count
    
    # Find extra entities
    for cls, count in other_counts.items():
        if cls not in ideal_counts:
            extra_entities[cls] = count
    
    return missing_entities, extra_entities
 
 
def main():
    selected_folder_path = select_folder(folder_path)
    print(f"You selected: {selected_folder_path}")
    
    ideal_results = ideal_images(selected_folder_path)
    
    # Get paths of images in "other" folder
    other_images_paths = get_other_images(selected_folder_path)
    
    for other_path in other_images_paths:
        other_results = model.predict(other_path, save=True)
        
        # Compare entities between ideal and this other image
        missing_entities, extra_entities = compare_entities(ideal_results, other_results)
        
        # Write missing and extra entities to a CSV file with the same name as the other image
        other_image_name = os.path.splitext(os.path.basename(other_path))[0]
        csv_filename = f"{selected_folder_path}/other/{other_image_name}.csv"
        
        with open(csv_filename, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["Entity", "Difference", "Meta"])
            
            for cls, missing_count in missing_entities.items():
                csv_writer.writerow([f"Entity-class{cls}", "Missing", "xy location of bounding "])
            
            for cls, extra_count in extra_entities.items():
                csv_writer.writerow([f"Entity-class{cls}", "Extra", "xy location or size of bounding box"])
 
        print(f"Comparison {id} for '{other_image_name}' written to {csv_filename}")
 
 
 
if __name__ == "__main__":
    main()
 