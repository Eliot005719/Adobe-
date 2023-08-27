from ultralytics import YOLO
import os
import shutil
from PIL import Image

# Load YOLO model
model = YOLO('yolov8n.pt')
folder_path = './All_Images'  # Path where images are stored
output_folder = './top_entities'  # Path to folder where grouped entity images will be saved

def getFolder_Path(folder_path):
    file_list = []
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            file_list.append(file_name)
    return file_list

def detect_Segments():
    files_in_folder = getFolder_Path(folder_path)
    for i in files_in_folder:
        path = f'./All_Images/{i}'
        results = model.predict(path, save=False, conf=0.4)
        entities = extract_Entities(results)
        group_entities(i, entities)

def extract_Entities(results):
    entities = []
    for result in results:
        for entity in result:
            class_id = int(entity['class'])
            class_name = model.names[class_id]
            confidence = entity['score']
            bbox = entity['bbox']
            entities.append({
                'class_id': class_id,
                'class_name': class_name,
                'confidence': confidence,
                'bbox': bbox
            })
    return entities

def group_entities(image_name, entities):
    for entity in entities:
        class_id = entity['class_id']
        class_name = entity['class_name']
        confidence = entity['confidence']
        bbox = entity['bbox']

        entity_folder = f"{output_folder}/{class_name}"
        os.makedirs(entity_folder, exist_ok=True)

        image_path = f"./All_Images/{image_name}"
        image = Image.open(image_path)
        entity_image = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))

        entity_image_name = f"{image_name}_confidence_{confidence:.2f}.jpeg"
        entity_image_path = os.path.join(entity_folder, entity_image_name)
        entity_image.save(entity_image_path)

    print(f"Entities grouped for {image_name}")

if __name__ == "__main__":
    detect_Segments()
