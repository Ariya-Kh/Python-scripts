import json
import os
from PIL import Image  # Import Pillow for getting image size

# Define your index-to-label mapping
index_to_label = {
    0: 'Level',
    1: 'Cap',
    2: 'Gardeshi',
    3: 'Chasbi',
    4: 'Other',
    5: 'Unknown',
    6: 'Jetprint',
    7: 'Defect'
    # Add more classes as needed
}



def resize_yolo_annotations(yolo_file, output_file, image_path):

    image = Image.open(image_path)
    image_width, image_height = image.size  # Returns (width, height)
    name, ext = os.path.splitext(image_path)

    with open(yolo_file, 'r') as f:
        lines = f.readlines()

    shapes = []
    for line in lines:
        parts = line.strip().split()
        class_index = int(parts[0])
        polygon = list(map(float, parts[1:]))

        # Convert YOLO normalized coordinates to actual coordinates in the current size
        for i in range(0, len(polygon), 2):
            polygon[i] *= image_width
            polygon[i + 1] *= image_height

        # Convert polygon points to a list of (x, y) pairs
        points = [(int(polygon[i]), int(polygon[i + 1])) for i in range(0, len(polygon), 2)]

        # LabelMe format requires a polygon or bounding box
        shape = {
            "label": index_to_label.get(class_index, 'unknown'),
            "points": points,
            "group_id": None,
            "shape_type": "polygon",
            "flags": {}
        }
        shapes.append(shape)

    # Create LabelMe JSON structure
    labelme_format = {
        "version": "4.5.6",
        "flags": {},
        "shapes": shapes,
        "imagePath": os.path.basename(output_file).replace('.json', ext),  # or .png, .jpeg etc.
        "imageData": None,
        "imageHeight": image_height,
        "imageWidth": image_width
    }

    # Save to JSON file
    with open(output_file, 'w') as json_file:
        json.dump(labelme_format, json_file, indent=4)

def find_image_file(yolo_path, formats=['.jpg', '.png', '.jpeg']):
    """Find the image file corresponding to a YOLO annotation file."""
    for ext in formats:
        image_path = yolo_path.replace('.txt', ext)
        if os.path.exists(image_path):
            return image_path
    return None

def process_folder(yolo_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(yolo_folder):

        if filename.endswith('.txt'):
            yolo_path = os.path.join(yolo_folder, filename)
            image_path = find_image_file(yolo_path)
            output_path = os.path.join(output_folder, filename.replace('.txt', '.json'))
            resize_yolo_annotations(yolo_path, output_path, image_path)
            print(f"Processed {filename}")

yolo_folder = '/home/perticon/Projects/deep/yolo-torch-2/train_zamzam_defect_1403_9_5/labels/b'
output_folder = '/home/perticon/Projects/deep/yolo-torch-2/train_zamzam_defect_1403_9_5/labels/b'

process_folder(yolo_folder, output_folder)