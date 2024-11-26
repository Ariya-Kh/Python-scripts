import json
import os
import cv2

# Define your label-to-index mapping
label_to_index = {
    'Level': 0,
    'Cap': 1,
    'Gardeshi': 2,
    'Chasbi': 3,
    'Other': 4,
    'Unknown': 5,
    'Jetprint': 6,
    'defect': 7
    # Add more classes as needed
}

def convert_labelme_to_yolo(json_file, output_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    

    img_width = data['imageWidth']
    img_height = data['imageHeight']

    with open(output_file, 'w') as out_f:
        for shape in data['shapes']:
            label = shape['label']   
           
            if label not in label_to_index:
                print(f"Warning: {label} not in label_to_index mapping. Skipping.")
                continue
            label_index = label_to_index[label]
            points = shape['points']
            
            # YOLO format requires normalized coordinates
            yolo_format_points = []
            for x, y in points:
                yolo_format_points.append(x / img_width)
                yolo_format_points.append(y / img_height)
            # Convert points to a single line in YOLO format
            yolo_format_str = " ".join(map(str, yolo_format_points))
            
            # Write to the file in the format: <label_index> <x1> <y1> <x2> <y2> ... <xn> <yn>
            out_f.write(f"{label_index} {yolo_format_str}\n")

def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.endswith('.json'):
            json_path = os.path.join(input_folder, filename)
            txt_filename = filename.replace('.json', '.txt')
            txt_path = os.path.join(output_folder, txt_filename)
            convert_labelme_to_yolo(json_path, txt_path)
            print(f"Processed {filename}")

input_folder = '/home/perticon/Projects/deep/yolo-torch-2/train_zamzam_defect_1403_9_5/images/val'
output_folder = input_folder
# new_width = 1536 # Example new width    
# new_height = 3072  # Example new height

process_folder(input_folder, output_folder)