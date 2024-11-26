import json
import os
from PIL import Image, ImageDraw

def create_mask_from_json(json_path, output_dir):
    # Load the labelme JSON file
    with open(json_path) as f:
        data = json.load(f)

    # Get the image dimensions from the JSON file
    image_width = data['imageWidth']
    image_height = data['imageHeight']

    # Extract the base name of the image from the JSON filename
    base_name = os.path.splitext(os.path.basename(json_path))[0]

    # Initialize a blank binary mask for the entire image
    mask = Image.new('1', (image_width, image_height))  # '1' mode for binary mask

    # Iterate through each shape in the JSON and draw it on the mask
    for shape in data['shapes']:
        if shape['shape_type'] == 'polygon':
            points = shape['points']
            # Convert points to tuples of (x, y)
            polygon = [(int(x), int(y)) for x, y in points]

            # Draw the polygon on the mask
            ImageDraw.Draw(mask).polygon(polygon, outline=1, fill=1)

    # Save the combined binary mask image with the specified naming convention
    mask_filename = f"{base_name}.png"
    mask_path = os.path.join(output_dir, mask_filename)
    mask.save(mask_path)

    print(f"Saved combined mask {mask_filename} to {mask_path}")

def process_folder_of_jsons(json_folder, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over all JSON files in the folder
    for json_file in os.listdir(json_folder):
        if json_file.endswith('.json'):
            json_path = os.path.join(json_folder, json_file)
            create_mask_from_json(json_path, output_dir)

# Example usage
json_folder = '/home/perticon/Projects/deep/SimpleNet1/defect'  # Replace with the path to your folder containing JSON files
output_dir = '/home/perticon/Projects/deep/SimpleNet1/defect/masks'  # Replace with the path to save the mask images
process_folder_of_jsons(json_folder, output_dir)
