import cv2
import albumentations as A
import numpy as np
import os
from tqdm import tqdm
import json
import shutil

is_rotated = False  # Don't change this!
angle = 90
num_augmentations = 1  # Set this to the desired number of augmentations

class RotateWithAngle(A.Rotate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.applied_angle = None

    def __call__(self, force_apply=False, **kwargs):
        result = super().__call__(force_apply=force_apply, **kwargs)
        if result['image'] is not None and self.p > 0 and np.random.rand() < self.p:
            self.applied_angle = self.limit[0] if isinstance(self.limit, (tuple, list)) else self.limit
        return result

# Define the augmentation pipeline
def transform(image):
    rotate_transform = RotateWithAngle(limit=(-angle, -angle), border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0), p=1)  # Rotate by angle with black borders
    other_transform = A.Compose([
        A.ColorJitter(brightness=(0.8, 1), contrast=(0.8, 1), saturation=(0.8, 1), hue=(-0.05, 0.05), p=0.16),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.16),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.16),
        A.Blur(blur_limit=3, p=0.16),
        A.RGBShift(r_shift_limit=(-5, 5), g_shift_limit=(-5, 5), b_shift_limit=(-5, 5), p=0.16)
    ])

    # Apply only the Rotate to check if it changes the image
    rotate_transform_result = rotate_transform(image=image)
    rotated_image = rotate_transform_result['image']
    transformed_image = other_transform(image=rotated_image)['image']

    # Check if the rotated image is different from the original
    is_rotated = rotate_transform_result['image'] is not None and not np.array_equal(image, rotated_image)
    applied_angle = rotate_transform.applied_angle if is_rotated else None

    return is_rotated, transformed_image, applied_angle

# Function to apply augmentations to an image
def augment_image(image_path, output_path):
    # Read the image using OpenCV
    image = cv2.imread(image_path)
    # Convert the image from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Apply the augmentations
    is_rotated, augmented, applied_angle = transform(image)
    # Convert the augmented image back to BGR for saving
    augmented = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
    
    # Save the augmented image
    cv2.imwrite(output_path, augmented)

    return is_rotated, applied_angle

def rotate_points(input_file, output_file, angle_degrees):
    angle_radians = np.radians(angle_degrees)
    cos_theta = np.cos(angle_radians)
    sin_theta = np.sin(angle_radians)
    # Rotation matrix
    rotation_matrix = np.array([
        [cos_theta, -sin_theta],
        [sin_theta, cos_theta]
    ])

    with open(input_file, 'r') as f:
        data = json.load(f)
    
    img_width = data['imageWidth']
    img_height = data['imageHeight']
    data["imagePath"] = os.path.basename(output_file.rsplit('.', 1)[0] + '.jpg')
    data["imageData"] = None

    origin = (int(img_width / 2), int(img_height / 2))

    # Resize the points in the JSON data
    for shape in data['shapes']:
        points = shape['points']
        points = [tuple(point) for point in points]
        # Translate points to origin
        translated_points = np.array(points) - np.array(origin)
        # Apply rotation
        rotated_points = np.dot(translated_points, rotation_matrix.T)
        # Translate points back
        rotated_points += np.array(origin)
        rotated_points = [list(point) for point in rotated_points]

        # Clip points to be within image boundaries
        for point in rotated_points:
            if point[0] > img_width:
                point[0] = img_width
            if point[0] < 0:
                point[0] = 0
            if point[1] > img_height:
                point[1] = img_height
            if point[1] < 0:
                point[1] = 0        
        # rotated_points = [[
        #     max(0, min(point[0], img_width)),
        #     max(0, min(point[1], img_height))
        # ] for point in rotated_points]

        shape['points'] = rotated_points

    # Save the updated JSON data to the output file
    with open(output_file, 'w') as out_f:
        json.dump(data, out_f, indent=4)

def change_name(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
   
    data["imagePath"] = os.path.basename(output_file.rsplit('.', 1)[0] + '.jpg')
    data["imageData"] = None

    with open(output_file, 'w') as out_f:
        json.dump(data, out_f, indent=4)

# Function to process all images in the input directory
def process_directory(input_dir, output_dir):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files = [f for f in os.listdir(input_dir)]      

    # Iterate over all image files with a progress bar
    for file_name in tqdm(files, desc="Processing images"):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_file = os.path.join(input_dir, file_name)
            name, ext = os.path.splitext(file_name)
            
            for i in range(num_augmentations):
                suffix = chr(ord('a') + i)  # Create suffix 'a', 'b', 'c', etc.
                new_filename = f"{name}_{suffix}{ext}"
                output_file = os.path.join(output_dir, new_filename)

                # Apply augmentations and save the augmented image
                is_rotated, applied_angle = augment_image(input_file, output_file)

                # Handle corresponding JSON file if image was rotated
                json_file = f"{name}_{suffix}.json"
                if is_rotated:
                    json_input_file = os.path.join(input_dir, f"{name}.json")
                    json_output_file = os.path.join(output_dir, json_file)
                    
                    # Load the image to get its width
                    image = cv2.imread(input_file)
                    img_width = image.shape[1]
                    
                    # Copy the original JSON file with the new suffix
                    if os.path.exists(json_input_file):
                        # shutil.copy(json_input_file, json_output_file)
                        rotate_points(json_input_file, json_output_file, angle)
                else:
                    # Copy the original JSON file with the new suffix
                    json_input_file = os.path.join(input_dir, f"{name}.json")
                    json_output_file = os.path.join(output_dir, json_file)
                    if os.path.exists(json_input_file):
                        # shutil.copy(json_input_file, os.path.join(output_dir, json_file))
                        change_name(json_input_file, json_output_file)

# Example usage
input_directory = '/media/perticon/Backup/datasets/zamzam/jet_corrected_labels'
output_directory = '/media/perticon/Backup/datasets/zamzam/jet_corrected_labels_roatte'
process_directory(input_directory, output_directory)
