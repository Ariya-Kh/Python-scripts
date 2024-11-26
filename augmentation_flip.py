import cv2
import albumentations as A
import numpy as np
import os
from tqdm import tqdm
import json
import shutil

# Number of times to augment each image
num_augmentations = 2  # Set this to the desired number of augmentations

class TrackHorizontalFlip(A.HorizontalFlip):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.applied = False

    def __call__(self, force_apply=False, **kwargs):
        # Call the original transformation
        result = super().__call__(force_apply=force_apply, **kwargs)
        # Check if the transformation was applied
        if result['image'] is not None and self.p > 0 and np.random.rand() < self.p:
            self.applied = True
        return result
    
# Define the augmentation pipeline
# def transform(image):
#     flip_transform = A.HorizontalFlip(p=0.2)
#     other_transform = A.Compose([
#         A.ColorJitter(brightness=(0.8, 1), contrast=(0.8, 1), saturation=(0.8, 1), hue=(-0.05, 0.05), p=0.2),
#         A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.2),
#         A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
#         A.Blur(blur_limit=3, p=0.2),
#         A.RGBShift(r_shift_limit=(-5, 5), g_shift_limit=(-5, 5), b_shift_limit=(-5, 5), p=0.2),
#         A.HueSaturationValue(hue_shift_limit=(-5,5), sat_shift_limit=(-5,5), val_shift_limit=(-5,5), p=0.2)
#     ])

def transform(image):
    flip_transform = A.HorizontalFlip(p=0.0)
    other_transform = A.Compose([
        A.ColorJitter(brightness=(0.8, 1), contrast=(0.8, 1), saturation=(0.8, 1), hue=(-0.05, 0.05), p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.Blur(blur_limit=3, p=0.3),
        A.RGBShift(r_shift_limit=(-10, 10), g_shift_limit=(-10, 10), b_shift_limit=(-10, 10), p=0.3),
        A.HueSaturationValue(hue_shift_limit=(-5,5), sat_shift_limit=(-5,5), val_shift_limit=(-5,5), p=0.3)
    ])


    # Apply only the HorizontalFlip to check if it changes the image
    flipped_image = flip_transform(image=image)['image']
    transformed_image = other_transform(image=flipped_image)['image']

    # Check if the flipped image is different from the original
    if not np.array_equal(image, flipped_image):
        return True, transformed_image  # Flip was applied
    return False, transformed_image  # Flip was not applied

# Function to apply augmentations to an image
def augment_image(image_path, output_path):
    # Read the image using OpenCV
    image = cv2.imread(image_path)
    # Convert the image from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Apply the augmentations
    flag, augmented = transform(image)
    # Convert the augmented image back to BGR for saving
    augmented = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
    
    # Save the augmented image
    cv2.imwrite(output_path, augmented)

    return flag

def flip_points(input_file, output_file, img_width):
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Resize the points in the JSON data
    for shape in data['shapes']:
        points = shape['points']
        flipped_points = [[img_width - x, y] for x, y in points]

        shape['points'] = flipped_points
    data["imagePath"] = os.path.basename(output_file.rsplit('.', 1)[0] + '.jpg')
    data["imageData"] = None
    # Save the updated JSON data to the output file
    with open(output_file, 'w') as out_f:
        json.dump(data, out_f, indent=4)

def change_name(input_file, output_file, ext):
    with open(input_file, 'r') as f:
        data = json.load(f)
   
    data["imagePath"] = os.path.basename(output_file.rsplit('.', 1)[0] + ext)
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
                is_flipped = augment_image(input_file, output_file)
                
                # Handle corresponding JSON file if image was flipped
                json_file = f"{name}_{suffix}.json"
                if is_flipped:
                    json_input_file = os.path.join(input_dir, f"{name}.json")
                    json_output_file = os.path.join(output_dir, json_file)
                    
                    # Load the image to get its width
                    image = cv2.imread(input_file)
                    img_width = image.shape[1]
                    
                    # Copy the original JSON file with the new suffix
                    if os.path.exists(json_input_file):
                        # shutil.copy(json_input_file, json_output_file)
                        flip_points(json_input_file, json_output_file, img_width)
                else:
                    json_input_file = os.path.join(input_dir, f"{name}.json")
                    json_output_file = os.path.join(output_dir, json_file)

                    if os.path.exists(json_input_file):
                        # shutil.copy(json_input_file, os.path.join(output_dir, json_file))
                        change_name(json_input_file, json_output_file, ext)

# Example usage
input_directory = '/home/perticon/Projects/deep/yolo-torch-2/train_zamzam_defect_1403_9_5/images/train' 
output_directory = '/home/perticon/Projects/deep/yolo-torch-2/train_zamzam_defect_1403_9_5/images/train'
process_directory(input_directory, output_directory)
