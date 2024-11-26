from PIL import Image
import os

# Define the folder containing the images and the output folder
input_folder = "/media/perticon/Backup/datasets/zamzam/new_Zamzam_1403_09_05"
output_folder = "/media/perticon/Backup/datasets/zamzam/new_Zamzam_1403_09_05"

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Loop through all files in the input folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        # Full path to the input image
        input_path = os.path.join(input_folder, filename)
        
        # Open the image
        with Image.open(input_path) as img:
            # Rotate the image to the right (clockwise)
            rotated_img = img.rotate(-90, expand=True)
            
            # Full path to save the rotated image
            output_path = os.path.join(output_folder, filename)
            
            # Save the rotated image
            # rotated_img.save(output_path)
            rotated_img.save(output_path, format='JPEG', quality=100)
            print(f"Saved rotated image to {output_path}")
