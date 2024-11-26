import os
from PIL import Image

# Define the folder path containing images
folder_path = '/media/perticon/Backup/datasets/Kalleh/1403-06-17-Kalleh/5'

# Define the coordinates and dimensions for cropping (x, y, width, height)
# x, y, width, height = 0, 0, 1440, 350  # top values
x, y, width, height = 0, 125, 1440, 350  # middle values
# x, y, width, height = 0, 350, 1440, 350  # bottom values

# Create a folder to save cropped images if it doesn't exist
output_folder = os.path.join(folder_path, "cropped")
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Iterate over all the files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):  # Specify the image formats you want to process
        file_path = os.path.join(folder_path, filename)

        # Open the image
        with Image.open(file_path) as img:
            # Define the crop box (left, upper, right, lower)
            crop_box = (x, y, x + width, y + height)
            cropped_img = img.crop(crop_box)
            
            # Save the cropped image
            cropped_img.save(os.path.join(output_folder, f"cropped_{filename}"))

print(f"Cropped images are saved in {output_folder}")
