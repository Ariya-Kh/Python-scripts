from ultralytics import YOLO
import os
import json
from tqdm import tqdm
import matplotlib
from ultralytics import SAM
matplotlib.use('tkAgg')
import numpy as np
import cv2  
w = 1080
h = 1440
path = '/media/perticon/Backup/datasets/zamzam/2024_06_24 Limonade 6mm/others (1)/150-7/not edited/'
# model = YOLO("/home/perticon/Downloads/best (1).pt")
model = SAM("sam2_b.pt")
model.cuda()
points = [
    [585, 795],
    [100, 660]  # First bounding b# Third bounding box
    # Add more bounding boxes as needed
]
results = model(path, points=points)
# results[0].save(filename='/home/perticon/res.jpg')
for r in tqdm(results):
    # r.show()
    h = r.orig_shape[0]
    w = r.orig_shape[1]
    masks = r.masks.data.cpu().numpy()  # Convert the tensor to a NumPy array and move it to CPU if it's on GPU

    # Initialize an empty list to store all contour points
    all_contours = []

    # Iterate over each mask
    for mask in masks:
        # Convert the boolean mask to a binary image (0s and 255s)
        binary_mask = (mask * 255).astype(np.uint8)

        # Find contours in the binary mask
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = []
        largest_contour = None
        max_area = 0
        for contour in contours:
            # Calculate the area of the contour
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                largest_contour = contour
        # Store the contour points
        if(len(contours) > 1):
            for contour in contours:
                if len(contour) == len(largest_contour):
                    contour_points = contour.squeeze()  # Remove the extra dimension to get [n, 2] format
                    if contour_points.ndim == 1:  # Handle cases where contour is a single point
                        contour_points = np.expand_dims(contour_points, axis=0)

                    # Find the maximum y-coordinate in the contour
                    max_y = np.max(contour_points[:, 1])

                    # Filter out points with y-coordinate between max_y-20 and max_y
                    filtered_points = contour_points[contour_points[:, 1] < max_y - 10]

                    # Add the filtered points to the list if there are any remaining points
                    if filtered_points.size > 0:
                        filtered_contours.append(filtered_points)
                else:
                    contour_points = contour.squeeze()  # Remove the extra dimension to get [n, 2] format
                    if contour_points.ndim == 1:  # Handle cases where contour is a single point
                        contour_points = np.expand_dims(contour_points, axis=0)

                    # Find the minimum y-coordinate in the contour
                    min_y = np.min(contour_points[:, 1])

                    # Filter out points with y-coordinate between min_y and min_y + 20
                    filtered_points = contour_points[contour_points[:, 1] > min_y + 20]

                    # Add the filtered points to the list if there are any remaining points
                    if filtered_points.size > 0:
                        filtered_contours.append(filtered_points)
            filtered_contours = np.vstack(filtered_contours)
                  
        all_contours.append(filtered_contours)                    



    for mask in r.masks.xy:
    # Extract y-coordinates (assuming mask is a 2D array with shape [N, 2])
        y_coords = mask[:, 1]  # Take the second column, which corresponds to y-coordinates
        max_y = max(max_y, np.max(y_coords))
        print("Maximum y-coordinate across all masks:", max_y)
    if r.masks != None:
        json_content = {}
        json_content["version"] = "5.4.1"
        json_content["flags"] = {}
        shapes = []
        
        for xy in r.masks.xy:
            shape = {}
            points = []
            shape["label"] = "Bottle"
            # points
            for x, y in xy:
                point = []
                point.append(int(x))
                point.append(int(y))
                points.append(point) 
            shape["points"] = points
            shape["group_id"] = None
            shape["description"] = ""
            shape["shape_type"] = "polygon"
            shape["flags"] = {}
            shape["mask"] = None

            shapes.append(shape)
        json_content["shapes"] = shapes
        json_content["imagePath"] = os.path.basename(r.path)
        json_content["imageData"] = None
        json_content["imageHeight"] = h
        json_content["imageWidth"] = w
        json_object = json.dumps(json_content)
        json_name = r.path.split(".")[0]+".json"  
        with open(json_name, "w") as f:
            f.write(json_object)
     



