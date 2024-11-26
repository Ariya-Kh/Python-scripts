from ultralytics import YOLO
import os
import json
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use('tkAgg')

w = 960
h = 960
path = '/media/perticon/Backup/datasets/zamzam/zamzam_defect_office_1403_9_5/done'
# model = YOLO("/home/perticon/Projects/deep/yolo-torch-2/runs/segment/train3/weights/best.pt")
model = YOLO("/home/perticon/Projects/deep/yolo-torch-2/best-zamzam-yolo11.pt")

results = model.predict(path, save=False, imgsz=(h, w), conf=0.3, device='cuda:0')  # , classes = [0,1,2];

for r in tqdm(results):
    h = r.orig_shape[0]
    w = r.orig_shape[1]
    
    if r.masks is not None:
        json_name = os.path.splitext(r.path)[0] + ".json"  # JSON file corresponding to the image
        if os.path.exists(json_name):
            # Load existing JSON content
            with open(json_name, "r") as f:
                json_content = json.load(f)
        else:
            # Create a new JSON structure if file doesn't exist
            json_content = {
                "version": "5.4.1",
                "flags": {},
                "shapes": [],
                "imagePath": os.path.basename(r.path),
                "imageData": None,
                "imageHeight": h,
                "imageWidth": w,
            }
        
        # Generate shapes from results
        a = r.to_json()
        classes = json.loads(a)
        new_shapes = []
        for cls in classes:
            shape = {}
            points = []
            shape["label"] = cls["name"]
            # points
            x = cls["segments"]["x"]
            y = cls["segments"]["y"]
            for i in range(len(x)):
                point = []
                point.append(x[i])
                point.append(y[i])
                points.append(point)
            shape["points"] = points
            shape["group_id"] = None
            shape["description"] = ""
            shape["shape_type"] = "polygon"
            shape["flags"] = {}
            shape["mask"] = None

            new_shapes.append(shape)

        # Append new shapes to existing shapes
        json_content["shapes"].extend(new_shapes)
        
        # Save updated JSON file
        with open(json_name, "w") as f:
            json.dump(json_content, f, indent=4)