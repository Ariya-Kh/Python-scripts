from ultralytics import YOLO
import os
import json
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use('tkAgg')
w = 960
h = 960
path = '/media/perticon/Backup/datasets/zamzam/new_Zamzam_1403_09_05'
# model = YOLO("/home/perticon/Projects/deep/yolo-torch-2/runs/segment/train3/weights/best.pt")
model = YOLO("/home/perticon/Projects/deep/yolo-torch-2/best-zamzam-yolo11.pt")

results = model.predict(path, save=False, imgsz=(h,w), conf=0.3, device='cuda:0') #, classes = [0,1,2];

for r in tqdm(results):
    h = r.orig_shape[0]
    w = r.orig_shape[1]
    
    if r.masks != None:
        json_content = {}
        json_content["version"] = "5.4.1"
        json_content["flags"] = {}
        a = r.to_json()
        classes = json.loads(a)
        shapes = []

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
     



