from ultralytics import YOLO
import os
import json
from tqdm import tqdm
import numpy as np
import matplotlib
import cv2
matplotlib.use('tkAgg')
w = 1280
h = 960
# path = '/media/perticon/Backup/datasets/zamzam/2024_06_24 Limonade 6mm/others (1)/ZAMZAM_L_6mm32448 24-06-24 13-30-42.jpg'

path = '/media/perticon/Backup/datasets/panter/Pen/send'
# model = YOLO("/home/perticon/Projects/deep/yolo-torch-2/runs/segment/train3/weights/best.pt")
model = YOLO("/home/perticon/Projects/deep/yolo-torch-2/panter/train2/weights/best.pt")
# model = YOLO("/home/perticon/Projects/deep/yolo-torch-2/yolov8s-worldv2.pt")
# model.set_classes(["box with white color on the edge"])

# results = model.predict(path) #, classes = [0,1,2]

results = model.predict(path, save=False, imgsz=(h,w), conf=0.5,device='cuda:0', iou=0.7) #, classes = [0,1,2]
listt = ["0.7mm", "1mm", "Defect"]
classes = [(153, 255, 0), (0, 153, 255), (0,0,255)]
# Process detection results
for result in results:
    image = cv2.imread(result.path)
    for box in result.boxes.data.tolist():  # Each box contains [x1, y1, x2, y2, confidence, class_id]
        x1, y1, x2, y2 = map(int, box[:4])  # Extract box coordinates and convert to integers
        class_id = int(box[5])  # Extract class ID
        label = listt[class_id]  # Replace this with your class name mapping if available
        # Draw the bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), classes[class_id], 2)

        # Put the label (without confidence)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, classes[class_id], 2)
    cv2.imwrite(result.path, image)  # Save the result


# Save or display the image

# for result in tqdm(results):
    # result.save(filename=result.path, boxes=False)  # save to disk

    # result.show()