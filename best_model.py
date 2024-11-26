import os
import torch
from ultralytics import YOLO
import numpy as np
from glob import glob
import json

# Function to load ground truth boxes from JSON files (converting segments to bounding boxes)
def load_ground_truth_boxes(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    boxes = []
    for shape in data['shapes']:
        points = np.array(shape['points'])
        xmin, ymin = points.min(axis=0)
        xmax, ymax = points.max(axis=0)
        boxes.append([xmin, ymin, xmax, ymax])
    return boxes

# Function to calculate IoU between two boxes
def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# Function to evaluate a single model
def evaluate_model(model, image_folder, annotation_folder, conf = 0.2):
    iou_scores = []
    results = model.predict(source=image_folder, conf=conf)
    for result in results:
        image_path = os.path.basename(result.path)
        image_name = os.path.basename(image_path).replace(".jpg", "")
        annotation_path = os.path.join(annotation_folder, image_name + ".json")
        if not os.path.exists(annotation_path):
            continue

        ground_truth_boxes = load_ground_truth_boxes(annotation_path)    
        detected_boxes = []
        for bbox in result.boxes:
            box = bbox.xyxy[0].cpu().numpy()  # YOLO output in [x_min, y_min, x_max, y_max]
            detected_boxes.append(box)

        for gt_box in ground_truth_boxes:
            max_iou = 0
            for det_box in detected_boxes:
                iou = calculate_iou(gt_box, det_box)
                max_iou = max(max_iou, iou)
            iou_scores.append(max_iou)

    mean_iou = np.mean(iou_scores) if iou_scores else 0
    return mean_iou

# Function to evaluate all models and find the best one
def evaluate_models(model_folder, image_folder, annotation_folder, conf = 0.2):
    best_model = None
    best_iou = 0

    for model_path in glob(os.path.join(model_folder, "*.pt")):
        model = YOLO(model_path)
        mean_iou = evaluate_model(model, image_folder, annotation_folder, conf)
        print(f"Model {model_path} - Confidence: {conf} - Mean IoU: {mean_iou}")

        if mean_iou > best_iou:
            best_iou = mean_iou
            best_model = model_path

    print(f"Best model: {best_model} with Confidence: {conf} -> Mean IoU: {best_iou}")
    return best_model

# Define paths
model_folder = "/home/perticon/test1/models2"        # Folder containing .pt model files
image_folder = "/home/perticon/test1/behruz_imags"        # Folder containing images
annotation_folder = "/home/perticon/test1/behruz_imags"  # Folder containing ground truth JSON files
confidence = 0.3
# Evaluate all models and find the best one
best_model_path = evaluate_models(model_folder, image_folder, annotation_folder, confidence)
print(f"The best model is located at: {best_model_path}")
