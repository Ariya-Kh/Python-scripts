from ultralytics import YOLO

# Train
model = YOLO("/home/perticon/Projects/deep/yolo-torch-2/runs/segment/train14-zamzam-best/weights/best.pt")  # load a pretrained model (recommended for training)

# model = YOLO("yolov8n-seg.pt")  # load a pretrained model (recommended for training)
# model = YOLO("/home/perticon/Projects/deep/yolo-torch-2/yolov8n-seg.pt")
# model.cuda()
# results = model.train(data="val.yaml", epochs=800, imgsz=960, batch=1, overlap_mask=False, device='cuda:0', save_period=10)

## Train with augment
# model = YOLO("/home/perticon/Downloads/epoch336.pt") # load a pretrained model (recommended for training)
# model = YOLO("/home/perticon/Projects/deep/yolo-torch-2/runs/segment/train12-zamzam-best/weights/last.pt") # load a pretrained model (recommended for training)
# model = YOLO("/home/perticon/Projects/deep/yolo-torch-2/runs/segment/train6/weights/epoch310.pt") # load a pretrained model (recommended for training)

model.cuda()
h = 960
w = 960 
results = model.train(data="zamzam.yaml",
epochs=1000, imgsz=960, batch=8, overlap_mask=False, device='cuda', save_period=2, patience=50,
augment = True,
hsv_h = 0.015,
hsv_s = 0.7,
hsv_v = 0.4,
degrees = 10,
translate= 0.1,
scale = 0.5,
shear = 5,
perspective = 0.0,
flipud = 0.0, 
fliplr = 0.5,   
mosaic = 0.0,
erasing=0.0,    
crop_fraction=0.0)

# Inference
# model = YOLO("/home/perticon/Projects/deep/yolo/runs/segment/train3/weights/best.pt")  # load a pretrained model (recommended for training)
# path = "/home/perticon/Projects/deep/datasets/zamzam/9"
# w = 960
# h = 960
# results = model.predict(path, save=True, imgsz=(h,w), conf=0.60, device='cuda') #, classes = [0,1,2]
