from ultralytics import YOLO
from ultralytics.utils import ops


# model = YOLO("/home/perticon/test1/best_test.pt") # load a pretrained model (recommended for training)
model = YOLO("yolov8n-seg.pt") # load a pretrained model (recommended for training)

model.cuda()
# results = model_with_nms.model.predict(path, imgsz=(960,960), conf=0.4,device='cuda:0') #, classes = [0,1,2]
# model.export(format="torchscript",imgsz=(960,960), batch=1, device='cuda')

model.export(format="onnx", imgsz=(960,960), batch=2, device='cuda')
# model.export(format="torchscript", imgsz=(320,288), batch=2, device='cuda')

    