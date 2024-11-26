import torch
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("/home/perticon/Projects/deep/yolo-torch-2/runs/segment/train14-zamzam-best/weights/epoch0.pt")

# Simple NMS function compatible with TorchScript
def simple_nms(predictions, conf_thres=0.25, iou_thres=0.45):
    # Apply confidence threshold
    mask = predictions[..., 4] > conf_thres  # Confidence score threshold
    predictions = predictions[mask]

    # This example omits IoU calculation for simplicity; a full NMS implementation may be required here
    # Placeholder: return the filtered predictions directly for testing purposes
    return predictions

# Define a model wrapper that uses the simplified NMS function
class YOLOv8WithSimpleNMS(torch.nn.Module):
    def __init__(self, model, conf_thres=0.25, iou_thres=0.45):
        super(YOLOv8WithSimpleNMS, self).__init__()
        self.model = model.model  # Access the core model
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def forward(self, x):
        # Run the model's inference
        preds = self.model(x)
        
        # Apply simple NMS to the predictions
        results = simple_nms(preds, self.conf_thres, self.iou_thres)
        
        return results

# Wrap the original model with NMS applied in the forward pass
model_with_nms = YOLOv8WithSimpleNMS(model)

# Move model to GPU if available
model_with_nms = model_with_nms.cuda()

# Example input for tracing (assuming input size 960x960)
example_input = torch.rand(2, 3, 960, 960).cuda()  # Batch size of 2 for example

# Export the model with NMS applied as TorchScript
scripted_model = torch.jit.trace(model_with_nms, example_input)
scripted_model.save("/home/perticon/yolov8_with_nms.torchscript")

print("Model exported to TorchScript with simple NMS applied.")
