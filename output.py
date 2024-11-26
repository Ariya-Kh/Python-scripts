import torch

# Load the TorchScript model
model = torch.jit.load('/home/perticon/Application/yolov5n6_i960_b6_v2.torchscript')  # Replace with your model's path
model.to('cuda')  # Move model to GPU
model.eval()  # Set the model to evaluation mode
# Prepare a sample input tensor
# Replace the dimensions with the expected input shape for your model
# For example, (batch_size, channels, height, width) = (1, 3, 640, 640)
input_tensor = torch.randn(10, 3, 960, 960).to('cuda')  # Adjust dimensions as needed
# Run inference
with torch.no_grad():  # Disable gradient tracking for inference
    output = model(input_tensor)

# Print the output
print("Model output shape:", output[0].shape)
print(output[0][0][0])