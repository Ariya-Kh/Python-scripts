from ultralytics import SAM

# Load a model
model = SAM("sam2_s.pt")

# Display model information (optional)
# model.info()
bboxes = [
    [478, 139, 745, 1297],  # First bounding box
    [721, 127, 1018, 1317]   # Third bounding box
    # Add more bounding boxes as needed
]

points = [
    [602, 705],  # First bounding box
    [886, 724],
    [85, 717],
    [336,714]# Third bounding box
    # Add more bounding boxes as needed
]
results = model('/home/perticon/1.jpg', points=points)
for r in results:
  print(r.masks)
  r.show()
  r.save(filename="result_p.jpg")  # save to disk
