import os
import xml.etree.ElementTree as ET

def yolo_to_xml(yolo_txt_path, img_path, img_width, img_height, class_names):
    # Get file names without extensions
    file_name = os.path.basename(img_path)
    file_name_no_ext = os.path.splitext(file_name)[0]
    
    # Create the root element of the XML
    annotation = ET.Element("annotation")
    
    folder = ET.SubElement(annotation, "folder")
    folder.text = os.path.basename(os.path.dirname(img_path))
    
    filename = ET.SubElement(annotation, "filename")
    filename.text = file_name
    
    path = ET.SubElement(annotation, "path")
    path.text = img_path
    
    source = ET.SubElement(annotation, "source")
    database = ET.SubElement(source, "database")
    database.text = "Unknown"
    
    size = ET.SubElement(annotation, "size")
    width = ET.SubElement(size, "width")
    width.text = str(img_width)
    height = ET.SubElement(size, "height")
    height.text = str(img_height)
    depth = ET.SubElement(size, "depth")
    depth.text = "3"
    
    segmented = ET.SubElement(annotation, "segmented")
    segmented.text = "0"
    
    # Read YOLO txt file
    with open(yolo_txt_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            bbox_width = float(parts[3])
            bbox_height = float(parts[4])
            
            # Calculate xmin, ymin, xmax, ymax
            xmin = int((x_center - bbox_width / 2) * img_width)
            ymin = int((y_center - bbox_height / 2) * img_height)
            xmax = int((x_center + bbox_width / 2) * img_width)
            ymax = int((y_center + bbox_height / 2) * img_height)
            
            object = ET.SubElement(annotation, "object")
            name = ET.SubElement(object, "name")
            name.text = class_names[class_id] if class_id < len(class_names) else "unknown"
            pose = ET.SubElement(object, "pose")
            pose.text = "Unspecified"
            truncated = ET.SubElement(object, "truncated")
            truncated.text = "0"
            difficult = ET.SubElement(object, "difficult")
            difficult.text = "0"
            bndbox = ET.SubElement(object, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(xmin)
            ET.SubElement(bndbox, "ymin").text = str(ymin)
            ET.SubElement(bndbox, "xmax").text = str(xmax)
            ET.SubElement(bndbox, "ymax").text = str(ymax)
    
    # Write the tree to an XML file
    tree = ET.ElementTree(annotation)
    output_path = os.path.join(os.path.dirname(yolo_txt_path), file_name_no_ext + ".xml")
    tree.write(output_path)
    print(f"Saved XML to {output_path}")

# Example usage
image_path = "/home/perticon/Projects/deep/datasets/behrooz/yolo/image_30.jpg"
yolo_txt_path = "/home/perticon/Projects/deep/datasets/behrooz/yolo/output/image_30.txt"
img_width = 1440  # Use the actual width of the original image
img_height = 1080  # Use the actual height of the original image

# Define the class names (ensure this matches your dataset)
class_names = ["defect"]

# Convert YOLO to XML
yolo_to_xml(yolo_txt_path, image_path, img_width, img_height, class_names)