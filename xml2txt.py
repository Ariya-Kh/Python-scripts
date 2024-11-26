import os
import xml.etree.ElementTree as ET

def convert_voc_to_yolo(xml_folder, txt_folder, classes):
    """
    Converts Pascal VOC XML annotations to YOLO TXT format.
    
    Args:
        xml_folder (str): Path to the folder containing XML files.
        txt_folder (str): Path to the folder to save YOLO TXT files.
        classes (list): List of class names in the order of their class IDs.
    """
    if not os.path.exists(txt_folder):
        os.makedirs(txt_folder)

    for xml_file in os.listdir(xml_folder):
        if not xml_file.endswith(".xml"):
            continue
        
        # Parse XML file
        tree = ET.parse(os.path.join(xml_folder, xml_file))
        root = tree.getroot()
        
        # Get image dimensions
        size = root.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)
        
        # Prepare YOLO TXT file
        txt_file_path = os.path.join(txt_folder, os.path.splitext(xml_file)[0] + ".txt")
        with open(txt_file_path, "w") as txt_file:
            # Iterate through each object in the XML file
            for obj in root.findall("object"):
                class_name = obj.find("name").text
                if class_name not in classes:
                    print(f"Warning: Class '{class_name}' not in class list. Skipping...")
                    continue
                
                class_id = classes.index(class_name)
                
                # Get bounding box coordinates
                bndbox = obj.find("bndbox")
                xmin = int(bndbox.find("xmin").text)
                ymin = int(bndbox.find("ymin").text)
                xmax = int(bndbox.find("xmax").text)
                ymax = int(bndbox.find("ymax").text)
                
                # Convert to YOLO format
                x_center = ((xmin + xmax) / 2) / width
                y_center = ((ymin + ymax) / 2) / height
                box_width = (xmax - xmin) / width
                box_height = (ymax - ymin) / height
                
                # Write to TXT file
                txt_file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")

    print(f"Conversion completed! YOLO TXT files saved in '{txt_folder}'.")

# Example usage
xml_folder = "/media/perticon/Backup/datasets/panter/Pen/tt"
# print(os.listdir(xml_folder))
txt_folder = "/media/perticon/Backup/datasets/panter/Pen/tt"
classes = ["0.7mm", "1mm", "Defect"]  # Replace with your class names

convert_voc_to_yolo(xml_folder, txt_folder, classes)
