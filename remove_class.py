import json
import os

def remove_class_from_labelme_json_folder(input_folder, output_folder, class_name_to_remove):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Process each JSON file in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.json'):  # Only process JSON files
            input_file_path = os.path.join(input_folder, file_name)
            output_file_path = os.path.join(output_folder, file_name)
            
            # Open and modify the JSON file
            with open(input_file_path, 'r') as file:
                data = json.load(file)

            # Remove the specified class
            if 'shapes' in data:
                data['shapes'] = [shape for shape in data['shapes'] if shape['label'] != class_name_to_remove]

            # Save the updated JSON to the output folder
            with open(output_file_path, 'w') as file:
                json.dump(data, file, indent=4)

            print(f"Processed: {file_name} - Removed class '{class_name_to_remove}'")

# Example usage
input_folder_path = "/media/perticon/Backup/datasets/zamzam/bottles2"
output_folder_path = "/media/perticon/Backup/datasets/zamzam/bottles2"
class_to_remove = "Level"

remove_class_from_labelme_json_folder(input_folder_path, output_folder_path, class_to_remove)
