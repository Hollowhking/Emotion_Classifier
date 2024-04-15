import os

import os
import shutil

def copy_script_to_folders(script_path):
    # Get the current directory
    current_directory = os.getcwd()

    # List all entries (files and folders) in the current directory
    all_entries = os.listdir(current_directory)

    # Filter out only the folder names
    folder_names = [entry for entry in all_entries if os.path.isdir(entry)]

    # Loop through each folder and copy the script
    for folder_name in folder_names:
        folder_path = os.path.join(current_directory, folder_name)
        script_destination = os.path.join(folder_path, os.path.basename(script_path))

        # Copy the script to the folder
        shutil.copy(script_path, script_destination)
        print(f"Script copied to: {folder_path}")

# Example usage:
script_to_copy = "runnow.py"  # Replace with the actual script file path
copy_script_to_folders(script_to_copy)


# current_directory = os.getcwd()

# all_entries = os.listdir(current_directory)

#     # Filter out only the folder names
# folder_names = [entry for entry in all_entries if os.path.isdir(entry)]

# for folder_name in folder_names:
#     folder_path = os.path.join(current_directory, folder_name)
#     print("Processing folder:", folder_path)

# def create_empty_folders(num_folders):
#     base_folder = os.getcwd()
#     for i in range(309, num_folders + 1):
#         folder_path = os.path.join(base_folder, f"{i}")
#         os.makedirs(folder_path)
#         print(f"Folder created: {folder_path}")

# # Example usage
# num_folders = 492

# create_empty_folders(num_folders)
