import os
import random
import shutil
from pathlib import Path

# Define source and destination folders
ADE_U_folder = "/media/avalocal/T7/ADE/ADEChallengeData2016_complete"
ADE_folder = "/media/avalocal/T7/ADE/ADEChallengeData2016"

# Define the subfolders to process
subfolders = [
    os.path.join("images", "training"),
    os.path.join("annotations", "training"),
    os.path.join("annotations_instance", "training")
]

def main():
    # Get all base filenames (without extension) from the images/training folder
    image_folder = os.path.join(ADE_U_folder, "images", "training")
    all_image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
    
    # Extract base names without extensions
    all_base_names = [os.path.splitext(f)[0] for f in all_image_files]
    
    # Randomly select 400 sample names
    if len(all_base_names) < 100:
        print(f"Warning: Only {len(all_base_names)} samples available, using all of them.")
        selected_base_names = all_base_names
    else:
        selected_base_names = random.sample(all_base_names, 100)
    
    print(f"Selected {len(selected_base_names)} random samples.")
    
    # Copy files for each selected sample
    for base_name in selected_base_names:
        # Copy images (.jpg)
        copy_file(
            os.path.join(ADE_U_folder, "images", "training", f"{base_name}.jpg"),
            os.path.join(ADE_folder, "images", "training", f"{base_name}.jpg")
        )
        
        # Copy annotations (.png)
        copy_file(
            os.path.join(ADE_U_folder, "annotations", "training", f"{base_name}.png"),
            os.path.join(ADE_folder, "annotations", "training", f"{base_name}.png")
        )
        
        # Copy annotations_instance (.png)
        copy_file(
            os.path.join(ADE_U_folder, "annotations_instance", "training", f"{base_name}.png"),
            os.path.join(ADE_folder, "annotations_instance", "training", f"{base_name}.png")
        )
    
    print("Copy completed successfully.")

def copy_file(src, dst):
    # Create destination directory if it doesn't exist
    dst_dir = os.path.dirname(dst)
    os.makedirs(dst_dir, exist_ok=True)
    
    # Copy the file
    shutil.copy2(src, dst)
    
if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    main()
