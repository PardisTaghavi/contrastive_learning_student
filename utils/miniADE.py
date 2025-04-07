'''#create a miniade dataset propotional to the original dataset
folder = '/media/avalocal/T7/ADE/ADEChallengeData2016'
# images / annotations / annotations_instance / annotations_detectron2
#images : /media/avalocal/T7/ADE/ADEChallengeData2016/images/training/ADE_train_00000001.jpg
#annotations: /media/avalocal/T7/ADE/ADEChallengeData2016/annotations/training/ADE_train_00000001.png
#annotations_instance: /media/avalocal/T7/ADE/ADEChallengeData2016/annotations_instance/training/ADE_train_00000001.png
#annotations_detectron2: /media/avalocal/T7/ADE/ADEChallengeData2016/annotations_detectron2/training/ADE_train_00000001.png

filder_mini = '/media/avalocal/T7/ADE/ADEChallengeData2016_mini'
# images / annotations / annotations_instance / annotations_detectron2
#images : /media/avalocal/T7/ADE/ADEChallengeData2016_mini/images/training/ADE_train_00000001.jpg
#annotations: /media/avalocal/T7/ADE/ADEChallengeData2016_mini/annotations/training/ADE_train_00000001.png
#annotations_instance: /media/avalocal/T7/ADE/ADEChallengeData2016_mini/annotations_instance/training/ADE_train_00000001.png
#annotations_detectron2: /media/avalocal/T7/ADE/ADEChallengeData2016_mini/annotations_detectron2/training/ADE_train_00000001.png

scene_categories = "/media/avalocal/T7/ADE/ADEChallengeData2016/sceneCategories.txt"

# ADE_train_00000001 airport_terminal
# ADE_train_00000002 airport_terminal
# ADE_train_00000003 art_gallery
# ADE_train_00000004 badlands
# ADE_train_00000005 ball_pit
# ADE_train_00000006 bathroom
# ADE_train_00000007 bathroom
# ADE_train_00000008 bathroom
# ADE_train_00000009 bathroom

#20.2k --> 5k
# 1. read scene categories
# 20.2k/5k = 4.04
# 2. for each scene category, select 4 images randomly if the number of images is greater than 4
# 3. copy the selected images to the new folder
# 4. copy the corresponding annotations, annotations_instance, annotations_detectron2 to the new folder
# 5. update the scene categories file for mini folder'''

import os
import random
import shutil
import json
from collections import defaultdict
from tqdm import tqdm

# Define source and destination folders
src_folder = '/media/avalocal/T7/ADE/ADEChallengeData2016'
dst_folder = '/media/avalocal/T7/ADE/ADEChallengeData2016_mini'

# Define subdirectories for the different file types (training subset assumed)
subdirs = {
    'images': 'jpg',  # original images are .jpg files
    'annotations': 'png',
    'annotations_instance': 'png',
    'annotations_detectron2': 'png'
}
split_data = 'training'  # or 'training'

# Create destination directory structure if it doesn't exist
for subdir in subdirs.keys():
    dst_subdir = os.path.join(dst_folder, subdir, split_data)
    os.makedirs(dst_subdir, exist_ok=True)

# Read scene categories from the original file
if split_data == 'training':
    scene_categories_file = os.path.join(src_folder, 'sceneCategories_train.txt')
    instance_json = os.path.join(src_folder, 'ade20k_instance_train.json')
else:
    scene_categories_file = os.path.join(src_folder, 'sceneCategories_val.txt')
    instance_json = os.path.join(src_folder, 'ade20k_instance_val.json')

category_dict = defaultdict(list)

with open(scene_categories_file, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 2:
            image_id, category = parts
            category_dict[category].append(image_id)

# For each scene category, select up to 4 images randomly
selected_images = []  # holds tuples of (image_id, category)
for category, images in category_dict.items():
    if len(images) > 4:
        selected = random.sample(images, 4)
    else:
        #if less than 4 images, select all
        selected = images
    selected_images.extend([(img, category) for img in selected])

# Copy files for each selected image from each subdirectory
for img, category in tqdm(selected_images, desc="Copying files"):
    # Determine file names: images are jpg, annotations are png
    filename_img = img + '.jpg'
    filename_ann = img + '.png'
    
    # Loop over the different subdirectories to copy files
    for subdir, ext in subdirs.items():
        filename = filename_img if ext == 'jpg' else filename_ann
        src_path = os.path.join(src_folder, subdir, split_data, filename)
        dst_path = os.path.join(dst_folder, subdir, split_data, filename)
        
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
        else:
            print(f"Warning: {src_path} does not exist!")

# Update the scene categories file in the mini dataset folder
if split_data == 'training':
    mini_scene_categories_file = os.path.join(dst_folder, 'sceneCategories_train.txt')
else:
    mini_scene_categories_file = os.path.join(dst_folder, 'sceneCategories_val.txt')

with open(mini_scene_categories_file, 'w') as f:
    for img, category in selected_images:
        f.write(f"{img} {category}\n")

# --- Create filtered instance JSON for the mini dataset ---

# Load the original instance JSON
with open(instance_json, 'r') as f:
    instance_data = json.load(f)

# Build a set of selected image IDs (base names without extension)
selected_ids = set([img for img, _ in selected_images])

# Filter images: assume that each image entry has a "file_name" field like "img.jpg"
mini_images = []
for image in instance_data.get("images", []):
    # Remove the extension to get the base id
    base_id, _ = os.path.splitext(image.get("file_name", ""))
    if base_id in selected_ids:
        mini_images.append(image)

# Filter annotations based on the selected image IDs
mini_annotations = [ann for ann in instance_data.get("annotations", []) if str(ann.get("image_id")) in selected_ids]

# Assemble the mini instance JSON
mini_instance_data = {
    "images": mini_images,
    "annotations": mini_annotations,
    "categories": instance_data.get("categories", [])
}

# Save the new instance JSON to the mini dataset folder (same name as original)
dst_instance_json = os.path.join(dst_folder, os.path.basename(instance_json))
with open(dst_instance_json, 'w') as f:
    json.dump(mini_instance_data, f, indent=2)

print("Mini dataset and instance JSON created successfully!")
print(f"Mini dataset folder: {dst_folder}")
print(f"Mini instance JSON: {dst_instance_json}")




