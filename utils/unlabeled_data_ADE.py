import os
import random
import shutil
import json
from collections import defaultdict
from tqdm import tqdm

# Path definitions
src_folder = '/media/avalocal/T7/ADE/ADEChallengeData2016'
lbl_folder = '/media/avalocal/T7/ADE/ADE_L'
ulbl_folder = '/media/avalocal/T7/ADE/ADE_U'

# Ensure target directories exist
for folder in [
    os.path.join(ulbl_folder, 'images', 'training'),
    os.path.join(ulbl_folder, 'annotations', 'training'),
    os.path.join(ulbl_folder, 'annotations_instance', 'training'),
    os.path.join(ulbl_folder, 'annotations_detectron2', 'training')
]:
    os.makedirs(folder, exist_ok=True)

# 1. Identify images already in the labeled folder
labeled_images = set()
if os.path.exists(os.path.join(lbl_folder, 'images', 'training')):
    labeled_images = set(os.listdir(os.path.join(lbl_folder, 'images', 'training')))
    print(f"Found {len(labeled_images)} images in labeled dataset")

# 2. Get all images from source folder
source_images = set()
if os.path.exists(os.path.join(src_folder, 'images', 'training')):
    source_images = set(os.listdir(os.path.join(src_folder, 'images', 'training')))
    print(f"Found {len(source_images)} images in source dataset")

# 3. Identify images for unlabeled dataset (images in source but not in labeled)
unlabeled_images = source_images - labeled_images
print(f"Selected {len(unlabeled_images)} images for unlabeled dataset")

# 4. Load source JSON file
src_json_path = os.path.join(src_folder, 'ade20k_instance_train.json')
if not os.path.exists(src_json_path):
    print(f"Source JSON file not found at {src_json_path}")
    exit(1)

with open(src_json_path, 'r') as f:
    src_json = json.load(f)

# 5. Create new JSON for unlabeled dataset
ulbl_json = {
    'info': src_json.get('info', {}),
    'licenses': src_json.get('licenses', []),
    'categories': src_json.get('categories', []),
    'images': [],
    'annotations': []
}

# Map to track image ids
image_id_map = {}  # old_id -> new_id

# 6. Filter images and annotations for unlabeled dataset
print("Creating JSON for unlabeled dataset...")
new_image_id = 1
new_annotation_id = 1

# First, select images for unlabeled dataset
for img in src_json.get('images', []):
    file_name = img.get('file_name', '').split('/')[-1]
    if file_name in unlabeled_images:
        old_id = img['id']
        image_id_map[old_id] = new_image_id
        
        # Update image entry with new id
        updated_img = img.copy()
        updated_img['id'] = new_image_id
        # Make sure file_name has the correct path with training folder
        if 'file_name' in updated_img and not updated_img['file_name'].startswith('training/'):
            file_name = updated_img['file_name'].split('/')[-1]
            updated_img['file_name'] = f"training/{file_name}"
        ulbl_json['images'].append(updated_img)
        
        new_image_id += 1

# Then, select corresponding annotations
for ann in src_json.get('annotations', []):
    old_image_id = ann.get('image_id')
    if old_image_id in image_id_map:
        # Update annotation with new ids
        updated_ann = ann.copy()
        updated_ann['id'] = new_annotation_id
        updated_ann['image_id'] = image_id_map[old_image_id]
        ulbl_json['annotations'].append(updated_ann)
        
        new_annotation_id += 1

# 7. Save new JSON file
ulbl_json_path = os.path.join(ulbl_folder, 'ade20k_instance_train.json')
with open(ulbl_json_path, 'w') as f:
    json.dump(ulbl_json, f)
print(f"Created new JSON with {len(ulbl_json['images'])} images and {len(ulbl_json['annotations'])} annotations")

# 8. Copy files to unlabeled folder
print("Copying files to unlabeled folder...")

# Dictionary to track files by type
files_to_copy = {
    'images': [],
    'annotations': [],
    'annotations_instance': [],
    'annotations_detectron2': []
}

# Prepare list of files to copy
for img_file in unlabeled_images:
    base_name = os.path.splitext(img_file)[0]
    
    # Images (.jpg)
    src_img = os.path.join(src_folder, 'images', 'training', img_file)
    dst_img = os.path.join(ulbl_folder, 'images', 'training', img_file)
    if os.path.exists(src_img):
        files_to_copy['images'].append((src_img, dst_img))
    
    # Annotations (.png)
    ann_file = f"{base_name}.png"
    src_ann = os.path.join(src_folder, 'annotations', 'training', ann_file)
    dst_ann = os.path.join(ulbl_folder, 'annotations', 'training', ann_file)
    if os.path.exists(src_ann):
        files_to_copy['annotations'].append((src_ann, dst_ann))
    
    # Instance annotations (.png)
    src_inst = os.path.join(src_folder, 'annotations_instance', 'training', ann_file)
    dst_inst = os.path.join(ulbl_folder, 'annotations_instance', 'training', ann_file)
    if os.path.exists(src_inst):
        files_to_copy['annotations_instance'].append((src_inst, dst_inst))
    
    # Detectron2 annotations (JSON files)
    det_file = f"{base_name}.json"
    src_det = os.path.join(src_folder, 'annotations_detectron2', 'training', det_file)
    dst_det = os.path.join(ulbl_folder, 'annotations_detectron2', 'training', det_file)
    if os.path.exists(src_det):
        files_to_copy['annotations_detectron2'].append((src_det, dst_det))

# Copy files with progress bar
for file_type, file_pairs in files_to_copy.items():
    print(f"Copying {len(file_pairs)} {file_type}...")
    for src, dst in tqdm(file_pairs):
        shutil.copy2(src, dst)

print("Dataset preparation complete:")
print(f"- Source dataset: {len(source_images)} images")
print(f"- Labeled dataset: {len(labeled_images)} images")
print(f"- Unlabeled dataset: {len(unlabeled_images)} images")
print(f"Files copied to {ulbl_folder}")