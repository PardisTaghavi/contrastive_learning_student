import os
import random
import shutil
import json
from collections import defaultdict
from tqdm import tqdm

# Define source and destination folders
src_folder = '/media/avalocal/T7/ADE/ADEChallengeData2016_mini'
dst_folder = '/media/avalocal/T7/ADE/ADE_L'

# Define subdirectories for the different file types
subdirs = {
    'images': 'jpg',  # original images are .jpg files
    'annotations': 'png',
    'annotations_instance': 'png',
    'annotations_detectron2': 'png'
}
split_data = 'training'  # or 'validation'

# Create destination directory structure if it doesn't exist
for subdir in subdirs.keys():
    dst_subdir = os.path.join(dst_folder, subdir, split_data)
    os.makedirs(dst_subdir, exist_ok=True)

# Load instance JSON file which contains thing classes and images
if split_data == 'training':
    instance_json = os.path.join(src_folder, 'ade20k_instance_train.json')
else:
    instance_json = os.path.join(src_folder, 'ade20k_instance_val.json')

print(f"Loading instance JSON from {instance_json}")
with open(instance_json, 'r') as f:
    instance_data = json.load(f)

# Get all categories (thing classes)
categories = instance_data.get('categories', [])
print(f"Found {len(categories)} thing classes in the dataset") #100

# Build a mapping from image_id to file_name
image_id_to_filename = {}
for image in instance_data.get('images', []):
    image_id_to_filename[image['id']] = image['file_name']

# Create a dictionary mapping category_id to images containing that object
category_to_images = defaultdict(set)
for annotation in tqdm(instance_data.get('annotations', []), desc="Analyzing annotations"):
    category_id = annotation['category_id']
    image_id = annotation['image_id']
    if image_id in image_id_to_filename:
        # Store the filename without extension as the image identifier
        filename = image_id_to_filename[image_id]
        base_name, _ = os.path.splitext(filename)
        category_to_images[category_id].add(base_name)

# Select up to 5 images per category, aiming for a total of around 500 images
selected_images = set()
selected_by_category = {}

# Sort categories by ID for deterministic results
for category in sorted(categories, key=lambda x: x['id']):
    category_id = category['id']
    category_name = category['name']
    
    # Get images for this category
    images_with_category = list(category_to_images[category_id])
    
    # Select up to 5 images randomly for this category
    few = 17
    if len(images_with_category) > few:
        selected = random.sample(images_with_category, few)
    else:
        selected = images_with_category
    
    # Add to our tracking sets
    selected_by_category[category_id] = selected
    selected_images.update(selected)
    
    print(f"Selected {len(selected)} images for category '{category_name}' (id: {category_id})")

print(f"Total unique images selected: {len(selected_images)}")

# Copy files for each selected image from each subdirectory
for img_id in tqdm(selected_images, desc="Copying files"):
    # Determine file names: images are jpg, annotations are png
    filename_img = f"{img_id}.jpg"
    filename_ann = f"{img_id}.png"
    
    # Loop over the different subdirectories to copy files
    for subdir, ext in subdirs.items():
        filename = filename_img if ext == 'jpg' else filename_ann
        src_path = os.path.join(src_folder, subdir, split_data, filename)
        dst_path = os.path.join(dst_folder, subdir, split_data, filename)
        
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
        else:
            print(f"Warning: {src_path} does not exist!")

# Create a mapping file to document which images were selected for which categories
selection_mapping_file = os.path.join(dst_folder, f'thing_classes_selection_{split_data}.json')
selection_mapping = {
    'total_images': len(selected_images),
    'selection_criteria': '5 images per thing class',
    'categories': {}
}

# Add category information with selected images
for category in categories:
    category_id = category['id']
    if category_id in selected_by_category:
        selection_mapping['categories'][category['name']] = {
            'id': category_id,
            'selected_images': selected_by_category[category_id]
        }

with open(selection_mapping_file, 'w') as f:
    json.dump(selection_mapping, f, indent=2)

# --- Create filtered instance JSON for the mini dataset ---

# Filter images based on the selected image IDs
mini_images = []
for image in instance_data.get("images", []):
    # Remove the extension to get the base id
    file_name = image.get("file_name", "")
    base_id, _ = os.path.splitext(file_name)
    if base_id in selected_images:
        mini_images.append(image)

# Filter annotations based on the selected image IDs
mini_annotations = []
for ann in instance_data.get("annotations", []):
    image_id = ann.get("image_id")
    if image_id in [img['id'] for img in mini_images]:
        mini_annotations.append(ann)

# Assemble the mini instance JSON
mini_instance_data = {
    "images": mini_images,
    "annotations": mini_annotations,
    "categories": instance_data.get("categories", [])
}

# Save the new instance JSON to the mini dataset folder
base_instance_json = os.path.basename(instance_json)
dst_instance_json = os.path.join(dst_folder, base_instance_json)
with open(dst_instance_json, 'w') as f:
    json.dump(mini_instance_data, f, indent=2)

print("Mini dataset and instance JSON created successfully!")
print(f"Mini dataset folder: {dst_folder}")
print(f"Mini instance JSON: {dst_instance_json}")
print(f"Selection mapping: {selection_mapping_file}")
