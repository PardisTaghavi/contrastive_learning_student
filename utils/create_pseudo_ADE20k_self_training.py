import os,sys
sys.path.append('/home/avalocal/thesis23/KD')
import json
import cv2
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from shapely.geometry import Polygon
from model import InstanceTeacher
from torch.utils.data import Dataset, DataLoader

###################################################
#######################ADE20K######################
###################################################

# ADE20K class names
class_name = ('bed', 'windowpane', 'cabinet', 'person', 'door', 'table', 'curtain',
         'chair', 'car', 'painting', 'sofa', 'shelf', 'mirror', 'armchair',
         'seat', 'fence', 'desk', 'wardrobe', 'lamp', 'bathtub', 'railing',
         'cushion', 'box', 'column', 'signboard', 'chest of drawers',
         'counter', 'sink', 'fireplace', 'refrigerator', 'stairs', 'case',
         'pool table', 'pillow', 'screen door', 'bookcase', 'coffee table',
         'toilet', 'flower', 'book', 'bench', 'countertop', 'stove', 'palm',
         'kitchen island', 'computer', 'swivel chair', 'boat',
         'arcade machine', 'bus', 'towel', 'light', 'truck', 'chandelier',
         'awning', 'streetlight', 'booth', 'television receiver', 'airplane',
         'apparel', 'pole', 'bannister', 'ottoman', 'bottle', 'van', 'ship',
         'fountain', 'washer', 'plaything', 'stool', 'barrel', 'basket', 'bag',
         'minibike', 'oven', 'ball', 'food', 'step', 'trade name', 'microwave',
         'pot', 'animal', 'bicycle', 'dishwasher', 'screen', 'sculpture',
         'hood', 'sconce', 'vase', 'traffic light', 'tray', 'ashcan', 'fan',
         'plate', 'monitor', 'bulletin board', 'radiator', 'glass', 'clock',
         'flag')

# Create mapping dictionaries
num_to_name = {i: class_name[i] for i in range(len(class_name))}
name_to_num = {class_name[i]: i for i in range(len(class_name))}

class ADE_loader(Dataset):
    def __init__(self, data_dir, split='training'):
        """
        Dataset loader for ADE20K instance segmentation
        Args:
            data_dir: root directory of ADE20K dataset
            split: 'training' or 'validation'
        """
        self.data_dir = data_dir
        self.split = split
        self.image_dir = os.path.join(data_dir, 'images', split)
        self.json_dir = os.path.join(data_dir, 'mmGDINO', split)
        
        # Get all image filenames
        self.image_files = [f for f in os.listdir(self.image_dir) if f.endswith('.jpg') or f.endswith('.png')]
        self.json_files = [os.path.join(self.json_dir, os.path.splitext(f)[0] + '.json') for f in self.image_files]
        
        print(f"Found {len(self.image_files)} images in {self.image_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        json_path = self.json_files[idx]
        
        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1)  # HWC to CHW
        
        # Load json annotations
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Extract bounding boxes, class ids and scores
        boxes = []
        class_ids = []
        scores = []
        
        for annotation in data['annotations']:
            box = annotation['bbox']
            class_name_str = annotation['class_name']
            score = annotation['score']
            
            if class_name_str in name_to_num:
                class_id = name_to_num[class_name_str]
                boxes.append(box)
                class_ids.append(class_id)
                scores.append(score)
        
        # Convert to tensors
        max_instances = 100  # Maximum number of instances to consider
        
        # Pad or truncate to max_instances
        num_objects = min(len(boxes), max_instances)
        
        # Create padded tensors
        pseudo_bbox = torch.zeros((max_instances, 4))
        pseudo_ids = torch.zeros(max_instances)
        pseudo_scores = torch.zeros(max_instances)
        
        if boxes:
            boxes_tensor = torch.tensor(boxes[:num_objects])
            class_ids_tensor = torch.tensor(class_ids[:num_objects])
            scores_tensor = torch.tensor(scores[:num_objects])
            
            pseudo_bbox[:num_objects] = boxes_tensor
            pseudo_ids[:num_objects] = class_ids_tensor
            pseudo_scores[:num_objects] = scores_tensor
        
        return image_tensor, pseudo_ids, pseudo_bbox, pseudo_scores, num_objects, self.image_files[idx]

def json2instanceImg(data, dst_path, mode='instance'):
    """
    Convert json data to ADE20K instance annotation image.
    ADE20K annotation format: 
    - Instance segmentation: (id+1, instance_id, 0)
    - id is class id (1-indexed)
    - instance_id is the instance number for that class (1, 2, 3...)

    Args:
        data: Dictionary with object information
        dst_path: Path to save the instance annotation image
        mode: 'instance' for instance segmentation annotations
    """
    height = data["imgHeight"]
    width = data["imgWidth"]
    
    # Create a black image for the annotation
    instance_img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Track instance counts per class
    instance_counts = {}
    
    for obj in data["objects"]:
        # Get class name and ID
        label = obj["label"]
        if label in name_to_num:
            class_id = name_to_num[label]
            
            # Initialize instance count for this class if needed
            if class_id not in instance_counts:
                instance_counts[class_id] = 0
            
            # Increment instance count for this class
            instance_counts[class_id] += 1
            instance_id = instance_counts[class_id]
            
            # Create mask from polygon
            polygon = np.array(obj["polygon"])
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(mask, [polygon], 1)
            
            # Set pixel values according to ADE20K format: (id+1, instance_id, 0)
            instance_pixels = np.where(mask == 1)
            instance_img[instance_pixels[0], instance_pixels[1], 0] = class_id + 1  # Class ID (1-indexed)
            instance_img[instance_pixels[0], instance_pixels[1], 1] = instance_id    # Instance ID
            # instance_img[instance_pixels[0], instance_pixels[1], 2] = 0            # Already 0
    
    # Save the instance image in RGB format (cv2 uses BGR by default)
    # Convert to PIL Image to ensure RGB format is preserved
    pil_img = Image.fromarray(instance_img)
    pil_img.save(dst_path)

def masks_to_instance_format(masks, class_ids, dst_path):
    """
    Convert binary masks to ADE20K instance annotation image.
    ADE20K annotation format: 
    - (id+1, instance_id, 0) for each instance pixel
    - id is class id (1-indexed)
    - instance_id is the instance number for that class (1, 2, 3...)
    
    IMPORTANT: Ensures no overlapping instances to prevent multiple class IDs per instance
    """
    # Check if masks is empty
    if masks is None or (isinstance(masks, torch.Tensor) and masks.numel() == 0) or \
       (isinstance(masks, np.ndarray) and masks.size == 0):
        # Create a small black image if no dimensions are given
        H, W = 1024, 2048  # Default size
        instance_img = np.zeros((H, W, 3), dtype=np.uint8)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        # Save black image
        pil_img = Image.fromarray(instance_img)
        pil_img.save(dst_path)
        return instance_img
    
    # Get dimensions
    if masks.ndim == 3:  # [N, H, W]
        N, H, W = masks.shape
    elif masks.ndim == 4:  # [N, 1, H, W]
        N, _, H, W = masks.shape
        masks = masks.squeeze(1)
    
    # Convert tensors to numpy
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy()
    if isinstance(class_ids, torch.Tensor):
        class_ids = class_ids.cpu().numpy()
    
    # Create a black image for the annotation
    instance_img = np.zeros((H, W, 3), dtype=np.uint8)
    
    # Create an occupancy map to prevent overlapping instances
    occupied = np.zeros((H, W), dtype=bool)
    
    # Process masks in order of their confidence or size (if available)
    # You can change the order here if you have confidence scores
    mask_order = list(range(N))
    
    # Track instance counts per class
    instance_counts = {}
    
    for idx in mask_order:
        mask = masks[idx]
        class_id = int(class_ids[idx])
        
        # Skip if mask is empty or class_id is invalid
        if mask.sum() == 0 or class_id >= len(class_name):
            continue
        
        # Convert to binary mask
        binary_mask = (mask > 0.5).astype(np.uint8)
        
        # Remove already occupied pixels from this mask to prevent overlaps
        # This is the critical step to ensure each pixel only belongs to one instance
        binary_mask[occupied] = 0
        
        # Skip if the mask becomes empty after removing occupied pixels
        if binary_mask.sum() == 0:
            continue
        
        # Initialize instance count for this class if needed
        if class_id not in instance_counts:
            instance_counts[class_id] = 0
        
        # Increment instance count for this class
        instance_counts[class_id] += 1
        instance_id = instance_counts[class_id]
        
        # Set pixel values according to ADE20K format: (id+1, instance_id, 0)
        instance_pixels = np.where(binary_mask == 1)
        instance_img[instance_pixels[0], instance_pixels[1], 0] = class_id + 1  # Class ID (1-indexed)
        instance_img[instance_pixels[0], instance_pixels[1], 1] = instance_id    # Instance ID
        # instance_img[instance_pixels[0], instance_pixels[1], 2] = 0            # Already 0
        
        # Update occupied map
        occupied[instance_pixels[0], instance_pixels[1]] = True
    
    # Save the instance image in RGB format
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    # Convert to PIL Image to ensure RGB format is preserved
    pil_img = Image.fromarray(instance_img)
    pil_img.save(dst_path)
    
    # Verify the instance image to make sure it's valid
    # This will help debug any issues with the generated annotations
    verify_instance_image(instance_img, dst_path)
    
    return instance_img

def verify_instance_image(instance_img, dst_path):
    """
    Verify that the instance image follows the ADE20K format:
    - Each instance (unique instance_id > 0) must have exactly one category ID
    """
    # Get dimensions
    H, W = instance_img.shape[:2]
    
    # Get instance IDs (green channel)
    instance_ids = instance_img[:, :, 1]
    
    # Get category IDs (red channel)
    category_ids = instance_img[:, :, 0]
    
    # Check each unique instance ID
    for instance_id in np.unique(instance_ids):
        if instance_id == 0:  # Skip background
            continue
            
        # Get mask for this instance
        mask = instance_ids == instance_id
        
        # Get unique category IDs for this instance
        unique_categories = np.unique(category_ids[mask])
        
        # Check if there's exactly one category ID
        if len(unique_categories) != 1:
            # If there's a problem, fix it by keeping only the majority category
            if len(unique_categories) > 1:
                print(f"Warning: Instance {instance_id} in {os.path.basename(dst_path)} has multiple categories: {unique_categories}")
                
                # Count occurrences of each category
                category_counts = {}
                for cat in unique_categories:
                    category_counts[cat] = np.sum((category_ids == cat) & mask)
                
                # Find the majority category
                majority_category = max(category_counts, key=category_counts.get)
                
                # Set all pixels of this instance to the majority category
                pixels = np.where(mask)
                instance_img[pixels[0], pixels[1], 0] = majority_category
                
                print(f"Fixed by setting all to category {majority_category}")
                
                # Save the fixed image
                pil_img = Image.fromarray(instance_img)
                pil_img.save(dst_path)

                
def save_pseudo_labels(image, masks, ids, file_name, folder):
    """
    Save pseudo-labels in ADE20K format
    
    Args:
        image: Image tensor
        masks: Binary masks tensor
        ids: Class IDs tensor
        file_name: Base filename
        folder: Base folder path
    """
    # Create destination path for annotation
    dst_path = os.path.join(folder, 'pseudo_annotations_instance', 'training', f"{file_name}.png")
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    
    # Convert masks to instance format and save
    instance_img = masks_to_instance_format(masks, ids, dst_path)
    
    # Convert data to JSON format for debugging (optional)
    height, width = masks.shape[1], masks.shape[2]
    data = {
        "imgHeight": height,
        "imgWidth": width,
        "objects": []
    }
    
    for i in range(masks.shape[0]):
        mask = masks[i].cpu().numpy() if isinstance(masks, torch.Tensor) else masks[i]
        # Convert to binary mask (0 or 1 values)
        mask = (mask > 0.5).astype(np.uint8)
        
        # Find the contours of the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if len(contour) < 4:
                print(f"Skipping small contour with area: {cv2.contourArea(contour)}")
                continue
            
            polygon = Polygon(contour.reshape(-1, 2))
            
            if polygon.is_valid:
                polygon_coords = list(polygon.exterior.coords)
                # Format of polygon coords is int
                polygon_coords = [[int(coord[0]), int(coord[1])] for coord in polygon_coords]
                
                # Skip if class ID is invalid
                if ids[i].item() >= len(class_name):
                    continue
                
                # Create the object data for the polygon
                object_data = {
                    "label": num_to_name[ids[i].item()],
                    "polygon": polygon_coords
                }
                # Add the object data to the list
                data["objects"].append(object_data)
    
    # Save JSON for debugging (optional)
    json_path = os.path.join(folder, 'pseudo_annotations_json', 'training', f"{file_name}.json")
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"Saved pseudo labels for {file_name}")

def validate(teacher_model, val_loader, device):
    """
    Create pseudo-labels using the teacher model
    
    Args:
        teacher_model: InstanceTeacher model
        val_loader: DataLoader for validation data
        device: Device to run the model on
    
    Returns:
        count: Number of processed images
    """
    torch.cuda.empty_cache()
    folder = "/media/avalocal/T7/ADE_U"
    
    teacher_model.eval()
    count = 0
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader)):
            torch.cuda.empty_cache()
            image, pseudo_ids, pseudo_bbox, pseudo_scores, num_pseudo_objects, file_name = batch
            
            # Extract base filename
            file_name = file_name[0].split('.')[0]
            
            # Filter by score threshold
            num_pseudo_objects = (pseudo_scores[0] > 0.3).sum().item()
            boxes_ = pseudo_bbox[0][:num_pseudo_objects]  # N, 4
            class_ids_ = pseudo_ids[0][:num_pseudo_objects].long()  # N
            
            # Convert image for visualization
            tmp = image[0].permute(1, 2, 0).numpy().copy()
            tmp = tmp.astype(np.uint8)
            
            if boxes_.shape[0] != 0:
                # Get masks from teacher model
                masks_, scores_, _ = teacher_model(tmp, boxes_.cpu().numpy())
                masks_ = torch.from_numpy(masks_)  # N, 1, H, W
                masks_ = masks_.squeeze(1)
                scores_ = torch.from_numpy(scores_)
                if scores_.ndim == 2:
                    scores_ = scores_.squeeze(1)
                
                # Save pseudo-labels
                save_pseudo_labels(image, masks_, class_ids_, file_name, folder)
                count += 1
            else:
                # Save a black image if no instances detected
                H, W = tmp.shape[:2]
                black_img = np.zeros((H, W, 3), dtype=np.uint8)
                dst_path = os.path.join(folder, 'pseudo_annotations_instance', 'training', f"{file_name}.png")
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                
                # Save using PIL to ensure RGB format
                pil_img = Image.fromarray(black_img)
                pil_img.save(dst_path)
                
                print(f"No instances found for {file_name}, saved a black mask")
    
    return count

def main():
    data_dir = "/media/avalocal/T7/ADE_U"
    
    # Create datasets
    train_dataset = ADE_loader(data_dir, split='training')
    
    # Load teacher model
    teacher_model = InstanceTeacher()
    for _, param in teacher_model.named_parameters():
        param.requires_grad = False
    
    # Set device
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    teacher_model = teacher_model.to(device)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    print(f'Training samples: {len(train_dataset)}')
    
    # Create pseudo-labels
    count = validate(teacher_model, train_loader, device)
    print(f"Total pseudo labels created: {count}")

if __name__ == '__main__':
    main()