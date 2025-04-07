import os, sys
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import time
from tqdm import tqdm
from PIL import Image
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# Add your project path if needed
sys.path.append('/home/avalocal/Mask2Former')

# Import your models
from model import InstanceTeacher

# Set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

class ADE20KDataset(Dataset):
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
        self.annotation_dir = os.path.join(data_dir, 'annotations', split)
        self.json_dir = os.path.join(data_dir, 'mmGDINO', split)
        
        # Get all image filenames
        self.image_files = [f for f in os.listdir(self.image_dir) if f.endswith('.jpg') or f.endswith('.png')]
        self.annotation_files = [os.path.join(self.annotation_dir, os.path.splitext(f)[0] + '.png') for f in self.image_files]
        self.json_files = [os.path.join(self.json_dir, os.path.splitext(f)[0] + '.json') for f in self.image_files]
        
        print(f"Found {len(self.image_files)} images in {self.image_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        annotation_path = self.annotation_files[idx]
        json_path = self.json_files[idx]
        
        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0  # HWC to CHW and normalize
        
        # Load annotation if available
        if os.path.exists(annotation_path):
            annotation = np.array(Image.open(annotation_path))
            
            # Extract instance masks and classes
            # In ADE20K: R channel (annotation[:,:,0]) is class id+1, G channel (annotation[:,:,1]) is instance id
            instance_masks = []
            instance_classes = []
            
            # Get unique combinations of class and instance IDs
            class_ids = annotation[:,:,0]  # R channel has class ID + 1
            instance_ids = annotation[:,:,1]  # G channel has instance ID
            
            # Find unique instances
            unique_instances = {}
            for y in range(annotation.shape[0]):
                for x in range(annotation.shape[1]):
                    cls_id = class_ids[y, x]
                    inst_id = instance_ids[y, x]
                    
                    if cls_id > 0 and inst_id > 0:  # Ignore background
                        key = (cls_id, inst_id)
                        if key not in unique_instances:
                            unique_instances[key] = []
                        unique_instances[key].append((y, x))
            
            # Create binary masks for each instance
            for (cls_id, inst_id), pixels in unique_instances.items():
                mask = np.zeros((annotation.shape[0], annotation.shape[1]), dtype=np.uint8)
                for y, x in pixels:
                    mask[y, x] = 1
                
                instance_masks.append(mask)
                instance_classes.append(cls_id - 1)  # Subtract 1 to get 0-indexed class ID
            
            # Convert to tensors
            if instance_masks:
                instance_masks = torch.tensor(np.stack(instance_masks), dtype=torch.bool)
                instance_classes = torch.tensor(instance_classes, dtype=torch.long)
            else:
                instance_masks = torch.zeros((0, annotation.shape[0], annotation.shape[1]), dtype=torch.bool)
                instance_classes = torch.zeros(0, dtype=torch.long)
        else:
            # Create empty tensors if annotation not available
            h, w = image.shape[:2]
            instance_masks = torch.zeros((0, h, w), dtype=torch.bool)
            instance_classes = torch.zeros(0, dtype=torch.long)
        
        # Load json annotations (detection results for pseudo-labels)
        if os.path.exists(json_path):
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
        else:
            boxes = []
            class_ids = []
            scores = []
        
        # Convert to tensors
        max_instances = 100  # Maximum number of instances to consider
        
        # Pad or truncate to max_instances
        num_objects = len(instance_masks)
        num_pseudo_objects = min(len(boxes), max_instances)
        
        # Create padded tensors for pseudo labels
        pseudo_bbox = torch.zeros((max_instances, 4))
        pseudo_ids = torch.zeros(max_instances)
        pseudo_scores = torch.zeros(max_instances)
        
        if boxes:
            boxes_tensor = torch.tensor(boxes[:num_pseudo_objects])
            class_ids_tensor = torch.tensor(class_ids[:num_pseudo_objects])
            scores_tensor = torch.tensor(scores[:num_pseudo_objects])
            
            pseudo_bbox[:num_pseudo_objects] = boxes_tensor
            pseudo_ids[:num_pseudo_objects] = class_ids_tensor
            pseudo_scores[:num_pseudo_objects] = scores_tensor
        
        return (
            image_tensor,           # [3, H, W]
            instance_masks,         # [N, H, W] - Ground truth instance masks
            instance_classes,       # [N] - Ground truth class IDs
            pseudo_ids,             # [100] - Pseudo label class IDs (padded)
            pseudo_bbox,            # [100, 4] - Pseudo label bounding boxes (padded)
            pseudo_scores,          # [100] - Pseudo label confidence scores (padded)
            num_objects,            # Scalar - Number of ground truth objects
            num_pseudo_objects,     # Scalar - Number of pseudo objects
            self.image_files[idx]   # Filename
        )

# This function is only used for saving pseudo labels, not needed for evaluation
def masks_to_instance_format(masks, ids, dst_path):
    pass

# These functions are not needed for evaluation only
# Keeping the function signatures empty to maintain compatibility
def save_pseudo_labels(image, masks, ids, file_name, folder):
    pass

def generate_pseudo_labels(teacher_model, data_loader, output_dir):
    pass

def evaluate_pseudo_labels(teacher_model, val_loader, score_threshold=0.3):
    """
    Evaluate pseudo-labels using mIoU
    """
    torch.cuda.empty_cache()
    
    # Set up metrics
    metric = MeanAveragePrecision(iou_type="segm", average='macro')
    
    teacher_model.eval()
    
    total_images = 0
    processed_images = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating pseudo-labels"):
            (image, gt_masks, gt_classes, pseudo_ids, pseudo_bbox, 
             pseudo_scores, num_objects, num_pseudo_objects, file_name) = batch
            
            total_images += 1
            
            # Skip if no ground truth objects
            if num_objects == 0:
                print(f"Skipping {file_name} - No ground truth objects")
                continue
            
            # Filter by score threshold
            valid_indices = pseudo_scores[0] > score_threshold
            num_filtered_objects = valid_indices.sum().item()
            
            if num_filtered_objects == 0:
                print(f"No objects with score > {score_threshold} for {file_name}")
                # Add empty prediction to metric
                pred = {
                    "masks": torch.zeros((0, gt_masks.shape[1], gt_masks.shape[2]), dtype=torch.bool),
                    "labels": torch.zeros(0, dtype=torch.int64),
                    "scores": torch.zeros(0)
                }
                
                target = {
                    "labels": gt_classes.to(torch.int64),
                    "masks": gt_masks.bool()
                }
                
                metric.update([pred], [target])
                continue
                
            boxes_ = pseudo_bbox[0][valid_indices]  # [N, 4]
            class_ids_ = pseudo_ids[0][valid_indices].long()  # [N]
            
            # Convert image for teacher model
            tmp = image[0].permute(1, 2, 0).numpy().copy() * 255
            tmp = tmp.astype(np.uint8)
            
            # Get masks from teacher model
            masks_, scores_, _ = teacher_model(tmp, boxes_.cpu().numpy())
            masks_ = torch.from_numpy(masks_)  # [N, 1, H, W]
            masks_ = masks_.squeeze(1)  # [N, H, W]
            scores_ = torch.from_numpy(scores_)
            if scores_.ndim == 2:
                scores_ = scores_.squeeze(1)
            
            # Add to metric
            pred = {
                "masks": masks_.bool(),
                "labels": class_ids_.to(torch.int64),
                "scores": scores_
            }
            
            target = {
                "labels": gt_classes.to(torch.int64),
                "masks": gt_masks.bool()
            }
            
            metric.update([pred], [target])
            processed_images += 1
            
            # Log progress occasionally
            if processed_images % 10 == 0:
                print(f"Processed {processed_images}/{total_images} images with valid annotations")
    
    # Compute final metrics
    results = metric.compute()
    
    # Print detailed results
    print("\nEvaluation Results:")
    print(f"Processed {processed_images}/{total_images} images")
    print(f"Score threshold: {score_threshold}")
    
    for key, value in results.items():
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            print(f"{key}: {value.item():.4f}")
        else:
            print(f"{key}: {value}")
    
    # Add processing statistics to results
    results['processed_images'] = processed_images
    results['total_images'] = total_images
    results['score_threshold'] = score_threshold
    
    return results

def parse_args():
    import argparse
    
    parser = argparse.ArgumentParser(description='ADE20K Pseudo-Label Evaluation')
    parser.add_argument('--data_dir', type=str, default='/media/avalocal/T7/ADE_U', 
                        help='path to ADE20K dataset')
    parser.add_argument('--output_dir', type=str, default='/media/avalocal/T7/ADE_U/evaluation_results', 
                        help='path to save evaluation results')
    parser.add_argument('--batch_size', type=int, default=1, 
                        help='batch size for evaluation')
    parser.add_argument('--score_threshold', type=float, default=0.3,
                        help='confidence score threshold for pseudo-labels')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Only create validation dataset
    val_dataset = ADE20KDataset(args.data_dir, split='validation')
    
    # Create data loader
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f'Validation samples: {len(val_dataset)}')
    print(f'Using score threshold: {args.score_threshold}')
    
    # Load teacher model
    teacher_model = InstanceTeacher()
    for name, param in teacher_model.named_parameters():
        param.requires_grad = False
    teacher_model = teacher_model.to(DEVICE)
    teacher_model.eval()
    
    # Run evaluation
    print("Evaluating pseudo-labels on validation set...")
    results = evaluate_pseudo_labels(teacher_model, val_loader, score_threshold=args.score_threshold)
    
    # Save results to file
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        results_file = os.path.join(args.output_dir, f"evaluation_results_{timestamp}.json")
        with open(results_file, "w") as f:
            # Convert tensor values to float for JSON serialization
            serializable_results = {}
            for k, v in results.items():
                if isinstance(v, torch.Tensor):
                    if v.numel() == 1:
                        serializable_results[k] = float(v.item())
                    else:
                        serializable_results[k] = [float(x) for x in v.tolist()]
                else:
                    serializable_results[k] = v
            
            json.dump(serializable_results, f, indent=4)
        
        print(f"Evaluation results saved to {results_file}")

if __name__ == '__main__':
    main()