#author : Pardis Taghavi
#date : December 2024

import os
import json
import cv2
import nltk
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from mmdet.apis import DetInferencer
from torch.serialization import add_safe_globals
from mmengine.logging.history_buffer import HistoryBuffer
import numpy.core.multiarray as multiarray

# Add safe globals for torch serialization
add_safe_globals([HistoryBuffer, multiarray._reconstruct])

'''
#for running evaluation run:
python tools/test.py configs/mm_grounding_dino/ade20k/grounding_dino_swin_finetune_ade20k.py /home/avalocal/Mask2Former/mmdetection/zero_shot/grounding_dino_swin-l_pretrain_all-56d69e78.pth
'''

'''
#for fine-tuning run

'''




#####################################################
#####################################################
################### ADE20K ##########################
#####################################################
#####################################################

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

palette = [(204, 5, 255), (230, 230, 230), (224, 5, 255),
                    (150, 5, 61), (8, 255, 51), (255, 6, 82), (255, 51, 7),
                    (204, 70, 3), (0, 102, 200), (255, 6, 51), (11, 102, 255),
                    (255, 7, 71), (220, 220, 220), (8, 255, 214),
                    (7, 255, 224), (255, 184, 6), (10, 255, 71), (7, 255, 255),
                    (224, 255, 8), (102, 8, 255), (255, 61, 6), (255, 194, 7),
                    (0, 255, 20), (255, 8, 41), (255, 5, 153), (6, 51, 255),
                    (235, 12, 255), (0, 163, 255), (250, 10, 15), (20, 255, 0),
                    (255, 224, 0), (0, 0, 255), (255, 71, 0), (0, 235, 255),
                    (0, 173, 255), (0, 255, 245), (0, 255, 112), (0, 255, 133),
                    (255, 0, 0), (255, 163, 0), (194, 255, 0), (0, 143, 255),
                    (51, 255, 0), (0, 82, 255), (0, 255, 41), (0, 255, 173),
                    (10, 0, 255), (173, 255, 0), (255, 92, 0), (255, 0, 245),
                    (255, 0, 102), (255, 173, 0), (255, 0, 20), (0, 31, 255),
                    (0, 255, 61), (0, 71, 255), (255, 0, 204), (0, 255, 194),
                    (0, 255, 82), (0, 112, 255), (51, 0, 255), (0, 122, 255),
                    (255, 153, 0), (0, 255, 10), (163, 255, 0), (255, 235, 0),
                    (8, 184, 170), (184, 0, 255), (255, 0, 31), (0, 214, 255),
                    (255, 0, 112), (92, 255, 0), (70, 184, 160), (163, 0, 255),
                    (71, 255, 0), (255, 0, 163), (255, 204, 0), (255, 0, 143),
                    (133, 255, 0), (255, 0, 235), (245, 0, 255), (255, 0, 122),
                    (255, 245, 0), (214, 255, 0), (0, 204, 255), (255, 255, 0),
                    (0, 153, 255), (0, 41, 255), (0, 255, 204), (41, 0, 255),
                    (41, 255, 0), (173, 0, 255), (0, 245, 255), (0, 255, 184),
                    (0, 92, 255), (184, 255, 0), (255, 214, 0), (25, 194, 194),
                    (102, 255, 0), (92, 0, 255)]



# Create mapping dictionaries
num_to_name = {i: class_name[i] for i in range(len(class_name))}
name_to_num = {class_name[i]: i for i in range(len(class_name))}

# Print the maximum valid class index
print(f"Number of classes in dictionary: {len(class_name)}")
print(f"Maximum class index: {max(num_to_name.keys())}")

def convert_to_json_format(preds, img_path, img_width=2048, img_height=1024):
    """
    Convert prediction results to JSON format
    
    Args:
        preds: Prediction results from inferencer
        img_path: Path to the input image
        img_width: Width of the image
        img_height: Height of the image
        
    Returns:
        Dictionary in the specified JSON format
    """
    input_boxes = preds['bboxes']
    scores = preds['scores']
    labels = preds['labels']
    
    # Filter out predictions with labels not in our dictionary
    valid_indices = []
    class_names = []
    
    for i, label in enumerate(labels):
        if label in num_to_name:
            valid_indices.append(i)
            class_names.append(num_to_name[label])
        else:
            print(f"Warning: Skipping unknown class label: {label}")
    
    # Filter boxes and scores to match valid class names
    filtered_boxes = [input_boxes[i] for i in valid_indices]
    filtered_scores = [scores[i] for i in valid_indices]
    
    # Ensure all arrays have the same length
    assert len(filtered_boxes) == len(filtered_scores) == len(class_names), "Mismatch in prediction lengths"
    
    results = {
        "image_path": img_path,
        "annotations": [
            {
                "class_name": class_label,
                "bbox": box.tolist() if isinstance(box, np.ndarray) else box,
                "score": float(score) if isinstance(score, (np.ndarray, torch.Tensor)) else score,
            }
            for class_label, box, score in zip(class_names, filtered_boxes, filtered_scores)
        ],
        "box_format": "xyxy",
        "img_width": img_width,
        "img_height": img_height
    }
    return results

def main():
    # Download NLTK data
    nltk_data_dir = os.path.expanduser('~/nltk_data')
    os.makedirs(nltk_data_dir, exist_ok=True)
    nltk.download('punkt', download_dir=nltk_data_dir)
    nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_dir)
    
    # Define paths - consider using argparse for these in a real application
    dataset_folder = '/media/avalocal/T7/ADE/ADE_U'
    saved_folder = '/media/avalocal/T7/ADE/ADE_U/mmGDINO'
    checkpoint = '/home/avalocal/Mask2Former/mmdetection/zero_shot/grounding_dino_swin-l_pretrain_all-56d69e78.pth'
    config_path = '/home/avalocal/Mask2Former/mmdetection/configs/mm_grounding_dino/ade20k/grounding_dino_swin_finetune_ade20k.py'
    
    # Initialize the inferencer
    inferencer = DetInferencer(model=config_path, weights=checkpoint)
    
    # Select the dataset split
    split = 'training'  # 'training' or 'validation'
    
    # Ensure output directories exist
    os.makedirs(os.path.join(saved_folder, split), exist_ok=True)
    
    # Create the text prompt for all classes (joined with periods)
    text_prompt = ' . '.join(class_name)
    
    # Process all images in the selected split
    images_list = os.listdir(os.path.join(dataset_folder, 'images', split))
    
    for img in tqdm(images_list):
        img_path = os.path.join(dataset_folder, 'images', split, img)
        
        # Run inference
        results = inferencer(
            inputs=img_path,
            texts=text_prompt,
            return_vis=True,
            pred_score_thr=0.3
        )
        
        out = results
        vis = out['visualization']
        preds = out['predictions'][0]
        
        # Save visualization
        vis_path = os.path.join(saved_folder, split, img)
        print(f"Visualization shape: {vis[0].shape}")  # h,w,c
        
        # Convert BGR to RGB for visualization
        vis_rgb = cv2.cvtColor(vis[0], cv2.COLOR_BGR2RGB)
        cv2.imwrite(vis_path, vis_rgb)
        
        # Save JSON results
        json_results = convert_to_json_format(preds, img_path)
        json_path = os.path.join(saved_folder, split, f"{os.path.splitext(img)[0]}.json")
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=4)

if __name__ == "__main__":
    main()