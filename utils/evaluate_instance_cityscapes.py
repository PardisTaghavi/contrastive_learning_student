
import argparse, os , time, wandb, cv2
from tqdm import tqdm
import torch
import torchvision.transforms.functional as TF
import random
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_warmup as warmup
from model import InstanceStudent, InstanceTeacher
from cityscapes_original import CityscapesSV, CityscapesPseudo
import albumentations as A
from detectron2.evaluation import COCOEvaluator
from detectron2.structures import Instances, Boxes
from detectron2.data import MetadataCatalog
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import supervision as sv
from pycocotools import mask as mask_util
import json
from detectron2.structures import BitMasks
from pycocotools import mask as coco_mask
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from utils import LargeScaleJittering, instance_inference
import numpy as np
from dinoV2_dpt_models.semseg.dptInstance import DPT

torch.manual_seed(42)

#metrics for instance segmentation
metric_name = ['ap', 'aps', 'apm', 'apl', 'ap_boundry']
name_to_num={'person': 0, 'rider': 1, 'car': 2, 'truck': 3, 'bus': 4, 'train': 5, 'motorcycle': 6, 'bicycle': 7, '': 8}
num_to_name = {0: 'person', 1: 'rider', 2: 'car', 3: 'truck', 4: 'bus', 5: 'train', 6: 'motorcycle', 7: 'bicycle', 8: ''}


def parse_args():
    parser = argparse.ArgumentParser(description='Instance Seg Training')
    parser.add_argument('--epochs', type=int, default=400, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size for training')
    parser.add_argument('--lr', type=float,  default=1e-4, help='learning rate')
    parser.add_argument('--data_dir', type=str, default='/media/avalocal/T7/pardis/pardis/perception_system/datasets/cityscapes', help='path to cityscapes dataset')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='path to save checkpoints')
    parser.add_argument('--type', type=str, help='define relative vs metric for depth estimation')
    parser.add_argument('--experiment', type=int, help='define the experiment number')
    parser.add_argument('--student', type=str, help='[DinoDepthPredictor, DPTDepthPredictor, DPT]')
    parser.add_argument('--dataset_size', type=int, default=-1, help='number of samples to use for training')
    parser.add_argument('--results_dir', type=str, default='./results', help='path to save results')
    parser.add_argument('--norm', type=str, default='minmaxglobal', help='normalization method')
    parser.add_argument('--warmup_epochs', type=int, default=3, help='number of warmup epochs')
    parser.add_argument('--resume', type=str, default=None, help='path to checkpoint to resume training')
    return parser.parse_args()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


from detectron2.evaluation import COCOEvaluator
from detectron2.structures import Instances, Boxes
from detectron2.data import MetadataCatalog
from pycocotools.coco import COCO
import torch

#this validate results of the model on 500 val dataset of cityscapes
def validate(model, teacher_model, val_loader, device, pseudo_label_evaluation, pred_evaluation):

    #cityscapes_original import CityscapesSV


    torch.cuda.empty_cache()
    mean = torch.tensor([[[123.6750]], [[116.2800]], [[103.5300]]]).to(device)
    std = torch.tensor([[[58.3950]], [[57.1200]], [[57.3750]]]).to(device)

    # metadata = MetadataCatalog.get("cityscapes_fine_instance_seg_val" )#cityscapes")
    # # metadata.set(thing_classes=["person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"])
    # print("Metadata Categories:", metadata.thing_classes)

    metric = MeanAveragePrecision(iou_type="segm", average='macro')

    if pred_evaluation:
        model.eval()



    elif pseudo_label_evaluation:
        teacher_model.eval()


    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader)):


            if pseudo_label_evaluation:
                teacher_model.eval()
                image, masks, ids, _, num_objects, pseudo_ids, pseudo_bbox, pseudo_scores, num_pseudo_objects,file_name = batch
                # print(f"image shape:{image.shape}, masks shape:{masks.shape}, ids shape:{ids.shape}, num_objects:{num_objects}")
                
                num_pseudo_objects = torch.tensor(100)

                
                num_pseudo_objects = (pseudo_scores>0.30).sum()
                print(f"num_of_real_objects:{num_objects}")
                print(f"num_pseudo_objects:{num_pseudo_objects}")
                print(f"pseudo_scores:{pseudo_scores}")
                #number of pseudo scores>0.3

            elif pred_evaluation:
                image, instance, ids,  bbox, num_objects, file_name = batch
                # print(f"image shape:{image.shape}, instance shape:{instance.shape}, ids shape:{ids.shape}, bbox shape:{bbox.shape}, num_objects:{num_objects}")
                gt_mask = instance[0][:num_objects] #N, 1024, 2048
                gt_ids = ids[0][:num_objects]


            image*=255.0

            if pseudo_label_evaluation:

                boxes_ = pseudo_bbox[0][:num_pseudo_objects] #N, 4
                class_ids_ = pseudo_ids[0][:num_pseudo_objects] #N

                tmp = image[0].permute(1, 2, 0).numpy().copy()#.astype(np.uint8) #H, W, 3
                tmp = tmp.astype(np.uint8)

                if boxes_.shape[0] != 0:
                    with torch.cuda.amp.autocast():
                        masks_, scores_, _ = teacher_model(tmp, boxes_.numpy())
                    masks_ = torch.from_numpy(masks_)#.to(device) #N, 1, H, W
                    masks_ = masks_.squeeze(1)
                    scores_ = torch.from_numpy(scores_)#.to(device) #N
                    #if shape:torch.Size([0]) score values:tensor([]) not squeeze
                    if scores_.ndim == 2:
                        scores_ = scores_.squeeze(1)
                    # print(f"masks_ shape:{masks_.shape}") #masks_ shape:torch.Size([8, 1024, 2048])
                else:
                    continue

                if class_ids_.shape[0] == 0: continue


                pred = {
                "masks": masks_.bool(), #N, H, W
                "labels": class_ids_.to(torch.int64), #N
                "scores": scores_, #N
                }

                target = {
                    "labels": ids[0][:num_objects].to(torch.int64), #N
                    "masks": masks[0][:num_objects].bool().to(device), #N, 1024, 2048
                }

                metric.update([pred], [target])

            elif pred_evaluation:

                model.eval()
                image = TF.resize(image, (798, 798),  interpolation=TF.InterpolationMode.NEAREST) #1, 3, 512, 1024
                image = image.to(device)
                image = (image - mean) / std

                pred_out = model(image)
                pred_out['pred_masks'] = TF.resize(pred_out['pred_masks'], (1024, 2048),  interpolation=TF.InterpolationMode.NEAREST)

                mask_pred_results = pred_out['pred_masks'] #1, 100, 1024, 2048
                mask_cls_results = pred_out['pred_logits'] #1, 100, 9

                del pred_out #just for memory

                processed_results = []
                for mask_cls_result, mask_pred_result in zip(
                    mask_cls_results, mask_pred_results
                ):
                    instance_result = instance_inference(mask_cls_result, mask_pred_result)
                    processed_results.append(
                        {
                            "classes": instance_result.pred_classes, #N
                            "masks": instance_result.pred_masks, #N, 1024, 2048
                            "scores": instance_result.scores, #N
                        }
                    )

                instance_result = instance_inference(mask_cls_results[0], mask_pred_results[0])
                pred = {
                    "masks": processed_results[0]["masks"].bool(),    #N, 1024, 2048
                    "labels": processed_results[0]["classes"].to(torch.int64), #N
                    "scores": processed_results[0]["scores"], #N
                }
                target = {
                    "labels": gt_ids.to(torch.int64).to(device), #N
                    "masks": gt_mask.bool().to(device), #N, 1024, 2048
                }
                metric.update([pred], [target])      
    results = metric.compute()
    print(f'Validation Results: {results}')

    return results



def main():

    data_dir = "/media/avalocal/T7/pardis/pardis/perception_system/datasets/cityscapes"
    pseudo_label_evaluation = True
    pred_evaluation= False

    if pseudo_label_evaluation==True:
        val_dataset   = CityscapesPseudo(data_dir, split='val')
        teacher_model = InstanceTeacher()
        for name, param in teacher_model.named_parameters():
            param.requires_grad = False
        teacher_model = teacher_model.to(DEVICE)
        model = None

    elif pred_evaluation==True:
        val_dataset   = CityscapesSV(data_dir, split='val')
        teacher_model = None
        # ckpt = "/home/avalocal/thesis23/KD/sam2/checkpoints/student_instance_KD_20241201-141638.pth"
        #ckpt = "/home/avalocal/Downloads/checkpoint_epoch_150.pth"
        ckpt = "/home/avalocal/thesis23/KD/checkpoints/20250112-152910/checkpoint_epoch_150.pth" #this is best KD step I have by now
        #ckpt ="/home/avalocal/thesis23/KD/checkpoints/20250117-232827/checkpoint_epoch_150.pth" #this is best KD step I have by now - 250 epochs
        # ckpt = "/home/avalocal/thesis23/KD/checkpoints/20250113-165207/checkpoint_epoch_50.pth"  # this for best few-shot I have by now
        #ema.pth"
        checkpoint = torch.load(ckpt, map_location=DEVICE)
        checkpoint['model_state_dict'] = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()}
        #add model to the checkpoint
        # checkpoint['model_state_dict'] = 

        # model =InstanceStudent("base_plus")
        model = DPT(
            encoder_size='base', 
            nclass= 8, #8
            features=128,   # 128
            out_channels=[96, 192, 384, 768], 
            use_bn=True,
            dec_layers_num=9,
        )
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        model = model.to(DEVICE)
        model.eval()

    val_loader  = DataLoader(val_dataset, batch_size=1, shuffle=False)
    print(f'Validation samples: {len(val_dataset)}')
    assert len(val_dataset) == 500, "original Validation dataset of cityscapes should have 500 samples"
    results_ = validate(model, teacher_model, val_loader, DEVICE, pseudo_label_evaluation, pred_evaluation)
    print(f'Validation Results: {results_}')


if __name__ == '__main__':

    main()


