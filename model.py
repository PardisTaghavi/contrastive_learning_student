
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))


path_3 = os.path.join(current_dir, "Mask2Former")
sys.path.append(path_3)

path_4 = os.path.join(current_dir, "sam2")
sys.path.append(path_4)

import torch, torch.nn as nn, torch.nn.functional as F

import json
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger

########################################################mask2former 

# maskFormerPath = '/home/avalocal/thesis23/KD/Mask2Former'
# sys.path.append(maskFormerPath)

import itertools
import os

from collections import OrderedDict
from typing import Any, Dict, List, Set

from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.modeling import  build_backbone, build_sem_seg_head
import detectron2.utils.comm as comm
from detectron2.config import get_cfg



# MaskFormer student model
from mask2former import (
    add_maskformer2_config,

)

def setup():
    """
    Create configs and perform basic setups.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(current_dir, "Mask2Former/configs/cityscapes/instance-segmentation/swin/maskformer2_swin_base_IN21k_384_bs16_90k.yaml")
    # config_file ="/home/test/thesis23/KD/Mask2Former/configs/cityscapes/instance-segmentation/swin/maskformer2_swin_base_IN21k_384_bs16_90k.yaml"
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(config_file)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former")
    # print(cfg)
    return cfg

class Mask2FormerStudent(nn.Module):
    def __init__(self):
        super(Mask2FormerStudent, self).__init__()

        cfg = setup()
        self.backbone = build_backbone(cfg)
        self.sem_seg_head = build_sem_seg_head(cfg, self.backbone.output_shape())
        self.size_divisibility = 32
        self.pretrained = "/home/avalocal/thesis23/KD/Mask2Former/swin_base_patch4_window12_384_22k.pth"
        self.backbone.load_state_dict(torch.load(self.pretrained, map_location="cpu")['model'], strict=False)
        

    def forward(self, x):
        #x: (B, 3, H, W)
        

        features = self.backbone(x)
        outputs = self.sem_seg_head(features)
        
        return outputs


    
    
##################################################################|
##################################################################| instance segmentation
#######################| this is student instance segmentation model

import torch.nn.functional as F
from typing import List

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


from sam2.modeling.backbones.hieradet import Hiera  
from sam2.modeling.position_encoding import PositionEmbeddingSine
from sam2.modeling.backbones.image_encoder import FpnNeck
from sam2.modeling.backbones.image_encoder import ImageEncoder


from sam2.modeling.instance import transformer_decoder
from sam2.modeling.sam2_instance_model import SAM2Instance
from sam2.modeling.sam.mask_decoder import MaskDecoderStudent, MaskDecoderSemantic
from sam2.modeling.sam.transformer import TwoWayTransformer


# get path of the current file
# sam2_checkpoint = os.path.join(current_dir, "sam2/checkpoints/sam2.1_hiera_tiny.pt")


class InstanceStudent(nn.Module):
    def __init__(self, sz):

        super(InstanceStudent, self).__init__()

        self.sz = sz
        if sz == "base_plus":

            sam2_checkpoint = os.path.join(current_dir, "sam2/checkpoints/sam2.1_hiera_base_plus.pt")
            # sam2_checkpoint="/home/test/Downloads/sam2.1_hiera_base_plus.pt"

            #based on model b+
            self.image_encoder = ImageEncoder(trunk=Hiera(
                drop_path_rate = 0.2,
            ), neck=FpnNeck(
                position_encoding=PositionEmbeddingSine(num_pos_feats=256),
                d_model=256,
                backbone_channel_list=[896, 448, 224, 112], #based on model b+
                kernel_size=1,
                stride=1,
                padding=0,
                fpn_interp_model="nearest",
                fuse_type="sum",
                fpn_top_down_levels=[2,3], #None,
            ), scalp=1)
            mask_decoder = MaskDecoderStudent(transformer_dim=256, transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
            ), activation=nn.GELU)

        # elif sz == "tiny":
        #     #based on model tiny

        #     sam2_checkpoint = os.path.join(current_dir, "sam2/checkpoints/sam2.1_hiera_tiny.pt")

        #     self.image_encoder = ImageEncoder(trunk=Hiera( 
        #         embed_dim = 96,
        #         num_heads = 1,
        #         stages = [1, 2, 7, 2],
        #         global_att_blocks = [5, 7, 9],
        #         window_pos_embed_bkg_spatial_size = [7, 7],
        #         drop_path_rate = 0.1,
        #     ), neck=FpnNeck(
        #         position_encoding=PositionEmbeddingSine(num_pos_feats=256),
        #         d_model=256,
        #         backbone_channel_list=[768, 384, 192, 96], #based on config of model tiny
        #         kernel_size=1,
        #         stride=1,
        #         padding=0,
        #         fpn_interp_model="bilinear",
        #         fuse_type="sum",
        #         fpn_top_down_levels=[2,3], #None,
        #     ), scalp=1)
        #     mask_decoder = MaskDecoderStudentTWO(transformer_dim=256, transformer=TwoWayTransformer(
        #         depth=2,
        #         embedding_dim=256,
        #         mlp_dim=2048,
        #         num_heads=8,
        #     ), activation=nn.GELU)

        self.model = SAM2Instance(image_encoder=self.image_encoder, mask_decoder=mask_decoder)

        sam2_checkpoint = torch.load(sam2_checkpoint, map_location="cpu", weights_only=True)
        state_dict = sam2_checkpoint['model']
        filtered_state_dict = {k: v for k, v in state_dict.items() if "image_encoder" in k}
        self.model.load_state_dict(filtered_state_dict, strict=False) # load the model weights for image_encoder
        self.model.to(DEVICE)
        #for the rest, we will initialize the weights from scratch

        for name, param in self.model.named_parameters():
            if "image_encoder" not in name:
                if 'weight' in name:
                    if param.dim() >= 2:  # Ensure the parameter has at least 2 dimensions
                        nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
                elif 'bias' in name:
                    nn.init.constant_(param, 0)

    def forward(self, x):
        '''
        input: x (B, 3, H, W)
        '''
        output = self.model(x)
        # print(output.keys(), "this is output of instance student")
        return output
    




class SemanticStudent(nn.Module):
    def __init__(self):

        super(SemanticStudent, self).__init__()

        sam2_checkpoint = os.path.join(current_dir, "sam2/checkpoints/sam2.1_hiera_base_plus.pt")

        #based on model b+
        self.image_encoder = ImageEncoder(trunk=Hiera(
            drop_path_rate = 0.2,
        ), neck=FpnNeck(
            position_encoding=PositionEmbeddingSine(num_pos_feats=256),
            d_model=256,
            backbone_channel_list=[896, 448, 224, 112], #based on model b+
            kernel_size=1,
            stride=1,
            padding=0,
            fpn_interp_model="nearest",
            fuse_type="sum",
            fpn_top_down_levels=[2,3], #None,
        ), scalp=1)
        
        mask_decoder = MaskDecoderSemantic(transformer_dim=256,
         transformer=TwoWayTransformer(
            depth=2,
            embedding_dim=256,
            mlp_dim=2048,
            num_heads=8,
        ), activation=nn.GELU)

        self.model = SAM2Instance(image_encoder=self.image_encoder, mask_decoder=mask_decoder)

        sam2_checkpoint = torch.load(sam2_checkpoint, map_location="cpu", weights_only=True)
        state_dict = sam2_checkpoint['model']

        filtered_state_dict = {k: v for k, v in state_dict.items() if "image_encoder" in k}
        self.model.load_state_dict(filtered_state_dict, strict=False) # load the model weights for image_encoder
        self.model.to(DEVICE)

        # self.model.load_state_dict(state_dict, strict=False) # load the model weights for image_encoder
        # self.model.to(DEVICE)

        #for the rest, we will initialize the weights from scratch

        for name, param in self.model.named_parameters():
            if "image_encoder" not in name:
                if 'weight' in name:
                    if param.dim() >= 2:  # Ensure the parameter has at least 2 dimensions
                        nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
                elif 'bias' in name:
                    nn.init.constant_(param, 0)

    def forward(self, x):
        '''
        input: x (B, 3, H, W)
        '''
        output = self.model(x)
        # print(output.keys(), "this is output of instance student")
        return output



####python trainInstance.py --batch_size 4 ################################################ teachers for instance segmentation

# sam2_checkpoint = "/home/avalocal/thesis23/KD/segment-anything-2/checkpoints/sam2_hiera_large.pt"
# model_cfg = "sam2_hiera_l.yaml"

# sam2_checkpoint="/home/avalocal/thesis23/KD/sam2/checkpoints/sam2.1_hiera_large.pt"


class InstanceTeacher(nn.Module):   #used for teacher model of depth prediction
    def __init__(self):
        super(InstanceTeacher, self).__init__()

        sam2_checkpoint = os.path.join(current_dir, "sam2/checkpoints/sam2.1_hiera_large.pt")

        #sam2_checkpoint = "/home/avalocal/thesis23/KD/sam2/checkpoints/sam2.1_hiera_large_finetuned_instance_1e-5.pth"
        #sam2_checkpoint =  "/home/avalocal/thesis23/KD/sam2/checkpoints/sam2.1_hiera_large_finetuned_instance_5e-5.pth"
        # sam2_checkpoint="/home/avalocal/thesis23/KD/sam2/checkpoints/sam2.1_hiera_large_finetuned_instance.pth"

        model_cfg= "sam2.1_hiera_l.yaml"
        self.sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)

    def forward(self, x, bbox):
        '''
        input: x | should be np.array with size (h,w,3)
        bbox: array w size (n, 4) where n is the number of bounding boxes
        '''        
        self.sam2_predictor.set_image(x)
        masks, scores, logits = self.sam2_predictor.predict(
                            point_coords=None,
                            point_labels=None,
                            box=bbox,
                            multimask_output=False,
                        )
        return masks, scores, logits




if __name__=="__main__":
    
    # x = torch.randn(4, 3, 448, 448)
    # student_model = DPTDepthPredictor()
    # feats, out = student_model(x)

    model = SemanticStudent()
    model = nn.DataParallel(model)
    model = model.cuda()

    x = torch.randn(4, 3, 1024, 1024).cuda()
    output = model(x)
    print(output.keys())  
    print(output['pred_masks'].shape)  
        
