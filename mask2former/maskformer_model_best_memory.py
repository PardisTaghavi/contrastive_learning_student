# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom

from .modeling.criterion import SetCriterion
from .modeling.matcher import HungarianMatcher
import copy
import torchvision.transforms as T
import torch
import torch.nn.functional as F
from collections import deque
import math

def second_augmentation(image):
    
    transform = T.Compose([
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),  # Color jitter
        T.RandomGrayscale(p=0.2),  # Convert some images to grayscale
        T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),  # Apply Gaussian blur
        # T.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),  # Random erasing
    ])
    aug = transform(image.float()/255.0) * 255.0
    return aug
import torch
import torch.nn.functional as F



class MemoryBank:
    """
    Maintains a memory bank of features, instance masks, and class probabilities
    for improved negative sampling in contrastive learning.
    """
    def __init__(self, feature_dim, max_samples=1024, momentum=0.99):
        """
        Initialize the memory bank.
        
        Args:
            feature_dim (int): Dimension of feature embeddings
            max_samples (int): Maximum number of samples to store
            momentum (float): Momentum for updating features if they already exist
        """
        self.feature_dim = feature_dim
        self.max_samples = max_samples
        self.momentum = momentum
        
        # Using deque for efficient FIFO queue management
        self.features = deque(maxlen=max_samples)
        self.probs = deque(maxlen=max_samples)
        self.image_ids = deque(maxlen=max_samples)
        
        self.next_image_id = 0
        
    def update(self, features, probs):
        """
        Update the memory bank with new samples.
        
        Args:
            features (Tensor): Feature embeddings, shape (B, N, D)
            probs (Tensor): Class probabilities, shape (B, N, num_classes)
        """
        B, N, _ = features.shape
        
        # Flatten features and probs for storage
        features_flat = features.reshape(B*N, -1)
        probs_flat = probs.reshape(B*N, -1)
        
        # Generate image IDs for the batch
        batch_image_ids = torch.full((B*N,), -1, device=features.device)
        for b in range(B):
            batch_image_ids[b*N:(b+1)*N] = self.next_image_id + b
        
        # Add to memory bank
        for feat, prob, img_id in zip(features_flat, probs_flat, batch_image_ids):
            self.features.append(feat.detach())
            self.probs.append(prob.detach())
            self.image_ids.append(img_id.item())
        
        self.next_image_id += B
    
    def get_negatives(self, anchor_image_ids, device):
        """
        Get features and probs from the memory bank, excluding those from the same images
        as the anchors.
        
        Args:
            anchor_image_ids (Tensor): Image IDs for the anchors, shape (B,)
            device: Device for the returned tensors
            
        Returns:
            Tuple[Tensor, Tensor]: Negative features and their class probabilities
        """
        if len(self.features) == 0:
            return None, None
        
        # Convert deques to tensors
        features = torch.stack(list(self.features), dim=0).to(device)
        probs = torch.stack(list(self.probs), dim=0).to(device)
        image_ids = torch.tensor(list(self.image_ids), device=device)
        
        # Create mask for items from different images
        mask = torch.ones_like(image_ids, dtype=torch.bool)
        for id in anchor_image_ids:
            mask &= (image_ids != id)
        
        # Return filtered negatives
        return features[mask], probs[mask]


def instance_positive_loss(z_weak, z_strong, pseudo_mask, h, w, current_T):
    """
    Computes a contrastive loss where the positive set for each pixel includes 
    all pixels predicted to belong to the same instance.
    
    Args:
        z_weak (Tensor): Feature embeddings from the weak branch, shape (B, N, D)
        z_strong (Tensor): Feature embeddings from the strong branch, shape (B, N, D)
        pseudo_mask (Tensor): Pseudo-instance masks (logits) with shape (B, num_instances, H, W)
        h, w (int): Height and width of the feature map (N = h*w)
        current_T (float): Current temperature.
        
    Returns:
        Tensor: Scalar loss.
    """
    B, N, _ = z_weak.shape  # where N == h*w

    # Resize pseudo_mask to the feature-map resolution.
    # (This requires pseudo_mask to be of shape (B, num_instances, H, W).)
    pseudo_mask_resized = F.interpolate(pseudo_mask.float(), size=(h, w), mode='nearest') #logits
    pseudo_mask_resized = F.softmax(pseudo_mask_resized, dim=1)  # (B, num_instances, h, w)
    
    # For each pixel, select the instance with the highest probability.
    instance_ids = torch.argmax(pseudo_mask_resized, dim=1)  # shape: (B, h, w)
    instance_ids = instance_ids.view(B, -1)  # shape: (B, N)

    loss = 0.0
    for b in range(B):
        # Use the weak branch as anchors and the strong branch as positives.
        anchors   = z_weak[b]    # shape: (N, D)
        positives = z_strong[b]   # shape: (N, D)
        
        # Compute the cosine similarity matrix between all anchors and positives.
        sim_matrix = torch.matmul(anchors, positives.t()) / current_T  # shape: (N, N)
        
        # Build a binary mask where (i,j) is 1 if pixels i and j share the same predicted instance.
        pos_mask = (instance_ids[b].unsqueeze(1) == instance_ids[b].unsqueeze(0)).float()
        # Optionally, remove self-similarity by zeroing out the diagonal.
        # pos_mask = pos_mask - torch.eye(N, device=pos_mask.device)
        
        # For each anchor pixel i, sum over all positives.
        numerator = (torch.exp(sim_matrix) * pos_mask).sum(dim=1)
        denominator = torch.exp(sim_matrix).sum(dim=1) + 1e-6
        
        loss += (-torch.log(numerator / denominator + 1e-6)).mean()
    
    return loss / B


def instance_aware_negative_loss_with_memory(z_weak, pos_sim, pseudo_probs, image_ids, 
                                           memory_bank, current_T, num_negatives,
                                           hard_negative_ratio=0.5):
    """
    Computes the contrastive loss using instance-aware negative sampling with memory bank.
    Includes hard negative mining for better performance.
    
    Args:
        z_weak (Tensor): Feature embeddings from the weak branch, shape (B, N, D).
        pos_sim (Tensor): Positive similarities per anchor; shape (B, N).
        pseudo_probs (Tensor): Pseudo-label probabilities per pixel, shape (B, N, num_classes).
        image_ids (Tensor): Unique IDs for each image in batch, shape (B,).
        memory_bank (MemoryBank): Memory bank for storing and retrieving features.
        current_T (float): Current temperature.
        num_negatives (int): Number of negatives to sample per anchor.
        hard_negative_ratio (float): Ratio of hard negatives to sample (vs random).
        
    Returns:
        Tensor: Scalar loss.
    """
    B, N, D = z_weak.shape
    loss_total = 0.0
    num_hard_negatives = int(num_negatives * hard_negative_ratio)
    num_random_negatives = num_negatives - num_hard_negatives
    
    # First, get batch-internal negatives (from other images in the batch)
    batch_neg_features = []
    batch_neg_probs = []
    for b in range(B):
        # Gather batch negatives (from other images than the current one)
        other_indices = [i for i in range(B) if i != b]
        if len(other_indices) > 0:
            batch_neg_feat = torch.cat([z_weak[i] for i in other_indices], dim=0)
            batch_neg_prob = torch.cat([pseudo_probs[i] for i in other_indices], dim=0)
            batch_neg_features.append(batch_neg_feat)
            batch_neg_probs.append(batch_neg_prob)
    
    # Get memory bank negatives (from past iterations)
    memory_neg_features, memory_neg_probs = memory_bank.get_negatives(image_ids, z_weak.device)
    
    for b in range(B):
        anchors = z_weak[b]           # (N, D)
        pos_sim_b = pos_sim[b]        # (N,)
        anchor_probs = pseudo_probs[b]  # (N, num_classes)
        
        # Combine batch negatives with memory bank negatives
        neg_features_list = []
        neg_probs_list = []
        
        if len(batch_neg_features) > 0 and b < len(batch_neg_features):
            neg_features_list.append(batch_neg_features[b])
            neg_probs_list.append(batch_neg_probs[b])
            
        if memory_neg_features is not None:
            neg_features_list.append(memory_neg_features)
            neg_probs_list.append(memory_neg_probs)
            
        if len(neg_features_list) == 0:
            continue
            
        neg_feats = torch.cat(neg_features_list, dim=0)
        neg_probs = torch.cat(neg_probs_list, dim=0)
        
        # Compute debiasing weights based on class probability similarity
        # Weight negatives by how different their class distribution is from the anchor
        debias_weights = 1.0 - torch.matmul(anchor_probs, neg_probs.t())
        debias_weights = torch.clamp(debias_weights, min=0.0)
        
        # Compute cosine similarities between anchors and negatives
        sim_neg = torch.matmul(anchors, neg_feats.t()) / current_T
        
        # Select hard negatives (most similar)
        if num_hard_negatives > 0 and neg_feats.size(0) >= num_hard_negatives:
            # Get top-k most similar negatives per anchor (hard negatives)
            hard_neg_indices = torch.topk(sim_neg, k=min(num_hard_negatives, neg_feats.size(0)), dim=1).indices
            hard_neg_sim = torch.gather(sim_neg, dim=1, index=hard_neg_indices)
            
            # Select random negatives for diversity
            if num_random_negatives > 0 and neg_feats.size(0) >= num_random_negatives:
                # Use Gumbel-softmax trick for random sampling (weighted by debias_weights)
                gumbel_noise = -torch.log(-torch.log(torch.rand_like(debias_weights) + 1e-6) + 1e-6)
                scores = torch.log(debias_weights + 1e-6) + gumbel_noise
                random_neg_indices = scores.topk(min(num_random_negatives, neg_feats.size(0)), dim=1).indices
                random_neg_sim = torch.gather(sim_neg, dim=1, index=random_neg_indices)
                
                # Combine hard and random negatives
                neg_sim_sampled = torch.cat([hard_neg_sim, random_neg_sim], dim=1)
            else:
                neg_sim_sampled = hard_neg_sim
        else:
            # Fall back to random sampling if not enough negatives or hard_negative_ratio=0
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(debias_weights) + 1e-6) + 1e-6)
            scores = torch.log(debias_weights + 1e-6) + gumbel_noise
            neg_indices = scores.topk(min(num_negatives, neg_feats.size(0)), dim=1).indices
            neg_sim_sampled = torch.gather(sim_neg, dim=1, index=neg_indices)
        
        # Use the provided positive similarities
        numerator = torch.exp(pos_sim_b)
        denominator = numerator + torch.exp(neg_sim_sampled).sum(dim=1) + 1e-6
        loss_total += (-torch.log(numerator / denominator + 1e-6)).mean()
    
    # Update memory bank with current batch features
    memory_bank.update(z_weak, pseudo_probs)
    
    return loss_total / B


# def dynamic_temperature_scheduler(step, total_steps, min_temp=0.07, max_temp=0.4, warmup_steps=1000):
#     """
#     Dynamic temperature scheduling with warmup and cosine annealing.
    
#     Args:
#         step (int): Current training step.
#         total_steps (int): Total number of training steps.
#         min_temp (float): Minimum temperature value.
#         max_temp (float): Maximum temperature value.
#         warmup_steps (int): Number of warmup steps.
        
#     Returns:
#         float: Temperature value for the current step.
#     """
#     if step < warmup_steps:
#         Linear warmup
#         return max_temp - (max_temp - min_temp) * (step / max(1, warmup_steps))
#     else:
#         Cosine annealing 
#         progress = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
#         return min_temp + 0.5 * (max_temp - min_temp) * (1 + math.cos(math.pi * progress))


def pixel_contrastive_loss_with_memory(z_weak, z_strong,
                                      pseudo_weak_mask, pseudo_weak_logits, 
                                      memory_bank, image_ids, h, w,
                                      temperature=0.4, T_final=0.07, step=0,
                                      total_steps=90000, num_negatives=200,
                                      hard_negative_ratio=0.5):
    """
    Enhanced contrastive loss with memory bank and hard negative mining.
    
    Args:
        z_weak (Tensor): Feature embeddings from the weak branch, shape (B, *, D).
        z_strong (Tensor): Feature embeddings from the strong branch, shape (B, *, D).
        pseudo_weak_mask (Tensor): Pseudo-instance masks (logits) with shape (B, num_instances, H, W).
        pseudo_weak_logits (Tensor): Instance-level logits, shape (B, num_instances, num_classes+1).
        memory_bank (MemoryBank): Memory bank for storing and retrieving features.
        image_ids (Tensor): Unique IDs for each image in the batch, shape (B,).
        h, w (int): Height and width of the feature map (where N = h * w).
        temperature (float): Initial temperature.
        T_final (float): Final temperature.
        step (int): Current training step.
        total_steps (int): Total number of training steps.
        num_negatives (int): Number of negatives to sample per anchor.
        hard_negative_ratio (float): Ratio of hard negatives to sample (vs random).
        
    Returns:
        Tensor: Scalar combined loss.
    """
    # Dynamic temperature scheduling
    # T = dynamic_temperature_scheduler(step, total_steps, T_final, temperature)
    # T=0.07
    #T with exponential decay
    T = T_final + (temperature - T_final) * (0.5 ** (step / total_steps))
    
    
    B, _, _, _ = pseudo_weak_mask.shape
    N = h * w  # number of pixels per image
    D = z_weak.size(-1)

    # print(z_weak.shape)
    # print(h, w, "h, w")
    # print(D)


    
    # Reshape feature maps.
    z_weak = z_weak.view(B, N, D)
    z_strong = z_strong.view(B, N, D)
    
    # Normalize features.
    z_weak = F.normalize(z_weak, dim=-1) 
    z_strong = F.normalize(z_strong, dim=-1)
    
    # Compute positive cosine similarity between weak and strong features.
    pos_sim = torch.sum(z_weak * z_strong, dim=-1) / T # (B, N)
    
    # ----- Build per-pixel pseudo probability vectors using soft aggregation -----
    # 1. Resize instance masks to the feature map resolution: (B, num_instances, h, w)
    pseudo_mask_resized = F.interpolate(pseudo_weak_mask.float(), size=(h, w), mode='bilinear')
    pseudo_mask_resized = F.softmax(pseudo_mask_resized/0.5, dim=1)  #0.5 based on the paper
    
    # 2. Compute instance probabilities from logits using softmax: (B, num_instances, num_classes+1)
    instance_probs = F.softmax(pseudo_weak_logits, dim=-1) 
    
    # 3. Multiply each instance's soft mask (expanded) with its probability distribution.
    mask_exp = pseudo_mask_resized.unsqueeze(-1)             # (B, num_instances, h, w, 1)
    instance_probs_exp = instance_probs.unsqueeze(2).unsqueeze(3)  # (B, num_instances, 1, 1, num_classes+1)
    instance_contrib = mask_exp * instance_probs_exp           # (B, num_instances, h, w, num_classes+1)
    
    # 4. Sum contributions over instances.
    agg_instance = instance_contrib.sum(dim=1)  # (B, h, w, num_classes+1)
    
    # 5. Renormalize per pixel.
    pseudo_probs = agg_instance / (agg_instance.sum(dim=-1, keepdim=True) + 1e-6)
    # 6. Flatten to (B, N, num_classes+1)
    pseudo_probs = pseudo_probs.view(B, N, -1)
    
    # Compute the instance-positive loss
    loss_pos = instance_positive_loss(z_weak, z_strong, pseudo_weak_mask, h, w, T)
    
    # Compute the instance-aware negative sampling loss with memory bank
    loss_neg = instance_aware_negative_loss_with_memory(
        z_weak, pos_sim, pseudo_probs, image_ids, memory_bank, T, 
        num_negatives, hard_negative_ratio
    )
    
    # Combine with weighted sum.
    lambda_pos = 1.0
    lambda_neg = 1.0
    loss_total = lambda_pos * loss_pos + lambda_neg * loss_neg
    return loss_total



@META_ARCH_REGISTRY.register()
class MaskFormer(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()



        self.backbone = backbone
        self.sem_seg_head = sem_seg_head


        #this is for res5 features
        # self.projection_head_w= nn.Sequential( 
        #     nn.Linear(768, 128, bias=False),  # Ensure this matches backbone output
        # )   
        # self.projection_head_s  = nn.Sequential(
        #     nn.Linear(768, 128, bias=False),  # Ensure this matches backbone output
        # )

        self.projection_head_w = nn.Sequential( 
            nn.LayerNorm(768),
            nn.Linear(768, 256, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128, bias=False),
        )   
            
        self.projection_head_s = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 256, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128, bias=False),
        )

        # Initialize memory bank for contrastive learning
        self.memory_bank = MemoryBank(feature_dim=128, max_samples=2048, momentum=0.99)
        self.register_buffer("next_image_id", torch.tensor(0))


        
        '''# Momentum components (weak branch)
        self.momentum_backbone = copy.deepcopy(backbone)  # Requires torch.no_grad()
        self.momentum_projection_head_w = copy.deepcopy(self.projection_head_w)  # Requires torch.no_grad()
        
        # Freeze momentum parameters initially
        for param in self.momentum_backbone.parameters():
            param.requires_grad = False
        for param in self.momentum_projection_head_w.parameters():
            param.requires_grad = False
            
        # Momentum parameter
        self.momentum = 0.999  # Can make this configurable'''


        '''path ="/home/avalocal/thesis23/KD/Mask2Former_Multi_Task/output/Infonce_1860_186/model_final.pth"
        checkpoint = torch.load(path, map_location="cpu")["model"]
        #remove encoder_k and projection_head_k from checkpoint
        checkpoint = {k: v for k, v in checkpoint.items() if "encoder_k" not in k and "projection_head_k" not in k}
        #remove if head_q is in the key
        checkpoint = {k: v for k, v in checkpoint.items() if "head_q" not in k}


        backbone_weights = {k: v for k, v in checkpoint.items() if "backbone" in k}
        #encoder_q.backbone.cls_token --> backbone.cls_token
        backbone_weights ={k[19:]: v for k, v in backbone_weights.items()}
        # backbone_weights ={k[9:]: v for k, v in backbone_weights.items()}
        if "" in backbone_weights:
            del backbone_weights[""]

        sem_seg_head_weights = {k: v for k, v in checkpoint.items() if "sem_seg_head" in k}
        sem_seg_head_weights ={k[13:]: v for k, v in sem_seg_head_weights.items()}

        self.backbone.load_state_dict(backbone_weights)
        self.sem_seg_head.load_state_dict(sem_seg_head_weights)'''

        path = "/home/avalocal/Mask2Former/output/second_best/model_final.pth"

        checkpoint = torch.load(path, map_location="cpu")["model"]
        backbone_weights = {k: v for k, v in checkpoint.items() if "backbone" in k}
        backbone_weights = {k[9:]: v for k, v in backbone_weights.items()}
        self.backbone.load_state_dict(backbone_weights, strict=True)

        
        sem_seg_head_weights = {k: v for k, v in checkpoint.items() if "sem_seg_head" in k}
        sem_seg_head_weights = {k[13:]: v for k, v in sem_seg_head_weights.items()}
        self.sem_seg_head.load_state_dict(sem_seg_head_weights, strict=True)



        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image
        self.current_step = 0
        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

    # def _update_momentum(self):
    #     """Update momentum encoders using EMA"""
    #     # Backbone
    #     for param_q, param_k in zip(self.backbone.parameters(), 
    #                               self.momentum_backbone.parameters()):
    #         param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)
            
    #     # Projection head
    #     for param_q, param_k in zip(self.projection_head_w.parameters(),
    #                               self.momentum_projection_head_w.parameters()):
    #         param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT
        # contrastive_weight =1.0

        # building criterion
        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}#, "loss_contrastive_mask": contrastive_weight}

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        weight_dict.update({"loss_contrastive": 0.1})
        losses = ["labels", "masks"]#, "contrastive"]

        criterion = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # inference
            "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        '''images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        features = self.backbone(images.tensor)
        outputs = self.sem_seg_head(features)'''

        B = len(batched_inputs)
        image_ids = torch.arange(self.next_image_id, self.next_image_id + B, device=self.device)
        self.next_image_id += B

        #weak imgs
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        
        #strong imgs
        strong_imgs =[second_augmentation(x["image"].to(self.device)) for x in batched_inputs]
        strong_imgs = [(x - self.pixel_mean) / self.pixel_std for x in strong_imgs]
        strong_imgs = ImageList.from_tensors(strong_imgs, self.size_divisibility)

        # with torch.no_grad():
        # with torch.no_grad():  # Important: no gradients for momentum branch
            # self._update_momentum()  # Update momentum parameters            
            # weak_features = self.momentum_backbone(images.tensor)
        weak_features = self.backbone(images.tensor)  # Original backbone
        outputs = self.sem_seg_head(weak_features)

        with torch.no_grad():
            strong_features = self.backbone(strong_imgs.tensor)  # Original backbone

        # strong_features = self.backbone(strong_imgs.tensor)  # Original backbone
        # outputs = self.sem_seg_head(strong_features)

        #these are lowest level features for contrastive loss
        low_weak_features = weak_features["res5"]     #res5: B, 768, 19, 37 (res2:96)
        h, w= low_weak_features.shape[-2:]
        low_strong_features = strong_features["res5"] #B, 768, 19, 3
        _, C, h, w = low_weak_features.shape
        low_weak_features= low_weak_features.permute(0, 2, 3, 1).contiguous().view(-1, C) #B*H*W, C
        low_strong_features= low_strong_features.permute(0, 2, 3, 1).contiguous().view(-1, C) #B*H*W, C

        # print(low_weak_features.shape, low_strong_features.shape)

        # z_weak = self.projection_head_w(low_weak_features).detach() #B*H*W, 128
        # z_weak =self.momentum_projection_head_w(low_weak_features) #B*H*W, 128
        z_weak = self.projection_head_w(low_weak_features)      #B*H*W, 128
        z_strong = self.projection_head_s(low_strong_features)      #B*H*W, 128

        #pseudos here are copy of outputs["pred_masks"] and outputs["pred_logits"] and does not have gradient
        pseudo_weak_mask   = outputs["pred_masks"].detach().clone()
        pseudo_weak_logits = outputs["pred_logits"].detach().clone()

        

        loss_contrastive = pixel_contrastive_loss_with_memory(
            z_weak, z_strong, 
            pseudo_weak_mask, pseudo_weak_logits, 
            self.memory_bank, image_ids, h, w,
            step=self.current_step, total_steps=90000,
             num_negatives=200
        )




        # loss_contrastive = pixel_contrastive_loss(
        #     z_weak, z_strong, pseudo_weak_mask, pseudo_weak_logits, h, w,
        #     temperature=0.3, T_final=0.07, step=self.current_step, 
        #     total_steps=90000, num_negatives=200
        # )
        # loss_contrastive = pixel_contrastive_loss(
            # z_weak, z_strong, pseudo_weak_mask, pseudo_weak_logits, h, w,
            # current_T=0.3, num_negatives=200, lambda_pos=1.0, lambda_neg=1.0
        # )
        self.current_step += 1
        if self.training:
            # mask classification target
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, images)
            else:
                targets = None

            # bipartite matching-based loss
            losses = self.criterion(outputs, targets)
            losses["loss_contrastive"] = loss_contrastive

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    print(f"Loss {k} not in weight_dict")
                    assert False, "Loss should be added"
                    
            return losses
        else:
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]
            # upsample masks
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            del outputs

            processed_results = []
            for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                processed_results.append({})

                if self.sem_seg_postprocess_before_inference:
                    mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                        mask_pred_result, image_size, height, width
                    )
                    mask_cls_result = mask_cls_result.to(mask_pred_result)

                # semantic segmentation inference
                if self.semantic_on:
                    r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                    if not self.sem_seg_postprocess_before_inference:
                        r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                    processed_results[-1]["sem_seg"] = r

                # panoptic segmentation inference
                if self.panoptic_on:
                    panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                    processed_results[-1]["panoptic_seg"] = panoptic_r
                
                # instance segmentation inference
                if self.instance_on:
                    instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result)
                    processed_results[-1]["instances"] = instance_r

            return processed_results

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                }
            )
        return new_targets

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def panoptic_inference(self, mask_cls, mask_pred):
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()

        keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )

            return panoptic_seg, segments_info

    def instance_inference(self, mask_cls, mask_pred):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]

        # [Q, K]
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
        labels_per_image = labels[topk_indices]

        topk_indices = topk_indices // self.sem_seg_head.num_classes
        # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
        mask_pred = mask_pred[topk_indices]

        # if this is panoptic segmentation, we only keep the "thing" classes
        if self.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()

            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]

        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > 0).float()
        result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
        # Uncomment the following to get boxes from masks (this is slow)
        # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

        # calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        return result
