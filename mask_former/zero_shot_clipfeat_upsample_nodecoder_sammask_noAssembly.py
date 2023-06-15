# Copyright (c) Facebook, Inc. and its affiliates.
from encodings import normalize_encoding
import logging
from turtle import ScrolledCanvas
from typing import Tuple

from numpy import number
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

import fvcore.nn.weight_init as weight_init

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList
from detectron2.utils.logger import log_first_n
from detectron2.layers import Conv2d, Linear
from .modeling.clip_adapter import (
    ClipAdapter,
    MaskFormerClipAdapter,
    MaskFormerClipFeatureAdapter,
    build_prompt_learner,
)
from .mask_former_model import MaskFormer
from .modeling.transformer.transformer import TransformerDecoder,TransformerDecoderLayerLinear, TransformerDecoderLinear
from .modeling.transformer.position_encoding import PositionEmbeddingSine

from segment_anything import SamAutomaticMaskGenerator,sam_model_registry,  sam_model_registry_feature, SamAutomaticMaskGeneratorMaskFeature

@META_ARCH_REGISTRY.register()
class ZeroShotClipfeatUpsampleNodecoderSAMMaskNoAssembly(MaskFormer):
    """
    Main class for zero shot mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        clipAsBackbone: bool,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        clip_adapter: nn.Module,
        region_clip_adapter: nn.Module = None,
        criterion: nn.Module,
        num_queries: int,
        panoptic_on: bool,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        clip_ensemble: bool,
        clip_ensemble_weight: float,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        clip_kd_loss: bool,
        maskformer_hiddendim: int,
        clipKdProj: bool,

        model_type_sam = "vit_b",
        # ckpt_sam = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/segment-anything/segment_anything/pretrain_weights/sam_vit_b_01ec64.pth",
        ckpt_sam = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/segment-anything/segment_anything/pretrain_weights/sam_vit_l_0b3195.pth",
        points_per_side_sam = 8,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            clip_adapter: adapter for clip-based mask classification
            num_queries: int, number of queries
            panoptic_on: bool, whether to output panoptic segmentation prediction
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
        """
        super().__init__(
            clipAsBackbone=clipAsBackbone,
            backbone=backbone,
            sem_seg_head=sem_seg_head,
            criterion=criterion,
            num_queries=num_queries,
            panoptic_on=panoptic_on,
            object_mask_threshold=object_mask_threshold,
            overlap_threshold=overlap_threshold,
            metadata=metadata,
            size_divisibility=size_divisibility,
            sem_seg_postprocess_before_inference=sem_seg_postprocess_before_inference,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
        )
        self.clip_adapter: ClipAdapter = clip_adapter
        self._region_clip_adapter = region_clip_adapter

        self.clip_ensemble: bool = clip_ensemble
        self.clip_ensemble_weight: float = clip_ensemble_weight
        self.clip_kd_loss: bool = clip_kd_loss
        self.clipKdProj: bool = clipKdProj
        if self.clipKdProj:
            self.kd_proj = nn.Linear(maskformer_hiddendim,maskformer_hiddendim)

        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.sem_seg_head.parameters():
            param.requires_grad = False


        # clip feature 和 mask 结合后 decoder 部分
        d_model = 1024
        nhead = 8
        dim_feedforward = 2048
        dropout = 0.1
        activation = "relu"
        normalize_before = False
        decoder_layer = TransformerDecoderLayerLinear(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        
        decoder_norm = nn.LayerNorm(d_model)
        # decoder_norm = None
        # decoder_norm = None
        num_decoder_layers = 3
        return_intermediate_dec = True
        self.decoder = TransformerDecoderLinear(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
        )
        hidden_dim = 1024
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        # self.query_embed = nn.Embedding(num_queries, hidden_dim)
        clip_rn50=True
        if clip_rn50:
            clip_feature_channels = 1024
        else:
            clip_feature_channels = 512
        clip_feature_proj_channels = 256
        self.input_proj = Conv2d(clip_feature_channels, clip_feature_proj_channels, kernel_size=1)
        weight_init.c2_xavier_fill(self.input_proj)

        self.decoder_query_proj = Linear(clip_feature_channels, clip_feature_proj_channels)
        weight_init.c2_xavier_fill(self.decoder_query_proj)

        # learnable param 3D NxHxW
        self.query_embedding = nn.Parameter(torch.empty(1024,160,160))
        # weight_init.c2_xavier_fill(self.query_embedding)
        nn.init.kaiming_uniform_(self.query_embedding, a=1)

        # decoder 输出
        embed_hidden_dim = 1024
        if clip_rn50:
            embedding_dim=1024
        else:
            embedding_dim = 512
        embed_layers = 2
        self.class_embed = MLP(
                hidden_dim, embed_hidden_dim, embedding_dim, embed_layers
            )
        
        sam = sam_model_registry[model_type_sam](checkpoint=ckpt_sam)
        
        self.mask_generator_feature = SamAutomaticMaskGenerator(sam, points_per_side = points_per_side_sam, min_mask_region_area= 100)
 
        for param in self.mask_generator_feature.predictor.model.parameters():
            param.requires_grad = False
        self.num_imgs = 0

    @classmethod
    def from_config(cls, cfg):
        init_kwargs = MaskFormer.from_config(cfg)
        prompt_learner = build_prompt_learner(cfg.MODEL.CLIP_ADAPTER)
        region_clip_adapter = None
        if cfg.MODEL.CLIP_ADAPTER.SEPERATE_ADAPTER:
            log_first_n(
                logging.WARNING,
                "Using different head for region classification and query classification",
            )
            cls_prompt_learner = build_prompt_learner(
                cfg.MODEL.CLIP_ADAPTER.REGION_CLIP_ADAPTER
            )
            region_clip_adapter =MaskFormerClipFeatureAdapter(
                cfg.MODEL.CLIP_ADAPTER.REGION_CLIP_ADAPTER.CLIP_MODEL_NAME,
                cls_prompt_learner,
                maskselfattn = True
            )
        # import pdb; pdb.set_trace()
        clip_adapter = MaskFormerClipFeatureAdapter(
            cfg.MODEL.CLIP_ADAPTER.CLIP_MODEL_NAME,
            prompt_learner,
            maskselfattn=True,
        )
        # import pdb;pdb.set_trace()
        
        init_kwargs["clip_adapter"] = clip_adapter
        init_kwargs["region_clip_adapter"] = region_clip_adapter
        init_kwargs["clip_ensemble"] = cfg.MODEL.CLIP_ADAPTER.CLIP_ENSEMBLE
        init_kwargs[
            "clip_ensemble_weight"
        ] = cfg.MODEL.CLIP_ADAPTER.CLIP_ENSEMBLE_WEIGHT

        init_kwargs["clip_kd_loss"] = cfg.MODEL.MASK_FORMER.CLIP_KD_LOSS
        init_kwargs["clipAsBackbone"] = cfg.MODEL.BACKBONE_CLIP
        init_kwargs["clipKdProj"] = cfg.MODEL.MASK_FORMER.CLIP_KD_PROJ
        init_kwargs["maskformer_hiddendim"] = cfg.MODEL.SEM_SEG_HEAD.EMBEDDING_DIM

        init_kwargs["model_type_sam"] = cfg.MODEL.MASK_FORMER.SAM_MODELTYPE
        init_kwargs["ckpt_sam"]=cfg.MODEL.MASK_FORMER.SAM_CKPT
        init_kwargs["points_per_side_sam"]=cfg.MODEL.MASK_FORMER.SAM_POINTS_PER_SIDE
        # init_kwargs["maskselfattnsoftmax"] = True
        return init_kwargs

    def forward(self, batched_inputs, class_list= None):
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
        dataset_name = [x["meta"]["dataset_name"] for x in batched_inputs]
        assert len(set(dataset_name)) == 1
        dataset_name = dataset_name[0]

        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        # import pdb; pdb.set_trace()
        if not class_list or len(class_list) == 0:
            class_names = self.get_class_name_list(dataset_name)
        else:
            class_names = class_list
        text_features = self.clip_adapter.get_text_features(class_names)
        
        
        with torch.no_grad():
            image_tensor_sam = [x["image"].permute(1,2,0).cpu().numpy() for x in batched_inputs]
            semseg_pred_masks = []
            
            self.mask_generator_feature.predictor.model.to(self.device)
            for imgTensor in image_tensor_sam:
                sam_masks = []
                
                mask_datas = self.mask_generator_feature.generate(imgTensor)
                sorted_mask_datas = sorted(mask_datas, key=lambda x: x["area"], reverse=True)
                sam_masks=[seg["segmentation"] for seg in sorted_mask_datas]
                
                sam_masks=self.sam_mask_filter(sam_masks)
                semseg_pred_mask = torch.stack(sam_masks)
                semseg_pred_mask = semseg_pred_mask.to(self.device)
                semseg_pred_masks.append(semseg_pred_mask)
                
        ################################################################################################
            semseg_pred_masks = torch.stack(semseg_pred_masks)
        
        processed_results = []
        for input_per_image, image_size in zip(
            batched_inputs, images.image_sizes
        ):
            image = input_per_image["image"].to(self.device)
            
            r, sam_seg_masks=self.semantic_inference_gt(image, class_names, dataset_name, text_features, semseg_pred_masks)
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            
            r = F.one_hot(r, num_classes=len(class_names)+1).permute(2,0,1)
            
            processed_results.append({"sem_seg": r, "pred_masks":semseg_pred_masks, "pred_sam_masks":sam_seg_masks})
            self.num_imgs += 1
                
            return processed_results
    
    def sam_mask_filter(self, sorted_masks):
        w, h = sorted_masks[0].shape
        mask = np.zeros((w, h))
        sub_overlap = []
        for mask_data in sorted_masks:
            mask = mask + mask_data
            if (mask > 1.1).any():
                mask_data[mask>1.1] = False
                mask[mask>1.1] = 1.0
            sub_overlap.append(mask_data)
        # import pdb; pdb.set_trace()
        filter_masks = []
        for mask_data in sorted_masks:
            if mask_data.any():
                filter_masks.append(torch.from_numpy(mask_data))
        return filter_masks
    def semantic_inference_gt(self, image, class_names, dataset_name, text_features, sam_masks):
        # semseg_masks, classes_gts = self._semseg2semmask(semseg_gt)
        image = (image-self.pixel_mean) / self.pixel_std
        clip_feature = self.clip_adapter.clip_model.visual(image.unsqueeze(0), return_cls = False, mask_feature = sam_masks,  maskselfattnsoftmax = False)
        
        clip_feature = clip_feature[0].permute(0,3,1,2)
        # import pdb ; pdb.set_trace()
        h, w = clip_feature.shape[2], clip_feature.shape[3]
        clip_feature = F.interpolate(clip_feature,size=(h*4, w*4),mode="bilinear",align_corners=False)
        # semseg_masks_downsample=F.interpolate(semseg_masks.unsqueeze(0),size=(h*4, w*4),mode="nearest")
        sam_masks = sam_masks.to(image)
        semseg_pred_masks_downsample = F.interpolate(sam_masks, size=(h*4, w*4), mode="nearest")
        validIdx = semseg_pred_masks_downsample.sum(dim = (-1,-2)) > 0.5
        semseg_pred_masks_downsample = semseg_pred_masks_downsample[validIdx][None, :, :, :]
       
        # clip_maskfeature=torch.einsum("bchw,bqhw->bqc", clip_feature, semseg_masks_downsample)
        clip_maskfeature = torch.einsum("bchw,bqhw->bqc", clip_feature, semseg_pred_masks_downsample)

        
        if clip_maskfeature.shape[0] == 1:
            clip_maskfeature = clip_maskfeature.squeeze(0)
        classes_clip = self.get_sim_logits(text_features,  self.normalize_feature(clip_maskfeature))
        
        if(len(classes_clip.shape)==1):
            classes_clip= classes_clip.unsqueeze(0)
        class_masks = classes_clip.argmax(dim=-1)
        # import pdb;pdb.set_trace()

        sam_masks = sam_masks[validIdx]
        # semseg_gt_out= (class_masks[:, None, None] * semseg_masks).sum(dim = 0).to(torch.int64)
        semseg_sam_out = (class_masks[:, None, None] * sam_masks).sum(dim = 0).to(torch.int64)
        return semseg_sam_out, sam_masks

    def semantic_inference(self, mask_cls, mask_pred, image, class_names, dataset_name):
        # import pdb; pdb.set_trace()
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        
        # get the classification result from clip model
        if self.clip_ensemble:
            clip_cls, valid_flag = self.region_clip_adapter(
                image, class_names, mask_pred, normalize=True
            )
            if clip_cls is None:
                clip_cls = torch.empty(0, mask_cls.shape[-1] + 1, device=self.device)
            # import pdb; pdb.set_trace()
            # softmax before index or after?
            clip_cls = F.softmax(clip_cls[:, :-1], dim=-1)
            if self.clip_ensemble_weight > 0:
                map_back_clip_cls = mask_cls.new_ones(mask_cls.shape)
                map_back_clip_cls[valid_flag] = clip_cls
                if hasattr(MetadataCatalog.get(dataset_name), "trainable_flag"):
                    trained_mask = torch.Tensor(
                        MetadataCatalog.get(dataset_name).trainable_flag
                    ).to(mask_cls.device)[None, :]
                else:
                    trained_mask = mask_cls.new_zeros(mask_cls.shape)
                mask_cls = trained_mask * torch.pow(
                    mask_cls, self.clip_ensemble_weight
                ) * torch.pow(map_back_clip_cls, 1 - self.clip_ensemble_weight) + (
                    1 - trained_mask
                ) * torch.pow(
                    mask_cls, 1 - self.clip_ensemble_weight
                ) * torch.pow(
                    map_back_clip_cls, self.clip_ensemble_weight
                )
            else:
                # only clip model predictions are used
                mask_cls = clip_cls
                mask_pred = mask_pred[valid_flag]
                mask_pred[mask_pred > 0.5] = 1.0
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def get_class_name_list(self, dataset_name):
        class_names = [
            c.strip() for c in MetadataCatalog.get(dataset_name).stuff_classes
        ]
        return class_names

    @property
    def region_clip_adapter(self):
        if self._region_clip_adapter is None:
            return self.clip_adapter
        return self._region_clip_adapter

    def kd_loss_cal(self, output, indices):
        semseg_pred_logits =output["pred_region_logits"]
        target_regionfeature_results=output["clip_region_logits"]
        # import pdb; pdb.set_trace()
        if self.clipKdProj:
            semseg_pred_logits = self.kd_proj(semseg_pred_logits)
        target_regionfeature_valid = output["clip_region_valid"]
        # import pdb; pdb.set_trace()
        # semseg_pred_logits = semseg_pred_logits[target_regionfeature_valid]
        # for idx in range(semseg_pred_logits.shape[0]):
        #     semseg_pred_logits[idx] = semseg_pred_logits[idx][target_regionfeature_valid[idx]]
        #     target_regionfeature_results[idx] = target_regionfeature_results[idx][target_regionfeature_valid[idx]]
        # import pdb; pdb.set_trace()
        # target_regionfeature_results = target_regionfeature_results[target_regionfeature_valid]
        # if()
        src_idx = self._get_src_permutation_idx(indices)
        loss_kd = F.l1_loss(semseg_pred_logits[src_idx], target_regionfeature_results[src_idx])
        # if target_regionfeature_valid.sum() > 0:
        #     loss_kd = F.l1_loss(semseg_pred_logits[target_regionfeature_valid],target_regionfeature_results[target_regionfeature_valid])
        # else:
        #     loss_kd = F.l1_loss(semseg_pred_logits,target_regionfeature_results)
        return {"loss_kd": loss_kd}
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _semseg2semmask(self,semseg_gt):
        sem_seg_gt = semseg_gt.cpu().numpy()
        
        classes = np.unique(sem_seg_gt)
        classes = classes[classes != 255]
        masks = []
        classes_gt = []
        for class_id in classes:
            mask = np.zeros_like(semseg_gt.cpu())
            mask[sem_seg_gt == class_id] = 1.0
            masks.append(mask)
            classes_gt.append(class_id)
        if len(masks) == 0:
            # Some image does not have annotation (all ignored)
            masks= torch.zeros(
                (0, sem_seg_gt.shape[-2], sem_seg_gt.shape[-1])
            )
        else:
            masks=torch.stack(
                [
                    torch.from_numpy(np.ascontiguousarray(x.copy()))
                    for x in masks
                ]
            )        
        return masks.to(torch.float32).to(semseg_gt.device), torch.tensor(classes_gt).to(torch.int64).to(semseg_gt.device)
    def get_sim_logits(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor,
        temperature: float = 100,
    ):
        return temperature * image_features @ text_features.T
    def normalize_feature(self, feat: torch.Tensor):

        return feat / feat.norm(dim=-1, keepdim=True)



class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x