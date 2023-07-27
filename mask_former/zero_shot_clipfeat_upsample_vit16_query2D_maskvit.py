# Copyright (c) Facebook, Inc. and its affiliates.
from encodings import normalize_encoding
import logging
from turtle import ScrolledCanvas
from typing import Tuple
import copy

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
from .modeling.transformer.transformer import TransformerDecoder,TransformerDecoderLayerLinear, TransformerDecoderLinear, TransformerDecoderLayer
from .modeling.transformer.position_encoding import PositionEmbeddingSine

@META_ARCH_REGISTRY.register()
class ZeroShotClipfeatUpsample_addnorm_vit16Query2D_maskvit(MaskFormer):
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
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        clip_kd_loss: bool,
        maskformer_hiddendim: int,
        clipKdProj: bool,
        num_decoder_layer: int,
        output_sig,
        mask_downsample,
        decoder_mask_feature,
        maskselfattn,
        maskselfattnsoftmax,
        ps_shortcut,
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

        self.clip_kd_loss: bool = clip_kd_loss
        self.clipKdProj: bool = clipKdProj
        if self.clipKdProj:
            self.kd_proj = nn.Linear(maskformer_hiddendim,maskformer_hiddendim)

        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.sem_seg_head.parameters():
            param.requires_grad = False
        
        self.ps_shortcut = ps_shortcut

        self.maskselfattn = maskselfattn
        self.maskselfattnsoftmax = maskselfattnsoftmax
        # clip feature 和 mask 结合后 decoder 部分
        d_model = 512
        nhead = 8
        dim_feedforward = 1024
        dropout = 0.1
        activation = "relu"
        normalize_before = False
        
        decoder_layer = TransformerDecoderLayerLinear(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before, num_decoder_layer>0,decoder_mask_feature
        )
      
        if num_decoder_layer > 0:
            decoder_norm = nn.LayerNorm(d_model)
        else:
            decoder_norm = None

        num_decoder_layers = num_decoder_layer
        return_intermediate_dec = True
        self.decoder = TransformerDecoderLinear(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
        )
        hidden_dim = 512
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        # learnable param 3D NxHxW
        num_queries = 100
        self.query_embedding = nn.Embedding(num_queries, hidden_dim)

        # 2023-01-31 更新，src_output sigmoid超参
        self.output_sig = output_sig

        self.mask_downsample = mask_downsample

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
            )
        # import pdb; pdb.set_trace()
        learn_position = cfg.MODEL.CLIP_ADAPTER.LEARN_POSITION
        position_layers = cfg.MODEL.CLIP_ADAPTER.POSITION_LAYERS
        input_resolution_run= cfg.INPUT.LEARNPOSITIONRES

        learn_token = cfg.MODEL.CLIP_ADAPTER.LEARN_TOKEN
        prompt_num_token = cfg.MODEL.CLIP_ADAPTER.VPT_NUM_TOKEN

        layermaskvit = cfg.MODEL.CLIP_ADAPTER.LAYERMASKVIT
        maskselfattn = cfg.MODEL.CLIP_ADAPTER.MASKSELFATTN
        maskselfattnsoftmax=cfg.MODEL.CLIP_ADAPTER.MASKSELFATTNSOFTMAX
        ps_shortcut = cfg.MODEL.CLIP_ADAPTER.PS_SHORTCUT
        # import pdb; pdb.set_trace()
        clip_adapter = MaskFormerClipFeatureAdapter(
            cfg.MODEL.CLIP_ADAPTER.CLIP_MODEL_NAME,
            prompt_learner,
            learn_token, prompt_num_token,
            learn_position,position_layers,
            input_resolution_run,layermaskvit,
            maskselfattn,
        )

        init_kwargs["mask_downsample"] = cfg.MODEL.MASK_FORMER.MASKDOWNSAMPLE
        init_kwargs["output_sig"] = cfg.INPUT.OUTPUT_SIG
        init_kwargs["clip_adapter"] = clip_adapter
        init_kwargs["region_clip_adapter"] = region_clip_adapter

        init_kwargs["clip_kd_loss"] = cfg.MODEL.MASK_FORMER.CLIP_KD_LOSS
        init_kwargs["clipAsBackbone"] = cfg.MODEL.BACKBONE_CLIP
        init_kwargs["clipKdProj"] = cfg.MODEL.MASK_FORMER.CLIP_KD_PROJ
        init_kwargs["maskformer_hiddendim"] = cfg.MODEL.SEM_SEG_HEAD.EMBEDDING_DIM
        init_kwargs["num_decoder_layer"] = cfg.MODEL.NUM_DECODER_LAYER 
        init_kwargs["decoder_mask_feature"] = cfg.MODEL.DECODER_MASK_FEATURE
        init_kwargs["maskselfattn"] = maskselfattn
        init_kwargs["maskselfattnsoftmax"] = maskselfattnsoftmax
        init_kwargs["ps_shortcut"] = ps_shortcut
        return init_kwargs

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
        dataset_name = [x["meta"]["dataset_name"] for x in batched_inputs]
        assert len(set(dataset_name)) == 1
        dataset_name = dataset_name[0]

        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        

        # clip image encoder 作为backbone
        # import pdb; pdb.set_trace()
        if self.clipAsBackbone:
            features = self.backbone(images.tensor)
            outputs = self.sem_seg_head(features)
        else:
            with torch.no_grad():
                features = self.backbone(images.tensor)
                outputs = self.sem_seg_head(features)

        class_names = self.get_class_name_list(dataset_name)
        text_features = self.clip_adapter.get_text_features(class_names)

        
        ################################################################################################

        semseg_pred_mask = outputs["pred_masks"]
        
        if self.maskselfattn:
            clip_images_features,_ = self.clip_adapter.clip_model.visual(images.tensor, return_cls = False,\
            mask_feature=semseg_pred_mask, maskselfattnsoftmax = self.maskselfattnsoftmax, ps_shortcut = self.ps_shortcut)
        else:
            clip_images_features,_ = self.clip_adapter.clip_model.visual(images.tensor, return_cls = False,\
         mask_feature=semseg_pred_mask, ps_shortcut = self.ps_shortcut)



        clip_images_features = clip_images_features / clip_images_features.norm(dim=-1, keepdim=True)
        
        clip_images_features = clip_images_features.permute(0, 3,1,2) # BCHW
        # import pdb; pdb.set_trace()
        semseg_pred_mask = outputs["pred_masks"]
        semseg_pred_mask = F.sigmoid(semseg_pred_mask)
        bs, h, w =clip_images_features.shape[0], clip_images_features.shape[2],clip_images_features.shape[3]
       
        # 2022-11-14 添加
        # 2022-12-09 修改
        scaleRatio = 1
        # 2022-11-14 添加
        # 2022-12-09 修改
        semseg_mask_downsample = F.interpolate(semseg_pred_mask, size=(h*scaleRatio,w*scaleRatio), mode="bilinear", align_corners=False)
        # clip_images_features = F.interpolate(clip_images_features, size=(h*scaleRatio, w*scaleRatio), mode="nearest")
        # 2022-11-13 添加
        if self.mask_downsample:
            outputs["pred_masks"] = semseg_mask_downsample
        # 2022-11-13 添加
        
        decoder_query_position = self.query_embedding.weight.unsqueeze(1).repeat(1, bs, 1)      
        
        pos_embed = self.pe_layer(clip_images_features)
        pos_embed = pos_embed.flatten(2).permute(2,0,1)
        # query_embed = self.query_embed.weight
        # bs = clip_images_features.shape[0]
        # query_embed = query_embed.unsqueeze(1).repeat(1,bs,1)
        # import pdb; pdb.set_trace()
        tgt = torch.zeros_like(decoder_query_position)
        mask = None
        # memory = self.input_proj(clip_images_features)
        memory = clip_images_features
        memory = memory.flatten(2).permute(2,0,1)

  
        hs = self.decoder(
            tgt,
            memory,
            memory_key_padding_mask=mask,
            pos=pos_embed,
            query_pos=decoder_query_position,
            proposal_mask=semseg_mask_downsample,
        )
        # import pdb; pdb.set_trace()
        hs = hs.transpose(1,2)
        # outputs_class=self.class_embed(hs)
        outputs_class = hs
        # import pdb; pdb.set_trace()
        outputs["pred_logits"] = outputs_class[-1]


        outputs.pop("aux_outputs")
        if "aux_outputs" in outputs.keys():
            for i in range(len(outputs["aux_outputs"])):
                outputs["aux_outputs"][i][
                    "pred_logits"
                ] = outputs_class[i]
        
        # outputs["pred_logits"] = self.clip_adapter.get_sim_logits(
        #     text_features, self.clip_adapter.normalize_feature(outputs["pred_logits"])
        # )
        outputs["pred_logits"] = self.clip_adapter.get_sim_logits(
            text_features, outputs["pred_logits"]
        )
       
        ################################################################################################

        if self.training:
            if "aux_outputs" in outputs.keys():
                for i in range(len(outputs["aux_outputs"])):
                    outputs["aux_outputs"][i][
                        "pred_logits"
                    ] = self.clip_adapter.get_sim_logits(
                        text_features,
                        self.clip_adapter.normalize_feature(
                            outputs["aux_outputs"][i]["pred_logits"]
                        ),
                    )
            # mask classification target
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, images)
                for idx, x in enumerate(batched_inputs):
                    targets[idx]["sem_seg"] = x["sem_seg"]
                # import pdb; pdb.set_trace()
            else:
                targets = None

            # bipartite matching-based loss
            losses, indicesSrc = self.criterion(outputs, targets, self.output_sig)
            if self.clip_kd_loss:
                outputs["pred_region_logits"] = semseg_pred_logits
                # outputs["pred_masks_semseg"]
                target_regionfeature_results, target_regionfeature_valid= self.kd_region_feature(semseg_pred_mask,batched_inputs, images)
                outputs["clip_region_logits"] = target_regionfeature_results
                outputs["clip_region_valid"] = target_regionfeature_valid
                losses.update(self.kd_loss_cal(output=outputs, indices=indicesSrc))
            # import pdb; pdb.set_trace()
            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            # print(losses)

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

            processed_results = []
            for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
            ):
                height = image_size[0]
                width = image_size[1]
                mask_pred_result = sem_seg_postprocess(
                    mask_pred_result, image_size, height, width
                )
                image = input_per_image["image"].to(self.device)
                # semantic segmentation inference
                r = self.semantic_inference(
                    mask_cls_result, mask_pred_result, image, class_names, dataset_name
                )
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = sem_seg_postprocess(r, image_size, height, width)
                processed_results.append({"sem_seg": r, "pred_masks":mask_pred_result})

                # evluate ap:
                # panoptic segmentation inference
                if self.panoptic_on:
                    panoptic_r = self.panoptic_inference(
                        mask_cls_result, mask_pred_result
                    )
                    processed_results[-1]["panoptic_seg"] = panoptic_r
            return processed_results
    
    # image 经过 clip image encode
    def kd_region_feature(self, mask_pred_results,batched_inputs, images):
        # upsample masks
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(images.tensor.shape[-2], images.tensor.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )
        regionfeature_results = []
        regionfeature_valid = []
        for mask_pred_result, input_per_image, image_size in zip(
             mask_pred_results, batched_inputs, images.image_sizes
        ):
            height = image_size[0]
            width = image_size[1]
            mask_pred_result = sem_seg_postprocess(
                mask_pred_result, image_size, height, width
            )
            image = input_per_image["image"].to(self.device)
            mask_pred_result = mask_pred_result.sigmoid()
            region_features, valid_flag = self.region_clip_adapter.get_region_features(
                image, mask_pred_result, normalize=True
            )
            regionfeature_results.append(region_features)
            regionfeature_valid.append(valid_flag)
        return torch.stack(regionfeature_results), torch.stack(regionfeature_valid)

        
    def semantic_inference(self, mask_cls, mask_pred, image, class_names, dataset_name):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
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