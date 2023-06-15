# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple, Union

import fvcore.nn.weight_init as weight_init
from torch import nn
from torch.nn import functional as F
import torch

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

from ..transformer.zero_shot_transformer_predictor import ZeroShotTransformerPredictor
from .pixel_decoder import build_pixel_decoder


@SEM_SEG_HEADS_REGISTRY.register()
class ZeroShotMaskFormerHead(nn.Module):

    _version = 2

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            # Do not warn if train from scratch
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if "sem_seg_head" in k and not k.startswith(prefix + "predictor") and not "pixel_decoder" in k:
                    newk = k.replace(prefix, prefix + "pixel_decoder.")
                    # logger.debug(f"{k} ==> {newk}")
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False

            if not scratch:
                logger.warning(
                    f"Weight format of {self.__class__.__name__} have changed! "
                    "Please upgrade your models. Applying automatic conversion now ..."
                )

    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        num_classes: int,
        pixel_decoder: nn.Module,
        loss_weight: float = 1.0,
        ignore_value: int = -1,
        # extra parameters
        transformer_predictor: nn.Module,
        transformer_in_feature: str,
        # clip image encoder作为backbone
        clipAsBackbone: bool,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            num_classes: number of classes to predict
            pixel_decoder: the pixel decoder module
            loss_weight: loss weight
            ignore_value: category id to be ignored during training.
            transformer_predictor: the transformer decoder that makes prediction
            transformer_in_feature: input feature name to the transformer_predictor
        """
        super().__init__()
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]
        feature_strides = [v.stride for k, v in input_shape]
        feature_channels = [v.channels for k, v in input_shape]

        self.ignore_value = ignore_value
        self.common_stride = 4
        self.loss_weight = loss_weight

        self.pixel_decoder = pixel_decoder
        self.predictor = transformer_predictor
        self.transformer_in_feature = transformer_in_feature

        self.num_classes = num_classes

        # clip image encoder作为backbone
        self.clipAsBackbone = clipAsBackbone
        if self.clipAsBackbone:
            dim = 512  # clip vit16 输出维度，可能需要修改
            out_dim = dim // 4 
            out_channels = 512  # 最终输出维度

            self.clipOutUpsampel = nn.Sequential(
                nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                # get_norm("LN", dim // 2),
                LayerNorm(dim//2),
                nn.GELU(),
                nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2),
                
                Conv2d(out_dim, out_channels, kernel_size=1, bias=False,
                # norm=get_norm("LN", out_channels),
                norm = LayerNorm(out_channels),
                ),
                Conv2d(out_channels,  out_channels, kernel_size=3, padding=1, bias=False, 
                # norm=get_norm("LN", out_channels),
                norm = LayerNorm(out_channels),
                ),
            )

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        return {
            "input_shape": {
                k: v
                for k, v in input_shape.items()
                if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
            },
            "ignore_value": cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "pixel_decoder": build_pixel_decoder(cfg, input_shape),
            "loss_weight": cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT,
            "transformer_in_feature": cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE,
            "transformer_predictor": ZeroShotTransformerPredictor(
                cfg,
                cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
                if cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == "transformer_encoder"
                else input_shape[cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE].channels,
                mask_classification=True,
            ),
            # clip image encoder 作为backbone
            "clipAsBackbone":cfg.MODEL.BACKBONE_CLIP,
        }

    def forward(self, features):
        return self.layers(features)

    def layers(self, features):
        if isinstance(features, List):
            features_imageendoder = features[0].permute(0,3,1,2)
            features_det = features[1]
            features_imageendoder = self.clipOutUpsampel(features_imageendoder)
            (
                mask_features,
                transformer_encoder_features 
            ) = self.pixel_decoder.forward_features({"res5":features_imageendoder})
        else:
            (
                mask_features,
                transformer_encoder_features,
            ) = self.pixel_decoder.forward_features(features)
        
        if self.transformer_in_feature == "transformer_encoder":
            assert (
                transformer_encoder_features is not None
            ), "Please use the TransformerEncoderPixelDecoder."
            predictions = self.predictor(transformer_encoder_features, mask_features)
        else:
            if self.clipAsBackbone:
                predictions = self.predictor(
                    features_det, mask_features
                )
            else:
                predictions = self.predictor(
                    features[self.transformer_in_feature], mask_features
                )
        return predictions

    def freeze_pretrained(self):
        for name, module in self.named_children():
            if name not in ["predictor"]:
                for param in module.parameters():
                    param.requires_grad = False
            else:
                module.freeze_pretrained()


class LayerNorm(nn.Module):
    """
    A LayerNorm variant, popularized by Transformers, that performs point-wise mean and
    variance normalization over the channel dimension for inputs that have shape
    (batch_size, channels, height, width).
    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa B950
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x