from collections import OrderedDict
from typing import Tuple, Union
from scipy import interpolate
import numpy as np

from functools import reduce
from operator import mul

import torch
import torch.nn.functional as F
from torch import nn

from torch.nn.modules.utils import _pair

from torch import _VF
from typing import Optional

from torch.nn.parameter import Parameter
import torch.nn.init as init
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
import math


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(
                OrderedDict(
                    [
                        ("-1", nn.AvgPool2d(stride)),
                        (
                            "0",
                            nn.Conv2d(
                                inplanes,
                                planes * self.expansion,
                                1,
                                stride=1,
                                bias=False,
                            ),
                        ),
                        ("1", nn.BatchNorm2d(planes * self.expansion)),
                    ]
                )
            )

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(
        self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5
        )
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads
        self.grid_size = spacial_dim

    def forward(self, x, mask=None, return_cls=True, mask_feature=None):
        b, c, gh, gw = x.shape
        # remove irrelated feature
        if mask is not None:
            mask = F.interpolate(mask[:, None, ...], size=(gh, gw)).squeeze(
                1
            )  # [N,H,W] -> [N,grid,grid]
            mask = (mask > 0.5).reshape(mask.shape[0], -1)
            mask = torch.cat([mask, mask.new_ones(mask.shape[0], 1)], dim=1)
            if x.size()[0] == 1:
                x = x.expand(mask.shape[0], c, gh, gw)

        x = x.reshape(x.shape[0], c, gh * gw).permute(2, 0, 1)  # NCHW -> (HW)NC

        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        positional_embedding = self.positional_embedding
        if not (self.positional_embedding.shape[0] == x.shape[0]):
            cls_pos = positional_embedding[0:1, :]
            per_pos_embedding = (
                F.interpolate(
                    positional_embedding[1:, :]
                    .permute(1, 0)
                    .view(1, -1, self.grid_size, self.grid_size),
                    size=(gh, gw),
                    mode="bicubic",
                )
                .reshape(-1, gh * gw)
                .permute(1, 0)
            )
            positional_embedding = torch.cat([cls_pos, per_pos_embedding])

        x = x + positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        # 2022-11-15修改，去掉resnet中self-attention部分
        # import pdb; pdb.set_trace()
        # x = torch._C._nn.linear(x, self.c_proj.weight, self.c_proj.bias)
        # x, _ = F.multi_head_attention_forward(
        #     query=x,
        #     key=x,
        #     value=x,
        #     embed_dim_to_check=x.shape[-1],
        #     num_heads=self.num_heads,
        #     q_proj_weight=self.q_proj.weight,
        #     k_proj_weight=self.k_proj.weight,
        #     v_proj_weight=self.v_proj.weight,
        #     in_proj_weight=None,
        #     in_proj_bias=torch.cat(
        #         [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]
        #     ),
        #     bias_k=None,
        #     bias_v=None,
        #     add_zero_attn=False,
        #     dropout_p=0,
        #     out_proj_weight=self.c_proj.weight,
        #     out_proj_bias=self.c_proj.bias,
        #     use_separate_proj_weight=True,
        #     training=self.training,
        #     need_weights=False,
        #     key_padding_mask=mask,
        # )
        x, _=multi_head_attention_forward_my(
            query=x,
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],   #
            num_heads=self.num_heads,           #
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None, #
            in_proj_bias=torch.cat(
                [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]
            ),   #
            bias_k=None,         #
            bias_v=None,         #
            add_zero_attn=False,    # 
            dropout_p=0,       # 
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
            key_padding_mask=mask,
            mask_feature = mask_feature,
        )
        if return_cls:
            return x[0]
        else:
            return x


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(
            3, width // 2, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(
            width // 2, width // 2, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(
            input_resolution // 32, embed_dim, heads, output_dim
        )

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, mask: torch.Tensor = None, return_cls=True, mask_feature = None):
        def stem(x):
            for conv, bn in [
                (self.conv1, self.bn1),
                (self.conv2, self.bn2),
                (self.conv3, self.bn3),
            ]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)  # 1/4,1/4
        x = self.layer1(x)
        x = self.layer2(x)  # 1/8,1/8
        x = self.layer3(x)  # 1/16,1/16
        x = self.layer4(x)  # 1/32,1/32
        b, c, gh, gw = x.shape
        x = self.attnpool(x, mask, return_cls, mask_feature)
        if not return_cls:
            return x[1:].permute(1, 0, 2).reshape(b, gh, gw, x.shape[-1])  # N,L,C
        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, **kwargs):
        self.attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None
            else None
        )
        return self.attn(
            x, x, x, need_weights=False, attn_mask=self.attn_mask, **kwargs
        )[0]

    def forward(self, x: torch.Tensor, **kwargs):
        x = x + self.attention(self.ln_1(x), **kwargs)
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)]
        )

    def forward(self, x: torch.Tensor, **kwargs):
        for block in self.resblocks:
            x = block(x, **kwargs)
        return x

class ResidualAttentionBlockVit(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, \
    outlayer = False,maskselfattn = False, layerVit = -1):
        super().__init__()

        # self.attn = nn.MultiheadAttention(d_model, n_head)
        self.layerVit = -1
        if outlayer:
            self.attn = MultiheadAttentionVit(d_model, n_head, maskselfattn = maskselfattn, layerVit = layerVit)
            self.layerVit = layerVit
        else:
            self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, **kwargs):
        self.attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None
            else None
        )
        return self.attn(
            x, x, x, need_weights=False, attn_mask=self.attn_mask, **kwargs
        )[0]
    def attention_getattn(self, x: torch.Tensor, **kwargs):
        self.attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None
            else None
        )
        
        x, selfattn, maskattn = self.attn(
            x, x, x, need_weights=True, attn_mask=self.attn_mask, **kwargs
        )
        return [selfattn, maskattn]

    def forward(self, x: torch.Tensor, **kwargs):
        return_attention = False
        if "return_attention" in kwargs and kwargs["return_attention"]:
            return_attention = True
            
        if "return_attention" in kwargs:
            kwargs.pop("return_attention")
        if return_attention:
            return self.attention_getattn(self.ln_1(x), **kwargs)
        
        alphaShortcut = 1.0
        if self.layerVit== 11:
            # import pdb; pdb.set_trace()
            alphaShortcut = 0.2
        x = alphaShortcut * x + self.attention(self.ln_1(x), **kwargs)
        
        x = x + self.mlp(self.ln_2(x))
        return x

class ResidualAttentionBlockVitNoattn(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, outlayer = False):
        super().__init__()

        # self.attn = nn.MultiheadAttention(d_model, n_head)
        if outlayer:
            self.attn = MultiheadAttentionVitNoattn(d_model, n_head)
        else:
            self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, **kwargs):
        self.attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None
            else None
        )
        return self.attn(
            x, x, x, need_weights=False, attn_mask=self.attn_mask, **kwargs
        )[0]

    def forward(self, x: torch.Tensor, **kwargs):
        return_attention = False
        if "return_attention" in kwargs and kwargs["return_attention"]:
            return_attention = True
        if "return_attention" in kwargs:
            kwargs.pop("return_attention")
        if return_attention:
            return self.attention(self.ln_1(x), **kwargs)
        x = x + self.attention(self.ln_1(x), **kwargs)
        
        x = x + self.mlp(self.ln_2(x))
        return x
class ResidualAttentionBlockVitToken(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, outlayer = False):
        super().__init__()

        # self.attn = nn.MultiheadAttention(d_model, n_head)
        if outlayer:
            self.attn = MultiheadAttentionVitToken(d_model, n_head)
        else:
            self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, **kwargs):
        self.attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None
            else None
        )
        return self.attn(
            x, x, x, need_weights=True, attn_mask=self.attn_mask, **kwargs
        )[0]
    def attention_getattn(self, x: torch.Tensor, **kwargs):
        self.attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None
            else None
        )
        return self.attn(
            x, x, x, need_weights=True, attn_mask=self.attn_mask, **kwargs
        )[1]

    def forward(self, x: torch.Tensor, **kwargs):
        return_attention = False
        if "return_attention" in kwargs and kwargs["return_attention"]:
            return_attention = True
        if "return_attention" in kwargs:
            kwargs.pop("return_attention")
        if return_attention:
            return self.attention_getattn(self.ln_1(x), **kwargs)
        latent=self.attention(self.ln_1(x), **kwargs)
        # if latent.shape
        # import pdb; pdb.set_trace()
        if x.shape[0] != latent.shape[0]:
            len_tokens = x.shape[0] - latent.shape[0]
            x = torch.cat((
                x[:1,:,:],
                x[1+len_tokens:, :,:]
            ),dim = 0)
        x = x + latent
        
        x = x + self.mlp(self.ln_2(x))
        return x

class ResidualAttentionBlockVitPosition(nn.Module):
    def __init__(self, d_model: int, n_head: int,vision_width,learn_position = False,\
        input_resolution_run = 384, attn_mask: torch.Tensor = None, outlayer = False, \
        maskselfattn = False, layerVit = -1):
        super().__init__()

        # self.attn = nn.MultiheadAttention(d_model, n_head)
        self.layerVit = -1
        if outlayer:
            self.attn = MultiheadAttentionVit(d_model, n_head, maskselfattn=maskselfattn, layerVit= layerVit)
            self.layerVit = layerVit
        else:
            self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask


        ########################################################################################
        # 第一层已经在最外面的transformer中添加了  目前vit中共有12层transformer
        self.learn_position = learn_position
        if learn_position:
            # import pdb; pdb.set_trace()
            input_resolution_run = input_resolution_run
            patch_size = 16
            width = vision_width
            scale = width ** -0.5
            self.learned_position = nn.Parameter(
                scale * torch.randn((input_resolution_run // patch_size) ** 2 + 1, width)
            )
        self.grid_size = input_resolution_run // 16

        ########################################################################################

    def attention(self, x: torch.Tensor, **kwargs):
        self.attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None
            else None
        )
        return self.attn(
            x, x, x, need_weights=False, attn_mask=self.attn_mask, **kwargs
        )[0]
    def attention_getattn(self, x: torch.Tensor, **kwargs):
        self.attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None
            else None
        )
        return self.attn(
            x, x, x, need_weights=False, attn_mask=self.attn_mask, **kwargs
        )[1]

    def forward(self, x: torch.Tensor, **kwargs):
        return_attention = False
        
        ps_shortcut = 1.0
        if "return_attention" in kwargs and kwargs["return_attention"]:
            return_attention = True
        if "return_attention" in kwargs:
            kwargs.pop("return_attention")
        if return_attention:
            return self.attention_getattn(self.ln_1(x), **kwargs)
        if "gh" in kwargs:
            gh = kwargs["gh"]
            gw = kwargs["gw"]
            kwargs.pop("gh")
            kwargs.pop("gw")
        if "ps_shortcut" in kwargs:
            ps_shortcut = kwargs["ps_shortcut"]
            kwargs.pop("ps_shortcut") 
        if self.learn_position:
            learned_positional_embedding = self.learned_position
            if x.shape[0] != self.learned_position.shape[0]:
                cls_pos = learned_positional_embedding[0:1, :]
                
                per_pos_embedding = (
                    F.interpolate(
                        learned_positional_embedding[1:, :]
                        .permute(1, 0)
                        .view(1, -1, self.grid_size, self.grid_size),
                        size=(gh, gw),
                        mode="bicubic",
                    )
                    .reshape(-1, gh * gw)
                    .permute(1, 0)
                ) 
                
                learned_positional_embedding = torch.cat([cls_pos, per_pos_embedding])
            x = (x.permute(1,0,2)+ learned_positional_embedding).permute(1,0,2)


        
            
        # if self.layerVit== 11:
        #     alphaShortcut = ps_shortcut
        #     x = alphaShortcut * x + self.attention(self.ln_1(x), **kwargs)
        # else:
        x = x + self.attention(self.ln_1(x), **kwargs)
        
        x = x + self.mlp(self.ln_2(x))
        return x
class TransformerVit(nn.Module):
    def __init__(
        self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, \
        layermaskvit = [11], maskselfattn = False
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        # import pdb; pdb.set_trace()
        self.layermaskvit = layermaskvit
        tmp = []
        for num_layer in range(self.layers):
            if num_layer in layermaskvit:
                # if num_layer == 11:
                #     tmp.append(ResidualAttentionBlockVitNoattn(width, heads, attn_mask, outlayer = True))
                # else:
                #     tmp.append(ResidualAttentionBlockVit(width, heads, attn_mask, outlayer = True))
                tmp.append(ResidualAttentionBlockVit(width, heads, attn_mask, outlayer = True, \
                maskselfattn = maskselfattn, layerVit = num_layer))
            else:
                tmp.append(ResidualAttentionBlockVit(width, heads, attn_mask))
        # num = layers-self.layermaskvit
        # tmp = [ResidualAttentionBlockVit(width, heads, attn_mask) for _ in range(layers-num)]
        # if num >0:
        #     tmp += [ResidualAttentionBlockVit(width, heads, attn_mask, outlayer = True)]
        #     if num > 1:
        #         tmp += [ResidualAttentionBlockVit(width, heads, attn_mask, outlayer = True) for _ in range(layers-num+1,layers)]
        self.resblocks= nn.Sequential(
            *tmp
        )
        # self.resblocks = nn.Sequential(
        #     *[ResidualAttentionBlockVit(width, heads, attn_mask) for _ in range(layers)]
        # )

    def forward(self, x: torch.Tensor, **kwargs):
        layerNum = 0
        intermediate_feature = []
        intermediate_feature.append(x)
        for block in self.resblocks:
            # x = block(x)
            # if layerNum -1== self.layermaskvit:
            
            if layerNum in self.layermaskvit or layerNum ==self.layers-1:
                if layerNum in self.layermaskvit:
                    x = block(x, **kwargs)
                else:
                    kwargs.pop("mask_feature")
                    kwargs.pop("maskselfattnsoftmax")
                    x = block(x, **kwargs)
            else:    
                x = block(x)
            layerNum += 1
            intermediate_feature.append(x)
        return x, intermediate_feature

class TransformerVitToken(nn.Module):
    def __init__(
        self, width: int, layers: int, heads: int, patch_size: int,attn_mask: torch.Tensor = None,num_tokens=5, \
        dropout_rate=0.1,layermaskvit = [11,], prompt_layers=[0,1,2,3,4,5,6,7,8,9,10,11]
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        
        self.layermaskvit = layermaskvit
        self.prompt_layers =prompt_layers

        tmp = []
        for num_layer in range(self.layers):
            if num_layer in layermaskvit:
                tmp.append(ResidualAttentionBlockVitToken(width, heads, attn_mask, outlayer = True))
            else:
                tmp.append(ResidualAttentionBlockVitToken(width, heads, attn_mask))


        # num = layers-self.layermaskvit
        # tmp = [ResidualAttentionBlockVitToken(width, heads, attn_mask) for _ in range(layers-num)]
        # if num >0:
        #     tmp += [ResidualAttentionBlockVitToken(width, heads, attn_mask, outlayer = True)]
        #     if num > 1:
        #         tmp += [ResidualAttentionBlockVitToken(width, heads, attn_mask, outlayer = True) for _ in range(layers-num+1,layers)]
        self.resblocks= nn.Sequential(
            *tmp
        )
        
        prompt_dim = width
        patch_size_tuple = _pair((patch_size, patch_size))
        val= math.sqrt(6. / float(3 * reduce(mul, patch_size_tuple, 1) + prompt_dim))  # noqa
        # import pdb; pdb.set_trace()
        total_d_layer = len(prompt_layers)
        self.num_tokens = num_tokens
        self.deep_prompt_embeddings  = nn.Parameter(torch.zeros(
                    total_d_layer, self.num_tokens, prompt_dim))
        # xavier_uniform initialization
        nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)
        self.prompt_dropout = torch.nn.Dropout(dropout_rate)

        # self.resblocks = nn.Sequential(
        #     *[ResidualAttentionBlockVit(width, heads, attn_mask) for _ in range(layers)]
        # )

    def forward(self, x: torch.Tensor, **kwargs):
        layerNum = 0
        intermediate_feature = []
        intermediate_feature.append(x)
        L,B,D=x.shape[0],x.shape[1],x.shape[2]
        # 输入x中包括了patch embedding+position embedding
        layer_prompt_num = 0
        for block in self.resblocks:
            # x = block(x)
            
            # if layerNum -1== self.layermaskvit:
            # import pdb; pdb.set_trace()
            if layerNum in self.prompt_layers:
                deep_prompt_emb=self.prompt_dropout(self.deep_prompt_embeddings[layer_prompt_num].expand(B,-1,-1)).permute(1,0,2)
                layer_prompt_num += 1
                if layerNum==0 or (not (layerNum -1) in self.prompt_layers ):
                    x = torch.cat((
                        x[:1,:,:],
                        deep_prompt_emb,
                        x[1:, :,:]
                    ),dim=0)
                else:
                    x = torch.cat((
                        x[:1,:,:],
                        deep_prompt_emb,
                        x[1+self.num_tokens:, :,:]
                    ),dim=0)

            if layerNum in self.layermaskvit:
                x = block(x, **kwargs)
            else:    
                x = block(x)
            
            layerNum += 1
            intermediate_feature.append(x)
        if self.layermaskvit==[]:
            x = torch.cat(
                (x[:1,:,:],
                    x[1+self.num_tokens:, :,:]
                ),dim=0)
        return x, intermediate_feature

class TransformerVitPosition(nn.Module):
    def __init__(
        self, width: int, layers: int, heads: int, vision_width,attn_mask: torch.Tensor = None,\
             position_layers=[],resolution_run=384,layermaskvit = [11], maskselfattn = False,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        
        self.layermaskvit = layermaskvit

        tmp = []
        for num_layer in range(self.layers):
            if num_layer in position_layers and num_layer in self.layermaskvit:
                tmp.append(ResidualAttentionBlockVitPosition(width, heads, vision_width,learn_position = True, \
                    input_resolution_run=resolution_run,attn_mask = attn_mask,outlayer = True, maskselfattn = maskselfattn,\
                        layerVit= num_layer))
            elif num_layer in position_layers and (not num_layer in self.layermaskvit):
                tmp.append(ResidualAttentionBlockVitPosition(width, heads,  vision_width, learn_position = True, \
                    input_resolution_run=resolution_run, attn_mask = attn_mask))
                pass
            elif (not num_layer in position_layers) and (num_layer in self.layermaskvit):
                tmp.append(ResidualAttentionBlockVitPosition(width, heads, vision_width,learn_position = False, \
                    input_resolution_run=resolution_run,attn_mask = attn_mask,outlayer = True, maskselfattn = maskselfattn, \
                        layerVit= num_layer))
                pass
            else:
                tmp.append(ResidualAttentionBlockVitPosition(width, heads,  vision_width, learn_position = False,\
                     input_resolution_run=resolution_run, attn_mask = attn_mask))
                pass

        
        self.resblocks= nn.Sequential(
            *tmp
        )
        
    def forward(self, x: torch.Tensor, **kwargs):
        layerNum = 0
        intermediate_feature = []
        intermediate_feature.append(x)
        for block in self.resblocks:
            # x = block(x)
            
            input_size = {}
            if("gh" in kwargs):
                input_size.update({"gh": kwargs["gh"]})
                input_size.update({"gw": kwargs["gw"]})
            if layerNum in self.layermaskvit:
                x = block(x, **kwargs)
            else:    
                x = block(x, **input_size)
            layerNum += 1
            intermediate_feature.append(x)
        return x, intermediate_feature


class VisionTransformer(nn.Module):
    def __init__(
        self,
        input_resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
        layermaskvit = [11],
        maskselfattn = False,
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width)
        )
        self.grid_size = input_resolution // patch_size
        self.ln_pre = LayerNorm(width)

        # self.transformer = Transformer(width, layers, heads)
        self.transformer = TransformerVit(width, layers, heads, layermaskvit = layermaskvit, \
        maskselfattn = maskselfattn)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        inter_method="bicubic",
        return_cls=True,
        mask_feature: torch.Tensor = None,
        return_attention=False,
        maskselfattnsoftmax = False,
    ):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        b, _, gh, gw = x.size()
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        positional_embedding = self.positional_embedding
        
        if not (self.positional_embedding.shape[0] == x.shape[1]):
            cls_pos = positional_embedding[0:1, :]
            if inter_method in ["bicubic", "bilinear"]:
                per_pos_embedding = (
                    F.interpolate(
                        positional_embedding[1:, :]
                        .permute(1, 0)
                        .view(1, -1, self.grid_size, self.grid_size),
                        size=(gh, gw),
                        mode="bicubic",
                    )
                    .reshape(-1, gh * gw)
                    .permute(1, 0)
                )
            else:

                def geometric_progression(a, r, n):
                    return a * (1.0 - r ** n) / (1.0 - r)

                left, right = 1.01, 1.5
                while right - left > 1e-6:
                    q = (left + right) / 2.0
                    gp = geometric_progression(1, q, self.grid_size // 2)
                    if gp > gh // 2:
                        right = q
                    else:
                        left = q
                dis = []
                cur = 1
                for i in range(self.grid_size // 2):
                    dis.append(cur)
                    cur += q ** (i + 1)
                r_ids = [-_ for _ in reversed(dis)]
                if self.grid_size % 2 == 0:
                    y = r_ids + dis
                else:
                    y = r_ids + [0] + dis
                left, right = 1.01, 1.5
                while right - left > 1e-6:
                    q = (left + right) / 2.0
                    gp = geometric_progression(1, q, self.grid_size // 2)
                    if gp > gw // 2:
                        right = q
                    else:
                        left = q
                dis = []
                cur = 1
                for i in range(self.grid_size // 2):
                    dis.append(cur)
                    cur += q ** (i + 1)
                r_ids = [-_ for _ in reversed(dis)]
                if self.grid_size % 2 == 0:
                    x = r_ids + [0] + dis[:-1]
                else:
                    x = r_ids + [0] + dis
                dx = np.arange(-gw // 2, gw / 2, 1.0)
                dy = np.arange(-gh // 2, gh / 2, 1.0)
                all_rel_pos_bias = []

                for i in range(positional_embedding.shape[-1]):
                    z = (
                        positional_embedding[1:, i]
                        .view(self.grid_size, self.grid_size)
                        .float()
                        .numpy()
                    )
                    f_cubic = interpolate.interp2d(x, y, z, kind="cubic")
                    all_rel_pos_bias.append(
                        torch.Tensor(f_cubic(dx, dy))
                        .contiguous()
                        .view(-1, 1)
                        .to(positional_embedding.device)
                    )
                per_pos_embedding = torch.cat(all_rel_pos_bias, dim=-1)

            positional_embedding = torch.cat([cls_pos, per_pos_embedding])
        x = x + positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        # remove irrelated feature
        if mask is not None:
            mask = F.interpolate(mask[:, None, ...], size=(gh, gw)).squeeze(
                1
            )  # [N,H,W] -> [N,grid,grid]
            mask = (mask > 0.5).reshape(mask.shape[0], -1)
            mask = torch.cat([mask, mask.new_ones(mask.shape[0], 1)], dim=1)
            if x.size()[1] == 1:
                x = x.expand(x.size()[0], mask.shape[0], x.size()[2])

        x,intermediate_feature = self.transformer(x, key_padding_mask=mask, mask_feature = mask_feature,\
         return_attention = return_attention,maskselfattnsoftmax = maskselfattnsoftmax )

        if return_attention:
            return x
        x = x.permute(1, 0, 2)  # LND -> NLD
        if return_cls:
            x = self.ln_post(x[:, 0, :])
        else:
            x = self.ln_post(x[:, 1:, :])

        if self.proj is not None:
            x = x @ self.proj
        if not return_cls:
            x = x.reshape(b, gh, gw, -1)
        return x, intermediate_feature
class VisionTransformerLearnToken(nn.Module):
    def __init__(
        self,
        input_resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
        prompt_num_token:int =5,
        prompt_layers=[0,1,2,3,4,5,6,7,8,9,10,11], 
        layermaskvit = [11]
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width)
        )
        self.grid_size = input_resolution // patch_size
        self.ln_pre = LayerNorm(width)
        
        self.transformer = TransformerVitToken(width, layers, heads, patch_size, num_tokens=prompt_num_token,\
            layermaskvit = layermaskvit, prompt_layers = prompt_layers)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        inter_method="bicubic",
        return_cls=True,
        mask_feature: torch.Tensor = None,
        return_attention=False
    ):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        b, _, gh, gw = x.size()
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        positional_embedding = self.positional_embedding
        
        if not (self.positional_embedding.shape[0] == x.shape[1]):
            cls_pos = positional_embedding[0:1, :]
            if inter_method in ["bicubic", "bilinear"]:
                per_pos_embedding = (
                    F.interpolate(
                        positional_embedding[1:, :]
                        .permute(1, 0)
                        .view(1, -1, self.grid_size, self.grid_size),
                        size=(gh, gw),
                        mode="bicubic",
                    )
                    .reshape(-1, gh * gw)
                    .permute(1, 0)
                )
            else:

                def geometric_progression(a, r, n):
                    return a * (1.0 - r ** n) / (1.0 - r)

                left, right = 1.01, 1.5
                while right - left > 1e-6:
                    q = (left + right) / 2.0
                    gp = geometric_progression(1, q, self.grid_size // 2)
                    if gp > gh // 2:
                        right = q
                    else:
                        left = q
                dis = []
                cur = 1
                for i in range(self.grid_size // 2):
                    dis.append(cur)
                    cur += q ** (i + 1)
                r_ids = [-_ for _ in reversed(dis)]
                if self.grid_size % 2 == 0:
                    y = r_ids + dis
                else:
                    y = r_ids + [0] + dis
                left, right = 1.01, 1.5
                while right - left > 1e-6:
                    q = (left + right) / 2.0
                    gp = geometric_progression(1, q, self.grid_size // 2)
                    if gp > gw // 2:
                        right = q
                    else:
                        left = q
                dis = []
                cur = 1
                for i in range(self.grid_size // 2):
                    dis.append(cur)
                    cur += q ** (i + 1)
                r_ids = [-_ for _ in reversed(dis)]
                if self.grid_size % 2 == 0:
                    x = r_ids + [0] + dis[:-1]
                else:
                    x = r_ids + [0] + dis
                dx = np.arange(-gw // 2, gw / 2, 1.0)
                dy = np.arange(-gh // 2, gh / 2, 1.0)
                all_rel_pos_bias = []

                for i in range(positional_embedding.shape[-1]):
                    z = (
                        positional_embedding[1:, i]
                        .view(self.grid_size, self.grid_size)
                        .float()
                        .numpy()
                    )
                    f_cubic = interpolate.interp2d(x, y, z, kind="cubic")
                    all_rel_pos_bias.append(
                        torch.Tensor(f_cubic(dx, dy))
                        .contiguous()
                        .view(-1, 1)
                        .to(positional_embedding.device)
                    )
                per_pos_embedding = torch.cat(all_rel_pos_bias, dim=-1)

            positional_embedding = torch.cat([cls_pos, per_pos_embedding])
        x = x + positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        # remove irrelated feature
        if mask is not None:
            mask = F.interpolate(mask[:, None, ...], size=(gh, gw)).squeeze(
                1
            )  # [N,H,W] -> [N,grid,grid]
            mask = (mask > 0.5).reshape(mask.shape[0], -1)
            mask = torch.cat([mask, mask.new_ones(mask.shape[0], 1)], dim=1)
            if x.size()[1] == 1:
                x = x.expand(x.size()[0], mask.shape[0], x.size()[2])

        x,intermediate_feature = self.transformer(x, key_padding_mask=mask, mask_feature = mask_feature,\
         return_attention = return_attention )
        # import pdb; pdb.set_trace()
        if return_attention:
            return x
        
        x = x.permute(1, 0, 2)  # LND -> NLD
        if return_cls:
            x = self.ln_post(x[:, 0, :])
        else:
            x = self.ln_post(x[:, 1:, :])

        if self.proj is not None:
            x = x @ self.proj
        if not return_cls:
            x = x.reshape(b, gh, gw, -1)
        return x, intermediate_feature


class VisionTransformerLearnPosition(nn.Module):
    def __init__(
        self,
        input_resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
        position_layers:list,
        input_resolution_run: int=384,
        layermaskvit = [11],
        maskselfattn = False,
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width)
        )
        ########################################################################################
        self.learned_position = nn.Parameter(
            scale * torch.randn((input_resolution_run // patch_size) ** 2 + 1, width)
        )
        ########################################################################################


        self.grid_size = input_resolution // patch_size
        self.ln_pre = LayerNorm(width)

        self.transformer = TransformerVitPosition(width, layers, heads, position_layers=position_layers,\
            resolution_run = input_resolution_run, vision_width = width,layermaskvit =layermaskvit,\
            maskselfattn = maskselfattn)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        inter_method="bicubic",
        return_cls=True,
        mask_feature: torch.Tensor = None,
        return_attention=False, 
        maskselfattnsoftmax = False,
        vitL = False,
        ps_shortcut = 0.2,
    ):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        b, _, gh, gw = x.size()
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        positional_embedding = self.positional_embedding
        if not (self.positional_embedding.shape[0] == x.shape[1]):
            cls_pos = positional_embedding[0:1, :]
            if inter_method in ["bicubic", "bilinear"]:
                per_pos_embedding = (
                    F.interpolate(
                        positional_embedding[1:, :]
                        .permute(1, 0)
                        .view(1, -1, self.grid_size, self.grid_size),
                        size=(gh, gw),
                        mode="bicubic",
                    )
                    .reshape(-1, gh * gw)
                    .permute(1, 0)
                )
            else:

                def geometric_progression(a, r, n):
                    return a * (1.0 - r ** n) / (1.0 - r)

                left, right = 1.01, 1.5
                while right - left > 1e-6:
                    q = (left + right) / 2.0
                    gp = geometric_progression(1, q, self.grid_size // 2)
                    if gp > gh // 2:
                        right = q
                    else:
                        left = q
                dis = []
                cur = 1
                for i in range(self.grid_size // 2):
                    dis.append(cur)
                    cur += q ** (i + 1)
                r_ids = [-_ for _ in reversed(dis)]
                if self.grid_size % 2 == 0:
                    y = r_ids + dis
                else:
                    y = r_ids + [0] + dis
                left, right = 1.01, 1.5
                while right - left > 1e-6:
                    q = (left + right) / 2.0
                    gp = geometric_progression(1, q, self.grid_size // 2)
                    if gp > gw // 2:
                        right = q
                    else:
                        left = q
                dis = []
                cur = 1
                for i in range(self.grid_size // 2):
                    dis.append(cur)
                    cur += q ** (i + 1)
                r_ids = [-_ for _ in reversed(dis)]
                if self.grid_size % 2 == 0:
                    x = r_ids + [0] + dis[:-1]
                else:
                    x = r_ids + [0] + dis
                dx = np.arange(-gw // 2, gw / 2, 1.0)
                dy = np.arange(-gh // 2, gh / 2, 1.0)
                all_rel_pos_bias = []

                for i in range(positional_embedding.shape[-1]):
                    z = (
                        positional_embedding[1:, i]
                        .view(self.grid_size, self.grid_size)
                        .float()
                        .numpy()
                    )
                    f_cubic = interpolate.interp2d(x, y, z, kind="cubic")
                    all_rel_pos_bias.append(
                        torch.Tensor(f_cubic(dx, dy))
                        .contiguous()
                        .view(-1, 1)
                        .to(positional_embedding.device)
                    )
                per_pos_embedding = torch.cat(all_rel_pos_bias, dim=-1)

            positional_embedding = torch.cat([cls_pos, per_pos_embedding])
         
        learned_position = self.learned_position
        if not (self.learned_position.shape[0] == x.shape[1]):
            cls_pos_learn = self.learned_position[0:1, :]
            length_grid = self.learned_position.shape[0]-1
            pos_learn_grid = int(length_grid ** 0.5)
            pos_learn_embedding = (
                    F.interpolate(
                        self.learned_position[1:, :]
                        .permute(1, 0)
                        .view(1, -1, pos_learn_grid, pos_learn_grid),
                        size=(gh, gw),
                        mode="bicubic",
                    )
                    .reshape(-1, gh * gw)
                    .permute(1, 0)
                )
            learned_position = torch.cat([cls_pos_learn, pos_learn_embedding])

        x = x + positional_embedding.to(x.dtype) + learned_position
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        # remove irrelated feature
        if mask is not None:
            mask = F.interpolate(mask[:, None, ...], size=(gh, gw)).squeeze(
                1
            )  # [N,H,W] -> [N,grid,grid]
            mask = (mask > 0.5).reshape(mask.shape[0], -1)
            mask = torch.cat([mask, mask.new_ones(mask.shape[0], 1)], dim=1)
            if x.size()[1] == 1:
                x = x.expand(x.size()[0], mask.shape[0], x.size()[2])

        x,intermediate_feature = self.transformer(x, key_padding_mask=mask, mask_feature = mask_feature,\
         return_attention = return_attention, gh = gh, gw=gw, maskselfattnsoftmax = maskselfattnsoftmax,\
         vitL = vitL, ps_shortcut=ps_shortcut)
        if return_attention:
            return x

        x = x.permute(1, 0, 2)  # LND -> NLD
        if return_cls:
            x = self.ln_post(x[:, 0, :])
        else:
            x = self.ln_post(x[:, 1:, :])

        if self.proj is not None:
            x = x @ self.proj
        if not return_cls:
            x = x.reshape(b, gh, gw, -1)
        return x, intermediate_feature

class CLIP(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        # vision
        image_resolution: int,
        vision_layers: Union[Tuple[int, int, int, int], int],
        vision_width: int,
        vision_patch_size: int,
        # text
        context_length: int,
        vocab_size: int,
        transformer_width: int,
        transformer_heads: int,
        transformer_layers: int,
        position_layers:list,
        input_resolution_run: int = 384,
        learn_position:bool=False,
        learn_token: bool=False,
        prompt_num_token: int = 5,
        layermaskvit = [11],
        maskselfattn = False,
    ):
        super().__init__()

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width,
            )
        else:
            # import pdb; pdb.set_trace()
            vision_heads = vision_width // 64
            if not learn_position and not learn_token:
                self.visual = VisionTransformer(
                    input_resolution=image_resolution,
                    patch_size=vision_patch_size,
                    width=vision_width,
                    layers=vision_layers,
                    heads=vision_heads,
                    output_dim=embed_dim,
                    layermaskvit= layermaskvit,
                    maskselfattn = maskselfattn,
                )
            elif learn_token:
                self.visual = VisionTransformerLearnToken(
                    input_resolution=image_resolution,
                    patch_size=vision_patch_size,
                    width=vision_width,
                    layers=vision_layers,
                    heads=vision_heads,
                    output_dim=embed_dim,
                    prompt_num_token=prompt_num_token,
                    prompt_layers=position_layers,
                    layermaskvit=layermaskvit,
                )
            elif learn_position:
                self.visual = VisionTransformerLearnPosition(
                    input_resolution=image_resolution,
                    patch_size=vision_patch_size,
                    width=vision_width,
                    layers=vision_layers,
                    heads=vision_heads,
                    output_dim=embed_dim,
                    position_layers=position_layers,
                    input_resolution_run = input_resolution_run,   
                    layermaskvit = layermaskvit, 
                    maskselfattn = maskselfattn, 
                )
            else:
                pass

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(
            torch.empty(self.context_length, transformer_width)
        )
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [
                self.visual.layer1,
                self.visual.layer2,
                self.visual.layer3,
                self.visual.layer4,
            ]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * (
            (2 * self.transformer.layers) ** -0.5
        )
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image, **kwargs):
        return self.visual(image.type(self.dtype), **kwargs)

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x
    
    def encode_text_all(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x) @ self.text_projection

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x


    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [
                *[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]],
                "in_proj_bias",
                "bias_k",
                "bias_v",
            ]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)

def build_model(maskselfattn, learn_token, layermaskvit,prompt_num_token,learn_position,position_layers,\
input_resolution_run,state_dict: dict):
    vit = "visual.proj" in state_dict
    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [
                k
                for k in state_dict.keys()
                if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")
            ]
        )
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round(
            (state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5
        )
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [
            len(
                set(
                    k.split(".")[2]
                    for k in state_dict
                    if k.startswith(f"visual.layer{b}")
                )
            )
            for b in [1, 2, 3, 4]
        ]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round(
            (state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5
        )
        vision_patch_size = None
        assert (
            output_width ** 2 + 1
            == state_dict["visual.attnpool.positional_embedding"].shape[0]
        )
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(
        set(
            k.split(".")[2]
            for k in state_dict
            if k.startswith(f"transformer.resblocks")
        )
    )

    model = CLIP(
        embed_dim,
        image_resolution,
        vision_layers,
        vision_width,
        vision_patch_size,
        context_length,
        vocab_size,
        transformer_width,
        transformer_heads,
        transformer_layers,
        
        position_layers,
        input_resolution_run,
        learn_position,
        
        learn_token,
        prompt_num_token,
        layermaskvit = layermaskvit,
        maskselfattn = maskselfattn,
    )
    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
  
    model_dict = model.state_dict()
    model_dict.update(state_dict)
    
    model.load_state_dict(model_dict)
    return model.eval()
    

Tensor = torch.Tensor
def multi_head_attention_forward_my(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,   #
    num_heads: int,           #
    q_proj_weight: Tensor,
    k_proj_weight: Tensor,
    v_proj_weight: Tensor,
    in_proj_weight: Tensor, #
    in_proj_bias: Tensor,   #
    bias_k: Tensor,         #
    bias_v: Tensor,         #
    add_zero_attn: bool,    # 
    dropout_p: float,       # 
    out_proj_weight: Tensor,
    out_proj_bias: Tensor,
    use_separate_proj_weight: bool,
    training: bool,
    need_weights:bool,
    key_padding_mask: Tensor,
    mask_feature: Tensor,
):
    tgt_len, bsz, embed_dim = query.size()
    src_len, _, _ = key.shape
    assert embed_dim == embed_dim_to_check
    assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5
    
    if isinstance(embed_dim, torch.Tensor):
        # embed_dim can be a tensor when JIT tracing
        head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
    else:
        head_dim = embed_dim // num_heads

    q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
    len1, len2 = q_proj_weight_non_opt.size()
    
    assert len1 == embed_dim and len2 == query.size(-1)

    k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
    len1, len2 = k_proj_weight_non_opt.size()
    assert len1 == embed_dim and len2 == key.size(-1)

    v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
    len1, len2 = v_proj_weight_non_opt.size()
    assert len1 == embed_dim and len2 == value.size(-1)

    if in_proj_bias is not None:
        q = linear(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim])
        k = linear(key, k_proj_weight_non_opt, in_proj_bias[embed_dim : (embed_dim * 2)])
        v = linear(value, v_proj_weight_non_opt, in_proj_bias[(embed_dim * 2) :])

    q = q * scaling

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    src_len = k.size(1)

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len
    
    BSZ,Mbs, Mh, Mw = mask_feature.shape
    

    Mh_new = int(Mh//8)
    Mw_new = int(Mw//8)
    mask_feature = F.interpolate(mask_feature, size=(Mh_new, Mw_new), mode = "nearest")
    mask_feature = mask_feature.to(torch.float32)
    
    mask_feature = mask_feature.sigmoid()
    
    mask_flatten =mask_feature.reshape(bsz, -1, Mh_new*Mw_new)
    mask_flatten_repeat = mask_flatten.repeat(num_heads, 1,1)
    
    mask_list = []
    for i in range(bsz):
        mask_ = mask_flatten_repeat[i::bsz, :,:]
        mask_list.append(mask_)
    
    mask_new = torch.stack(mask_list)
    mask_new = mask_new.view(bsz*num_heads,100,-1)
    
    tmp = torch.ones(bsz*num_heads,100,1).to(mask_new)
    mask_new =torch.cat((tmp, mask_new), dim = 2)
    k_mask = torch.einsum("qlc,qnl->qnlc", k, mask_new)
    attn_output_weights_mask = torch.einsum("qlc,qndc->qnld", q, k_mask)

    attn_output_weights_mask = softmax(attn_output_weights_mask, dim=-1)
    attn_output_weights_mask=dropout(attn_output_weights_mask, p=dropout_p, training=training)

    attn_output = torch.einsum("qnld, qdc->qnlc", attn_output_weights_mask, v)
    attn_output = attn_output.sum(dim = 1)
    
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]


    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
    
    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None
def linear(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    r"""
    Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.
    This operator supports :ref:`TensorFloat32<tf32_on_ampere>`.
    Shape:
        - Input: :math:`(N, *, in\_features)` N is the batch size, `*` means any number of
          additional dimensions
        - Weight: :math:`(out\_features, in\_features)`
        - Bias: :math:`(out\_features)`
        - Output: :math:`(N, *, out\_features)`
    """
    
    return torch._C._nn.linear(input, weight, bias)


def softmax(input: Tensor, dim: Optional[int] = None, _stacklevel: int = 3, dtype: Optional[int] = None) -> Tensor:
    r"""Applies a softmax function.
    Softmax is defined as:
    :math:`\text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}`
    It is applied to all slices along dim, and will re-scale them so that the elements
    lie in the range `[0, 1]` and sum to 1.
    See :class:`~torch.nn.Softmax` for more details.
    Args:
        input (Tensor): input
        dim (int): A dimension along which softmax will be computed.
        dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
          If specified, the input tensor is casted to :attr:`dtype` before the operation
          is performed. This is useful for preventing data type overflows. Default: None.
    .. note::
        This function doesn't work directly with NLLLoss,
        which expects the Log to be computed between the Softmax and itself.
        Use log_softmax instead (it's faster and has better numerical properties).
    """
    
    if dim is None:
        dim = _get_softmax_dim("softmax", input.dim(), _stacklevel)
    if dtype is None:
        ret = input.softmax(dim)
    else:
        ret = input.softmax(dim, dtype=dtype)
    return ret

# Activation functions
def dropout(input: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False) -> Tensor:
    r"""
    During training, randomly zeroes some of the elements of the input
    tensor with probability :attr:`p` using samples from a Bernoulli
    distribution.
    See :class:`~torch.nn.Dropout` for details.
    Args:
        p: probability of an element to be zeroed. Default: 0.5
        training: apply dropout if is ``True``. Default: ``True``
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``
    """
    if p < 0.0 or p > 1.0:
        raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
    return _VF.dropout_(input, p, training) if inplace else _VF.dropout(input, p, training)

class MultiheadAttentionVit(nn.Module):
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, \
    maskselfattn=False, layerVit = -1):
        super(MultiheadAttentionVit, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.maskselfattn = maskselfattn
        self.layerVit = layerVit

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = _LinearWithBias(embed_dim, embed_dim)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super(MultiheadAttentionVit, self).__setstate__(state)
    
    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None, \
                mask_feature : Optional[Tensor] = None, maskselfattnsoftmax = False,\
                vitL = False) -> Tuple[Tensor, Optional[Tensor]]:
        import pdb; pdb.set_trace()
        if self.maskselfattn and not maskselfattnsoftmax:
            
            return multi_head_attention_forward_vit_softmax_noattn_maskselfattn_addselfattn(        
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, mask_feature = mask_feature, maskselfattnsoftmax=maskselfattnsoftmax,
                layerVit = self.layerVit)
        if self.maskselfattn:
            return multi_head_attention_forward_vit_softmax_noattn_maskselfattn(        
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, mask_feature = mask_feature, maskselfattnsoftmax=maskselfattnsoftmax)
        elif vitL: 
            return multi_head_attention_forward_vit_softmax_noattn_vitL(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, mask_feature = mask_feature)
        else:
            return multi_head_attention_forward_vit_softmax_noattn(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, mask_feature = mask_feature)



class MultiheadAttentionVitNoattn(nn.Module):
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None):
        super(MultiheadAttentionVitNoattn, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = _LinearWithBias(embed_dim, embed_dim)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super(MultiheadAttentionVit, self).__setstate__(state)
    
    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None, \
                mask_feature : Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        return multi_head_attention_forward_vit_softmax_noattn_delall(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, mask_feature = mask_feature)


class MultiheadAttentionVitToken(nn.Module):
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, learn_token=True):
        super(MultiheadAttentionVitToken, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim

        self.learn_token = learn_token
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = _LinearWithBias(embed_dim, embed_dim)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super(MultiheadAttentionVit, self).__setstate__(state)
    
    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None, \
                mask_feature : Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        return multi_head_attention_forward_vit_softmax_noattn(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, mask_feature = mask_feature, learn_token=self.learn_token)
            
def multi_head_attention_forward_vit(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Tensor,
    in_proj_bias: Tensor,
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Tensor,
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
    mask_feature: Optional[Tensor] = None,
) -> Tuple[Tensor, Optional[Tensor]]:

    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    # allow MHA to have different sizes for the feature dimension
    assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5
    if not use_separate_proj_weight:
        if (query is key or torch.equal(query, key)) and (key is value or torch.equal(key, value)):
            # self-attention
            q, k, v = linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)
    q = q * scaling

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    src_len = k.size(1)
    
    #####################################################################################################################
   
    BSZ, Mbs, Mh, Mw = mask_feature.shape

    Mh_new = int(Mh//4)
    Mw_new = int(Mw//4)
    mask_feature = F.interpolate(mask_feature, size=(Mh_new, Mw_new), mode = "nearest")
    mask_feature = mask_feature.to(torch.float32)
    
    mask_feature = mask_feature.sigmoid()
    
    mask_flatten =mask_feature.reshape(bsz, -1, Mh_new*Mw_new)
    mask_flatten_repeat = mask_flatten.repeat(num_heads, 1,1)
    
    mask_list = []
    for i in range(bsz):
        mask_ = mask_flatten_repeat[i::bsz, :,:]
        mask_list.append(mask_)
    
    mask_new = torch.stack(mask_list)
    
    mask_new = mask_new.view(bsz*num_heads,100,-1)

    
    tmp = torch.ones(bsz*num_heads,100,1).to(mask_new)
    mask_new =torch.cat((tmp, mask_new), dim = 2)
    k_mask = torch.einsum("qlc,qnl->qnlc", k, mask_new)
    attn_output_weights_mask = torch.einsum("qlc,qndc->qnld", q, k_mask)

    torch.exp(attn_output_weights_mask, out=attn_output_weights_mask)
    summed = torch.sum(attn_output_weights_mask, dim=-1, keepdim=True)
    attn_output_weights_mask /= summed
   
    attn_output_weights_mask=dropout(attn_output_weights_mask, p=dropout_p, training=training)

    attn_output = torch.einsum("qnld, qdc->qnlc", attn_output_weights_mask, v)

    attn_output = attn_output.sum(dim = 1)


    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None

def multi_head_attention_forward_vit_inf(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Tensor,
    in_proj_bias: Tensor,
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Tensor,
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
    mask_feature: Optional[Tensor] = None,
) -> Tuple[Tensor, Optional[Tensor]]:

    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    # allow MHA to have different sizes for the feature dimension
    assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5
    if not use_separate_proj_weight:
        if (query is key or torch.equal(query, key)) and (key is value or torch.equal(key, value)):
            # self-attention
            q, k, v = linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)
    q = q * scaling

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    src_len = k.size(1)
    
    
    BSZ, Mbs, Mh, Mw = mask_feature.shape
    

    Mh_new = int(Mh//4)
    Mw_new = int(Mw//4)
    mask_feature = F.interpolate(mask_feature, size=(Mh_new, Mw_new), mode = "nearest")
    mask_feature = mask_feature.to(torch.float32)
    new_mask_feature = torch.ones_like(mask_feature, dtype = k.dtype)
    
    new_mask_feature.masked_fill_(mask_feature<0, float("-inf"))
    mask_feature = new_mask_feature
    
    mask_feature = mask_feature.sigmoid()
    
    mask_flatten =mask_feature.reshape(bsz, -1, Mh_new*Mw_new)
    mask_flatten_repeat = mask_flatten.repeat(num_heads, 1,1)
    
    mask_list = []
    for i in range(bsz):
        mask_ = mask_flatten_repeat[i::bsz, :,:]
        mask_list.append(mask_)
    
    mask_new = torch.stack(mask_list)
    
    mask_new = mask_new.view(bsz*num_heads,100,-1)

    
    tmp = torch.ones(bsz*num_heads,100,1).to(mask_new)
    mask_new =torch.cat((tmp, mask_new), dim = 2)
    k_mask = torch.einsum("qlc,qnl->qnlc", k, mask_new)
    attn_output_weights_mask = torch.einsum("qlc,qndc->qnld", q, k_mask)

    torch.exp(attn_output_weights_mask, out=attn_output_weights_mask)
    summed = torch.sum(attn_output_weights_mask, dim=-1, keepdim=True)
    attn_output_weights_mask /= summed

    attn_output_weights_mask=dropout(attn_output_weights_mask, p=dropout_p, training=training)

    attn_output = torch.einsum("qnld, qdc->qnlc", attn_output_weights_mask, v)

    attn_output = attn_output.sum(dim = 1)


    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None


def multi_head_attention_forward_vit_softmax_split(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Tensor,
    in_proj_bias: Tensor,
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Tensor,
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
    mask_feature: Optional[Tensor] = None,
) -> Tuple[Tensor, Optional[Tensor]]:

    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    # allow MHA to have different sizes for the feature dimension
    assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5
    if not use_separate_proj_weight:
        if (query is key or torch.equal(query, key)) and (key is value or torch.equal(key, value)):
            # self-attention
            q, k, v = linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)
    q = q * scaling

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    src_len = k.size(1)
    
    BSZ, Mbs, Mh, Mw = mask_feature.shape

    Mh_new = int(Mh//4)
    Mw_new = int(Mw//4)
    mask_feature = F.interpolate(mask_feature, size=(Mh_new, Mw_new), mode = "nearest")
    mask_feature = mask_feature.to(torch.float32)
    
    mask_flatten =mask_feature.reshape(bsz, -1, Mh_new*Mw_new)
    mask_flatten_repeat = mask_flatten.repeat(num_heads, 1,1)
    
    mask_list = []
    for i in range(bsz):
        mask_ = mask_flatten_repeat[i::bsz, :,:]
        mask_list.append(mask_)
    
    mask_new = torch.stack(mask_list)
    
    mask_new = mask_new.view(bsz*num_heads,100,-1)

    
    tmp = torch.ones(bsz*num_heads,100,1).to(mask_new)
    mask_new =torch.cat((tmp, mask_new), dim = 2)
    
    
    # mask_new_softmax = mask_new.sigmoid()
    
    attn_output_weights = torch.bmm(q, k.transpose(-2,-1))
    
    attn_output_weights = attn_output_weights.sigmoid()

    
    
    attn_output_weights_mask = attn_output_weights
    

    attn_output = torch.einsum("qld, qdc->qlc", attn_output_weights_mask, v)


    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None

# mask直接作用于value上，没有k,v计算attention
def multi_head_attention_forward_vit_softmax_noattn(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Tensor,
    in_proj_bias: Tensor,
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Tensor,
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
    mask_feature: Optional[Tensor] = None,
    learn_token: bool=False
) -> Tuple[Tensor, Optional[Tensor]]:

    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    # allow MHA to have different sizes for the feature dimension
    assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5

    if not use_separate_proj_weight:
        if (query is key or torch.equal(query, key)) and (key is value or torch.equal(key, value)):
            # self-attention
            q, k, v = linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)
    q = q * scaling

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    src_len = k.size(1)
    
   
    BSZ, Mbs, Mh, Mw = mask_feature.shape

    Mh_new = int(Mh//4)
    Mw_new = int(Mw//4)
    mask_feature = F.interpolate(mask_feature, size=(Mh_new, Mw_new), mode="bilinear")
    mask_feature = mask_feature.to(torch.float32)

    mask_flatten =mask_feature.reshape(bsz, -1, Mh_new*Mw_new)
    mask_flatten_repeat = mask_flatten.repeat(num_heads, 1,1)
    
    mask_list = []
    for i in range(bsz):
        mask_ = mask_flatten_repeat[i::bsz, :,:]
        mask_list.append(mask_)
    
    mask_new = torch.stack(mask_list)
    
    mask_new = mask_new.view(bsz*num_heads,100,-1)

    mask_new = mask_new.sigmoid()
    
    tmp = torch.ones(bsz*num_heads,100,1).to(mask_new)
    mask_new =torch.cat((tmp, mask_new), dim = 2)
    
    mask_new_softmax = mask_new

    attn_output_weights = torch.bmm(q, k.transpose(-2,-1))
    attn_output_weights = softmax(attn_output_weights, dim=-1)
    attn_output_weights = dropout(attn_output_weights, p = dropout_p)

    attn_output_weights_mask = mask_new_softmax

    len_tokens=0
    if learn_token and attn_output_weights_mask.shape[-1] != v.shape[1]:
        len_tokens = v.shape[1] - attn_output_weights_mask.shape[-1]
        v = torch.cat((
            v[:,:1,:],
            v[:,1+len_tokens:,:]
        ),dim = 1)
        
    attn_output = torch.einsum("qnl, qlc->qnlc", attn_output_weights_mask, v)
    
        
    attn_output = attn_output.sum(dim = 1)

    assert list(attn_output.size()) == [bsz * num_heads, tgt_len-len_tokens, head_dim]
    
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len-len_tokens, bsz, embed_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads, attn_output_weights_mask
        
    else:
        return attn_output, None

# mask直接作用于value上，没有k,v计算attention，Mask-based attention
def multi_head_attention_forward_vit_softmax_noattn_maskselfattn(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Tensor,
    in_proj_bias: Tensor,
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Tensor,
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
    mask_feature: Optional[Tensor] = None,
    learn_token: bool=False,
    maskselfattnsoftmax = False
) -> Tuple[Tensor, Optional[Tensor]]:

    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    # allow MHA to have different sizes for the feature dimension
    assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5
    # import pdb; pdb.set_trace()
    if not use_separate_proj_weight:
        if (query is key or torch.equal(query, key)) and (key is value or torch.equal(key, value)):
            # self-attention
            q, k, v = linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)
    q = q * scaling

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    src_len = k.size(1)
    
    
    BSZ, Mbs, Mh, Mw = mask_feature.shape
    # down_ratio = np.sqrt(Mh*Mw/(tgt_len-1))

    Mh_new = int(Mh*4//16)
    Mw_new = int(Mw*4//16)
    mask_feature = F.interpolate(mask_feature, size=(Mh_new, Mw_new), mode="bilinear")
    mask_feature = mask_feature.to(torch.float32)

    mask_flatten =mask_feature.reshape(bsz, -1, Mh_new*Mw_new)
    mask_flatten_repeat = mask_flatten.repeat(num_heads, 1,1)
    
    mask_list = []
    for i in range(bsz):
        mask_ = mask_flatten_repeat[i::bsz, :,:]
        mask_list.append(mask_)
    
    mask_new = torch.stack(mask_list)
    
    mask_new = mask_new.view(bsz*num_heads,100,-1)

    mask_new = mask_new.sigmoid()
    
    tmp = torch.ones(bsz*num_heads,100,1).to(mask_new)
    mask_new =torch.cat((tmp, mask_new), dim = 2)
    
    mask_new_softmax = mask_new
    
    q_mask = mask_new.permute(0, 2, 1)
    
    if maskselfattnsoftmax:
        k_mask = q_mask.clone()#.softmax(dim = -1)
    else:
        k_mask = q_mask.clone()
    
    attn_output_weights = torch.bmm(q_mask, k_mask.transpose(-2,-1))
    
    attn_output_weights = softmax(attn_output_weights, dim=-1)
    
    attn_output_weights = dropout(attn_output_weights, p = dropout_p)

    
    attn_output_weights_mask = mask_new_softmax
    
    
    len_tokens=0
    if learn_token and attn_output_weights_mask.shape[-1] != v.shape[1]:
        len_tokens = v.shape[1] - attn_output_weights_mask.shape[-1]
        v = torch.cat((
            v[:,:1,:],
            v[:,1+len_tokens:,:]
        ),dim = 1)

    attn_output=torch.bmm(attn_output_weights, v)
    
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len-len_tokens, head_dim]
    
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len-len_tokens, bsz, embed_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads, attn_output_weights
    else:
        return attn_output, None

# mask直接作用于value上，没有k,v计算attention
def multi_head_attention_forward_vit_softmax_noattn_maskselfattn_addselfattn(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Tensor,
    in_proj_bias: Tensor,
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Tensor,
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
    mask_feature: Optional[Tensor] = None,
    learn_token: bool=False,
    maskselfattnsoftmax = False, 
    layerVit = -1
) -> Tuple[Tensor, Optional[Tensor]]:

    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    # allow MHA to have different sizes for the feature dimension
    assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5
    # import pdb; pdb.set_trace()
    if not use_separate_proj_weight:
        if (query is key or torch.equal(query, key)) and (key is value or torch.equal(key, value)):
            # self-attention
            q, k, v = linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)
    
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    
    
    
    bias_attn_1= torch.eye(tgt_len, tgt_len).to(v)
    bias_attn = bias_attn_1.unsqueeze(0).repeat([bsz*num_heads,1,1])

    
    alpha = 0
    beta = 1.0 - alpha
    alpha_MSA = 1.0
    if layerVit > -1:
        
        # if layerVit == 10:
        #     alpha = 1.0
        #     beta = 1.0-alpha
        if layerVit == 11:
            alpha_MSA = 1.0
    
    # attn_output_weights = attn_output_weights_qk * alpha + bias_attn * beta
    attn_output_weights =  bias_attn
    
        
    len_tokens=0
    

    attn_output=torch.bmm(attn_output_weights, v)
    

    assert list(attn_output.size()) == [bsz * num_heads, tgt_len-len_tokens, head_dim]
    
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len-len_tokens, bsz, embed_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

    attn_output = attn_output * alpha_MSA

    if need_weights:
        # average attention weights over heads
    
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads, attn_output_weights
    
    else:
        return attn_output, None
# mask直接作用于value上，没有k,v计算attention vitL
def multi_head_attention_forward_vit_softmax_noattn_vitL(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Tensor,
    in_proj_bias: Tensor,
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Tensor,
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
    mask_feature: Optional[Tensor] = None,
    learn_token: bool=False
) -> Tuple[Tensor, Optional[Tensor]]:

    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    # allow MHA to have different sizes for the feature dimension
    assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5
    # import pdb; pdb.set_trace()
    if not use_separate_proj_weight:
        if (query is key or torch.equal(query, key)) and (key is value or torch.equal(key, value)):
            # self-attention
            q, k, v = linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)
    q = q * scaling

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    src_len = k.size(1)
    
    #####################################################################################################################
    # 处理mask feature的尺寸
    BSZ, Mbs, Mh, Mw = mask_feature.shape
    # down_ratio = np.sqrt(Mh*Mw/(tgt_len-1))

    Mh_new = int(Mh*4//14)
    Mw_new = int(Mw*4//14)
    mask_feature = F.interpolate(mask_feature, size=(Mh_new, Mw_new), mode="bilinear")
    mask_feature = mask_feature.to(torch.float32)

    

    # new_mask_feature = torch.ones_like(mask_feature, dtype = k.dtype)
    # new_mask_feature.masked_fill_(mask_feature<0, float("-inf"))    
    # mask_feature = new_mask_feature


    mask_flatten =mask_feature.reshape(bsz, -1, Mh_new*Mw_new)
    mask_flatten_repeat = mask_flatten.repeat(num_heads, 1,1)
    
    mask_list = []
    for i in range(bsz):
        mask_ = mask_flatten_repeat[i::bsz, :,:]
        mask_list.append(mask_)
    
    mask_new = torch.stack(mask_list)
    
    mask_new = mask_new.view(bsz*num_heads,100,-1)

    mask_new = mask_new.sigmoid()
    # mask_new = softmax(mask_new, dim = -1)
    tmp = torch.ones(bsz*num_heads,100,1).to(mask_new)
    mask_new =torch.cat((tmp, mask_new), dim = 2)
    
    # mask_new_softmax = softmax(mask_new, dim = -1)
    
    mask_new_softmax = mask_new
    ####################################################################################
    attn_output_weights = torch.bmm(q, k.transpose(-2,-1))
    attn_output_weights = softmax(attn_output_weights, dim=-1)
    attn_output_weights = dropout(attn_output_weights, p = dropout_p)
    # return attn_output_weights, None
    # attn_output_weights = attn_output_weights.sigmoid()

    # import pdb; pdb.set_trace()
    # attn_output_weights_mask = torch.einsum("qld, qnd->qnld", attn_output_weights, mask_new_softmax)
    
    # attn_output_weights_mask = attn_output_weights
    ####################################################################################
    attn_output_weights_mask = mask_new_softmax
    # import pdb; pdb.set_trace()
    if attn_output_weights_mask.isnan().any():
        print("attn_output_weights_mask is nan")
        # import pdb; pdb.set_trace()
    if v.isnan().any():
        print("v is nan")
        # import pdb; pdb.set_trace()
    len_tokens=0
    if learn_token and attn_output_weights_mask.shape[-1] != v.shape[1]:
        len_tokens = v.shape[1] - attn_output_weights_mask.shape[-1]
        v = torch.cat((
            v[:,:1,:],
            v[:,1+len_tokens:,:]
        ),dim = 1)
        # import pdb; pdb.set_trace()
        # v = torch.cat()
    attn_output = torch.einsum("qnl, qlc->qnlc", attn_output_weights_mask, v)
    if attn_output.isnan().any():
        print("attn_output is nan")
        # import pdb; pdb.set_trace()
    attn_output = attn_output.sum(dim = 1)
    # attn_output = v
    

    # attn_output = torch.einsum("qld, qdc->qlc", attn_output_weights_mask, v)
    # attn_output_weights = attn_output_weights_mask


    assert list(attn_output.size()) == [bsz * num_heads, tgt_len-len_tokens, head_dim]
    # import pdb; pdb.set_trace()
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len-len_tokens, bsz, embed_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        # average attention weights over heads
        # import pdb; pdb.set_trace()
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads, attn_output_weights_mask
        # return attn_output, attn_output_weights
    else:
        return attn_output, None


# mask直接作用于value上，没有k,v计算attention
def multi_head_attention_forward_vit_softmax_noattn_delall(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Tensor,
    in_proj_bias: Tensor,
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Tensor,
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
    mask_feature: Optional[Tensor] = None,
    learn_token: bool=False
) -> Tuple[Tensor, Optional[Tensor]]:

    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    # allow MHA to have different sizes for the feature dimension
    assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5
    if not use_separate_proj_weight:
        if (query is key or torch.equal(query, key)) and (key is value or torch.equal(key, value)):
            # self-attention
            q, k, v = linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)
    q = q * scaling

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    src_len = k.size(1)
    
    #####################################################################################################################
    # 处理mask feature的尺寸
    BSZ, Mbs, Mh, Mw = mask_feature.shape
    # down_ratio = np.sqrt(Mh*Mw/(tgt_len-1))

    Mh_new = int(Mh//4)
    Mw_new = int(Mw//4)
    mask_feature = F.interpolate(mask_feature, size=(Mh_new, Mw_new), mode="bilinear")
    mask_feature = mask_feature.to(torch.float32)

    

    # new_mask_feature = torch.ones_like(mask_feature, dtype = k.dtype)
    # new_mask_feature.masked_fill_(mask_feature<0, float("-inf"))    
    # mask_feature = new_mask_feature


    mask_flatten =mask_feature.reshape(bsz, -1, Mh_new*Mw_new)
    mask_flatten_repeat = mask_flatten.repeat(num_heads, 1,1)
    
    mask_list = []
    for i in range(bsz):
        mask_ = mask_flatten_repeat[i::bsz, :,:]
        mask_list.append(mask_)
    
    mask_new = torch.stack(mask_list)
    
    mask_new = mask_new.view(bsz*num_heads,100,-1)

    mask_new = mask_new.sigmoid()
    # mask_new = softmax(mask_new, dim = -1)
    tmp = torch.ones(bsz*num_heads,100,1).to(mask_new)
    mask_new =torch.cat((tmp, mask_new), dim = 2)
    
    # mask_new_softmax = softmax(mask_new, dim = -1)
    # import pdb; pdb.set_trace()
    mask_new_softmax = mask_new
    ####################################################################################
    # attn_output_weights = torch.bmm(q, k.transpose(-2,-1))
    # attn_output_weights = softmax(attn_output_weights, dim=-1)
    # attn_output_weights = dropout(attn_output_weights, p = dropout_p)
    # return attn_output_weights, None
    # attn_output_weights = attn_output_weights.sigmoid()

    # import pdb; pdb.set_trace()
    # attn_output_weights_mask = torch.einsum("qld, qnd->qnld", attn_output_weights, mask_new_softmax)
    
    # attn_output_weights_mask = attn_output_weights
    ####################################################################################
    attn_output_weights_mask = mask_new_softmax
    # import pdb; pdb.set_trace()
    if attn_output_weights_mask.isnan().any():
        print("attn_output_weights_mask is nan")
        # import pdb; pdb.set_trace()
    if v.isnan().any():
        print("v is nan")
        # import pdb; pdb.set_trace()
    len_tokens=0
    # import pdb; pdb.set_trace()
    if learn_token and attn_output_weights_mask.shape[-1] != v.shape[1]:
        len_tokens = v.shape[1] - attn_output_weights_mask.shape[-1]
        v = torch.cat((
            v[:,:1,:],
            v[:,1+len_tokens:,:]
        ),dim = 1)
        # import pdb; pdb.set_trace()
        # v = torch.cat()
    # attn_output = torch.einsum("qnl, qlc->qnlc", attn_output_weights_mask, v)
    # if attn_output.isnan().any():
    #     print("attn_output is nan")
    #     # import pdb; pdb.set_trace()
    # attn_output = attn_output.sum(dim = 1)
    attn_output = v
    

    # attn_output = torch.einsum("qld, qdc->qlc", attn_output_weights_mask, v)


    assert list(attn_output.size()) == [bsz * num_heads, tgt_len-len_tokens, head_dim]
    # import pdb; pdb.set_trace()
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len-len_tokens, bsz, embed_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None


# mask直接作用于value上，没有k,v计算attention;每张图分割为4部分 直接返回value， 后面再拼接
def multi_head_attention_forward_vit_softmax_noattn_plain(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Tensor,
    in_proj_bias: Tensor,
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Tensor,
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
    mask_feature: Optional[Tensor] = None,
) -> Tuple[Tensor, Optional[Tensor]]:

    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    # allow MHA to have different sizes for the feature dimension
    assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5
    if not use_separate_proj_weight:
        if (query is key or torch.equal(query, key)) and (key is value or torch.equal(key, value)):
            # self-attention
            q, k, v = linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)
    q = q * scaling

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    src_len = k.size(1)
    
    #####################################################################################################################
    # 处理mask feature的尺寸
    BSZ, Mbs, Mh, Mw = mask_feature.shape
    # down_ratio = np.sqrt(Mh*Mw/(tgt_len-1))

    Mh_new = int(Mh//4)
    Mw_new = int(Mw//4)
    mask_feature = F.interpolate(mask_feature, size=(Mh_new, Mw_new), mode="bilinear")
    mask_feature = mask_feature.to(torch.float32)

    

    # new_mask_feature = torch.ones_like(mask_feature, dtype = k.dtype)
    # new_mask_feature.masked_fill_(mask_feature<0, float("-inf"))    
    # mask_feature = new_mask_feature


    mask_flatten =mask_feature.reshape(bsz, -1, Mh_new*Mw_new)
    mask_flatten_repeat = mask_flatten.repeat(num_heads, 1,1)
    
    mask_list = []
    for i in range(bsz):
        mask_ = mask_flatten_repeat[i::bsz, :,:]
        mask_list.append(mask_)
    
    mask_new = torch.stack(mask_list)
    
    mask_new = mask_new.view(bsz*num_heads,100,-1)

    mask_new = mask_new.sigmoid()
    # mask_new = softmax(mask_new, dim = -1)
    tmp = torch.ones(bsz*num_heads,100,1).to(mask_new)
    mask_new =torch.cat((tmp, mask_new), dim = 2)
    
    # mask_new_softmax = softmax(mask_new, dim = -1)
    # import pdb; pdb.set_trace()
    mask_new_softmax = mask_new
    ####################################################################################
    # attn_output_weights = torch.bmm(q, k.transpose(-2,-1))
    # attn_output_weights = softmax(attn_output_weights, dim=-1)
    # attn_output_weights = dropout(attn_output_weights, p = dropout_p)
    # return attn_output_weights, None
    # attn_output_weights = attn_output_weights.sigmoid()

    # import pdb; pdb.set_trace()
    # attn_output_weights_mask = torch.einsum("qld, qnd->qnld", attn_output_weights, mask_new_softmax)
    
    # attn_output_weights_mask = attn_output_weights
    ####################################################################################
    attn_output_weights_mask = mask_new_softmax
    # 2022-12-24修改
    # attn_output = torch.einsum("qnl, qlc->qnlc", attn_output_weights_mask, v)

    # attn_output = attn_output.sum(dim = 1)
    attn_output = v
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None


# mask直接作用于value上，没有k,v计算attention, 最后特征为BNLC，N维度不做sum
def multi_head_attention_forward_vit_softmax_noattn_nosum(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Tensor,
    in_proj_bias: Tensor,
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Tensor,
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
    mask_feature: Optional[Tensor] = None,
) -> Tuple[Tensor, Optional[Tensor]]:

    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    # allow MHA to have different sizes for the feature dimension
    assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5
    if not use_separate_proj_weight:
        if (query is key or torch.equal(query, key)) and (key is value or torch.equal(key, value)):
            # self-attention
            q, k, v = linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)
    q = q * scaling

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    src_len = k.size(1)
    
    #####################################################################################################################
    # 处理mask feature的尺寸
    BSZ, Mbs, Mh, Mw = mask_feature.shape
    # down_ratio = np.sqrt(Mh*Mw/(tgt_len-1))

    Mh_new = int(Mh//4)
    Mw_new = int(Mw//4)
    mask_feature = F.interpolate(mask_feature, size=(Mh_new, Mw_new), mode = "nearest")
    mask_feature = mask_feature.to(torch.float32)

    

    # new_mask_feature = torch.ones_like(mask_feature, dtype = k.dtype)
    # new_mask_feature.masked_fill_(mask_feature<0, float("-inf"))    
    # mask_feature = new_mask_feature


    mask_flatten =mask_feature.reshape(bsz, -1, Mh_new*Mw_new)
    mask_flatten_repeat = mask_flatten.repeat(num_heads, 1,1)
    
    mask_list = []
    for i in range(bsz):
        mask_ = mask_flatten_repeat[i::bsz, :,:]
        mask_list.append(mask_)
    
    mask_new = torch.stack(mask_list)
    
    mask_new = mask_new.view(bsz*num_heads,100,-1)

    mask_new = mask_new.sigmoid()
    
    tmp = torch.ones(bsz*num_heads,100,1).to(mask_new)
    mask_new =torch.cat((tmp, mask_new), dim = 2)
    
    
    mask_new_softmax = mask_new
   
    attn_output_weights_mask = mask_new_softmax
    attn_output = torch.einsum("qnl, qlc->qnlc", attn_output_weights_mask, v)
    
    # attn_output = attn_output.sum(dim = 1)
    
    assert list(attn_output.size()) == [bsz * num_heads,Mbs, tgt_len, head_dim]

    attn_output = attn_output.permute(1,2,0,3).contiguous().view(Mbs, tgt_len,bsz,embed_dim)
    # attn_output = attn_output.permute(0,2,1,3).contiguous().view(tgt_len, bsz, Mbs, embed_dim)

    # attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None

# 没有mask feature参与计算，也不使用kv计算attention，只保留FFN等  
def multi_head_attention_forward_vit_noattn(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Tensor,
    in_proj_bias: Tensor,
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Tensor,
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
    mask_feature: Optional[Tensor] = None,
) -> Tuple[Tensor, Optional[Tensor]]:

    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    # allow MHA to have different sizes for the feature dimension
    assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5
    if not use_separate_proj_weight:
        if (query is key or torch.equal(query, key)) and (key is value or torch.equal(key, value)):
            # self-attention
            q, k, v = linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)
    q = q * scaling

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    src_len = k.size(1)
    
    #####################################################################################################################
    # 处理mask feature的尺寸
    BSZ, Mbs, Mh, Mw = mask_feature.shape
    # down_ratio = np.sqrt(Mh*Mw/(tgt_len-1))

    Mh_new = int(Mh//4)
    Mw_new = int(Mw//4)
    mask_feature = F.interpolate(mask_feature, size=(Mh_new, Mw_new), mode = "nearest")
    mask_feature = mask_feature.to(torch.float32)

    ######################################################################
    # new_mask_feature = torch.ones_like(mask_feature, dtype = k.dtype)
    
    # new_mask_feature.masked_fill_(mask_feature<0, float("-inf"))
    # mask_feature = new_mask_feature
    ######################################################################
    # import pdb; pdb.set_trace()
    # mask_feature = mask_feature.sigmoid()

    # new_mask_feature = torch.ones_like(mask_feature, dtype = k.dtype)
    # new_mask_feature.masked_fill_(mask_feature<0, float("-inf"))    
    # mask_feature = new_mask_feature


    # mask_flatten =mask_feature.reshape(bsz, -1, Mh_new*Mw_new)
    # mask_flatten_repeat = mask_flatten.repeat(num_heads, 1,1)
    
    # mask_list = []
    # for i in range(bsz):
    #     mask_ = mask_flatten_repeat[i::bsz, :,:]
    #     mask_list.append(mask_)
    
    # mask_new = torch.stack(mask_list)
    
    # mask_new = mask_new.view(bsz*num_heads,100,-1)

    # mask_new = mask_new.sigmoid()
    
    # tmp = torch.ones(bsz*num_heads,100,1).to(mask_new)
    # mask_new =torch.cat((tmp, mask_new), dim = 2)
    
    # mask_new_softmax = mask_new

    # attn_output_weights_mask = mask_new_softmax
    # attn_output = torch.einsum("qnl, qlc->qnlc", attn_output_weights_mask, v)

    # attn_output = attn_output.sum(dim = 1)
    attn_output = v

    # attn_output = torch.einsum("qld, qdc->qlc", attn_output_weights_mask, v)

    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None
class Linear(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
# This class exists solely for Transformer; it has an annotation stating
# that bias is never None, which appeases TorchScript
class _LinearWithBias(Linear):
    bias: Tensor  # type: ignore

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__(in_features, out_features, bias=True)  # type: ignore