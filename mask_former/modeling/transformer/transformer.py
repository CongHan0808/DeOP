# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/transformer.py
"""
Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
import math

import numpy as np


class Transformer(nn.Module):
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
    ):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )

        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
        )

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        # import pdb; pdb.set_trace()
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        if mask is not None:
            mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        # import pdb; pdb.set_trace()
        hs = self.decoder(
            tgt,
            memory,
            memory_key_padding_mask=mask,
            pos=pos_embed,
            query_pos=query_embed,
        )
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        output = src

        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                pos=pos,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(
            q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(
            q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        # 2022-11-16 修改,增加成员变量d_model和dim_feedforward
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(
            q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt,
                memory,
                tgt_mask,
                memory_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask,
                pos,
                query_pos,
            )
        return self.forward_post(
            tgt,
            memory,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            pos,
            query_pos,
        )



# decoder去除linear部分和TransformerDecoderLayerLinear配合使用
class TransformerDecoderLinear(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)

        d_model = decoder_layer.d_model
        nhead = 8
        dim_feedforward = decoder_layer.dim_feedforward
        dropout = 0.1
        activation = "relu"
        normalize_before = False
        # decoder_layer = TransformerDecoderLayerLinear(
        #     d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        # )
        self.outlayer = TransformerDecoderOutLayerLinear(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        proposal_mask: Optional[Tensor] = None,
        need_weights = False,
    ):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
                proposal_mask=proposal_mask,
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output))
        outputl=self.outlayer(
            output,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            pos=pos,
            query_pos=query_pos,
            proposal_mask=proposal_mask,
            need_weights = need_weights,
        )
        if need_weights:
            return outputl
        if self.return_intermediate:
            intermediate.append(outputl)
        # import pdb; pdb.set_trace()

        # if self.norm is not None:
        #     output = self.norm(output)
        #     if self.return_intermediate:
        #         intermediate.pop()
        #         intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return outputl.unsqueeze(0)

# decoder去除linear部分和TransformerDecoderLayerLinear配合使用, 参照Fusioner，在transformer层增加和text的融合
class TransformerDecoderLinearTextFuse(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)

        d_model = decoder_layer.d_model
        nhead = 8
        dim_feedforward = decoder_layer.dim_feedforward
        dropout = 0.1
        activation = "relu"
        normalize_before = False
        # decoder_layer = TransformerDecoderLayerLinear(
        #     d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        # )
        self.outlayer = TransformerDecoderOutLayerLinear(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        proposal_mask: Optional[Tensor] = None,
        need_weights = False,
    ):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
                proposal_mask=proposal_mask,
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output))
        outputl=self.outlayer(
            output,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            pos=pos,
            query_pos=query_pos,
            proposal_mask=proposal_mask,
            need_weights = need_weights,
        )
        if need_weights:
            return outputl
        if self.return_intermediate:
            intermediate.append(outputl)
        # import pdb; pdb.set_trace()

        # if self.norm is not None:
        #     output = self.norm(output)
        #     if self.return_intermediate:
        #         intermediate.pop()
        #         intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return outputl.unsqueeze(0)
# decoder去除linear部分和TransformerDecoderLayerLinear配合使用
class TransformerDecoderLinearSrcTransformer(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)

        d_model = decoder_layer.d_model
        nhead = 8
        dim_feedforward = decoder_layer.dim_feedforward
        dropout = 0.1
        activation = "relu"
        normalize_before = False
        # decoder_layer = TransformerDecoderLayerLinear(
        #     d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        # )
        # self.outlayer = TransformerDecoderLayerLinear()
        self.outlayer = TransformerDecoderLayerLinear(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        proposal_mask: Optional[Tensor] = None,
    ):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
                proposal_mask=proposal_mask,
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output))
        outputl=self.outlayer(
            output,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            pos=pos,
            query_pos=query_pos,
            proposal_mask=proposal_mask,
        )
        if self.return_intermediate:
            intermediate.append(outputl)
        # import pdb; pdb.set_trace()

        # if self.norm is not None:
        #     output = self.norm(output)
        #     if self.return_intermediate:
        #         intermediate.pop()
        #         intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return outputl.unsqueeze(0)


# 去除 对value的非线性部分
class TransformerDecoderLayerLinear(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        train_flag = True,
        mask_feature = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.mask_feature = mask_feature
        if train_flag:
            # dropout = 0
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            # self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

            #multihead atten implement by myself
            if self.mask_feature:
                self.multihead_attn_my = MultiHeadAttentionMaskFeatMy(d_model, nhead, dropout=dropout)
            else:
                self.multihead_attn_my=nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            # Implementation of Feedforward model
            self.linear1 = nn.Linear(d_model, dim_feedforward)
            self.dropout = nn.Dropout(dropout)
            self.linear2 = nn.Linear(dim_feedforward, d_model)

            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.norm3 = nn.LayerNorm(d_model)
            self.dropout1 = nn.Dropout(dropout)
            self.dropout2 = nn.Dropout(dropout)
            self.dropout3 = nn.Dropout(dropout)

            self.activation = _get_activation_fn(activation)
            self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        proposal_mask: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(tgt, query_pos)
        
        tgt2 = self.self_attn(
            q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        if self.mask_feature:
            tgt2= self.multihead_attn_my(
                query=self.with_pos_embed(tgt, query_pos),
                key=self.with_pos_embed(memory, pos),
                value=memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
                proposal_mask = proposal_mask,
            )[0]
            pass
        else:
            tgt2= self.multihead_attn_my(
                query=self.with_pos_embed(tgt, query_pos),
                key=self.with_pos_embed(memory, pos),
                value=memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
                # proposal_mask = proposal_mask,
            )[0]
        # import pdb; pdb.set_trace()
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    
    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        proposal_mask: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt,
                memory,
                tgt_mask,
                memory_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask,
                pos,
                query_pos,
                
            )
        return self.forward_post(
            tgt,
            memory,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            pos,
            query_pos,
            proposal_mask,
        )


# 去除 output decoder layer
class TransformerDecoderOutLayerLinear(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        # dropout = 0
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        #multihead atten implement by myself
        self.multihead_attn_my = MultiHeadAttentionMy(d_model, nhead, dropout=dropout, outDecoderLayer = True)
        
        # self.multihead_attn_my = MultiHeadAttentionMyNoattn(d_model, nhead, dropout=dropout, outDecoderLayer = True)
        
        # self.multihead_attn_my = MultiHeadAttentionMyChannelAtten(d_model, nhead, dropout=dropout, outDecoderLayer = True)
        # MultiHeadAttentionMyChannelAtten
        # Implementation of Feedforward model
        # self.linear1 = nn.Linear(d_model, dim_feedforward)
        # self.dropout = nn.Dropout(dropout)
        # self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        # self.norm2 = nn.LayerNorm(d_model)
        # self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        proposal_mask: Optional[Tensor] = None,
        need_weights = False,
    ):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # tgt2 = self.multihead_attn(
        #     query=self.with_pos_embed(tgt, query_pos),
        #     key=self.with_pos_embed(memory, pos),
        #     value=memory,
        #     attn_mask=memory_mask,
        #     key_padding_mask=memory_key_padding_mask,
        # )[0]
        if need_weights:
            tgt2= self.multihead_attn_my(
                query=self.with_pos_embed(tgt, query_pos),
                key=self.with_pos_embed(memory, pos),
                value=memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
                proposal_mask = proposal_mask,
            )[1]
            return tgt2
        else:
            tgt2= self.multihead_attn_my(
                query=self.with_pos_embed(tgt, query_pos),
                key=self.with_pos_embed(memory, pos),
                value=memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
                proposal_mask = proposal_mask,
            )[0]
        
        # self.non_object_embedding.norm(dim=-1, keepdim=True)
        if (tgt2.norm(p=2,dim =-1, keepdim=True)==0).any():
            tgt = tgt2/ (tgt2.norm(p=2,dim=-1, keepdim=True) + 1e-10)
            pass
        else:
            tgt = tgt2/ tgt2.norm(p=2,dim=-1, keepdim=True)
        if torch.isinf(tgt).any():
            print("tgt is inf line649")
            import pdb; pdb.set_trace()
        return tgt

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        proposal_mask: Optional[Tensor] = None,
        need_weights = False,
    ):
        return self.forward_post(
            tgt,
            memory,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            pos,
            query_pos,
            proposal_mask,
            need_weights,
        )
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")

class MultiHeadAttentionMy(nn.Module):
    def __init__(self,embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, \
            kdim=None, vdim=None, batch_first=False, device=None, dtype=None, multiheadAtten = None, outDecoderLayer = False):
        super(MultiHeadAttentionMy, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout_p = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads

        self.outDecoderLayer = outDecoderLayer

        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.register_parameter('q_proj_weight', None)
        self.register_parameter('k_proj_weight', None)
        self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        # self.out_proj = _LinearWithBias(embed_dim, embed_dim)

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = nn.Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None
        
        self.add_zero_attn = add_zero_attn

        

        if multiheadAtten is not None:
            self._reset_parameters_(multiheadAtten)
            pass
        else:
            self._reset_parameters()

        # pytorch 1.12
        # self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        # self.bias_k = self.bias_v = None
        
        # self.out_proj = _LinearWithBias(embed_dim, embed_dim)


        # self.add_zero_attn = add_zero_attn

        # self._reset_parameters()
        pass
    def _reset_parameters_(self, multiheadAtten):
        if self._qkv_same_embed_dim:
            nn.init.xavier_uniform_(self.in_proj_weight)
            self.in_proj_weight = multiheadAtten.in_proj_weight
        else:
            nn.init.xavier_uniform_(self.q_proj_weight)
            nn.init.xavier_uniform_(self.k_proj_weight)
            nn.init.xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

        self.out_proj.weight=multiheadAtten.out_proj.weight
    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            nn.init.xavier_uniform_(self.in_proj_weight)
        else:
            nn.init.xavier_uniform_(self.q_proj_weight)
            nn.init.xavier_uniform_(self.k_proj_weight)
            nn.init.xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            # nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)
    # def forward(self, query: Tensor, key: Tensor, value: Tensor,in_proj_bias: Tensor,key_padding_mask: Optional[Tensor] = None,need_weights: bool = True, attn_mask: Optional[Tensor] = None,average_attn_weights: bool = True,):
    def forward(self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
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
            proposal_mask: Optional[Tensor] = None,
        ):

        # set up shape vars
        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape

        if isinstance(embed_dim, torch.Tensor):
            # embed_dim can be a tensor when JIT tracing
            head_dim = embed_dim.div(self.num_heads, rounding_mode='trunc')
        else:
            head_dim = embed_dim // self.num_heads
        scaling = float(head_dim) ** -0.5

        # This is inline in_proj function with in_proj_weight and in_proj_bias
        _b = self.in_proj_bias
        _start = 0
        _end = embed_dim
        _w = self.in_proj_weight[_start:_end, :]
        if _b is not None:
            _b = _b[_start:_end]
        q = linear(query, _w, _b)

        # This is inline in_proj function with in_proj_weight and in_proj_bias
        _b = self.in_proj_bias
        _start = embed_dim
        _end = embed_dim * 2
        _w = self.in_proj_weight[_start:_end, :]
        if _b is not None:
            _b = _b[_start:_end]
        k = linear(key, _w, _b)

        # This is inline in_proj function with in_proj_weight and in_proj_bias
        _b = self.in_proj_bias
        _start = embed_dim * 2
        _end = None
        _w = self.in_proj_weight[_start:, :]
        if _b is not None:
            _b = _b[_start:]

        if not self.outDecoderLayer:
            v = linear(value, _w, _b)
        else:
            # import pdb; pdb.set_trace()
            v = value

        q = q*scaling
        
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)

        src_len = k.size(1)

        attn_output_weights = torch.bmm(q, k.transpose(1, 2))

        # proposal mask 处理
        mask_shape = proposal_mask.shape
        proposal_mask = proposal_mask.view(mask_shape[0], mask_shape[1],-1)
        proposal_mask_nhead = proposal_mask.repeat(self.num_heads, 1,1)
        mask_list = []
        for i in range(bsz):
            mask_head = proposal_mask_nhead[i::bsz, :,:]
            mask_list.append(mask_head)
        mask_list_stack = torch.stack(mask_list)
        mask_stack_shape = mask_list_stack.shape
        mask_nhead = mask_list_stack.view(mask_stack_shape[0] * mask_stack_shape[1], \
            mask_stack_shape[2], mask_stack_shape[3])

        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = F.dropout(attn_output_weights, p=self.dropout_p, training=training)
        
        # soft(k*v^T/sqrt(d))后添加mask
        attn_output_weights = attn_output_weights * mask_nhead
        # import pdb; pdb.set_trace()
        is_inf_attn = torch.isinf(attn_output_weights).any()
        if is_inf_attn:
            import pdb; pdb.set_trace()
            print("attn_output_weight line861 is inf")
            pass

        attn_output = torch.bmm(attn_output_weights, v)

        is_inf_output = torch.isinf(attn_output).any()
        if is_inf_output:
            import pdb; pdb.set_trace()
            print("attn_output line869 is inf")
            pass

        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        # import pdb; pdb.set_trace()
        if not self.outDecoderLayer:
            attn_output = linear(attn_output, self.out_proj.weight, self.out_proj.bias)
            

        if need_weights:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
            return attn_output, [attn_output_weights.sum(dim=1) / self.num_heads,attn_output]
        else:
            return attn_output, None

class MultiHeadAttentionMaskFeatMy(nn.Module):
    def __init__(self,embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, \
            kdim=None, vdim=None, batch_first=False, device=None, dtype=None, multiheadAtten = None, outDecoderLayer = False):
        super(MultiHeadAttentionMaskFeatMy, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout_p = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads

        self.outDecoderLayer = outDecoderLayer

        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.register_parameter('q_proj_weight', None)
        self.register_parameter('k_proj_weight', None)
        self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = _LinearWithBias(embed_dim, embed_dim)

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = nn.Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None
        
        self.add_zero_attn = add_zero_attn

        

        if multiheadAtten is not None:
            self._reset_parameters_(multiheadAtten)
            pass
        else:
            self._reset_parameters()

        # pytorch 1.12
        # self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        # self.bias_k = self.bias_v = None
        
        # self.out_proj = _LinearWithBias(embed_dim, embed_dim)


        # self.add_zero_attn = add_zero_attn

        # self._reset_parameters()
        pass
    def _reset_parameters_(self, multiheadAtten):
        if self._qkv_same_embed_dim:
            nn.init.xavier_uniform_(self.in_proj_weight)
            self.in_proj_weight = multiheadAtten.in_proj_weight
        else:
            nn.init.xavier_uniform_(self.q_proj_weight)
            nn.init.xavier_uniform_(self.k_proj_weight)
            nn.init.xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

        self.out_proj.weight=multiheadAtten.out_proj.weight
    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            nn.init.xavier_uniform_(self.in_proj_weight)
        else:
            nn.init.xavier_uniform_(self.q_proj_weight)
            nn.init.xavier_uniform_(self.k_proj_weight)
            nn.init.xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)
    # def forward(self, query: Tensor, key: Tensor, value: Tensor,in_proj_bias: Tensor,key_padding_mask: Optional[Tensor] = None,need_weights: bool = True, attn_mask: Optional[Tensor] = None,average_attn_weights: bool = True,):
    def forward(self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
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
            proposal_mask: Optional[Tensor] = None,
        ):

        # set up shape vars
        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape

        if isinstance(embed_dim, torch.Tensor):
            # embed_dim can be a tensor when JIT tracing
            head_dim = embed_dim.div(self.num_heads, rounding_mode='trunc')
        else:
            head_dim = embed_dim // self.num_heads
        scaling = float(head_dim) ** -0.5

        # This is inline in_proj function with in_proj_weight and in_proj_bias
        _b = self.in_proj_bias
        _start = 0
        _end = embed_dim
        _w = self.in_proj_weight[_start:_end, :]
        if _b is not None:
            _b = _b[_start:_end]
        q = linear(query, _w, _b)

        # This is inline in_proj function with in_proj_weight and in_proj_bias
        _b = self.in_proj_bias
        _start = embed_dim
        _end = embed_dim * 2
        _w = self.in_proj_weight[_start:_end, :]
        if _b is not None:
            _b = _b[_start:_end]
        k = linear(key, _w, _b)

        # This is inline in_proj function with in_proj_weight and in_proj_bias
        _b = self.in_proj_bias
        _start = embed_dim * 2
        _end = None
        _w = self.in_proj_weight[_start:, :]
        if _b is not None:
            _b = _b[_start:]

        if not self.outDecoderLayer:
            v = linear(value, _w, _b)
        else:
            # import pdb; pdb.set_trace()
            v = value

        q = q*scaling
        
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)

        src_len = k.size(1)

        attn_output_weights = torch.bmm(q, k.transpose(1, 2))

        # proposal mask 处理
        mask_shape = proposal_mask.shape
        proposal_mask = proposal_mask.view(mask_shape[0], mask_shape[1],-1)
        proposal_mask_nhead = proposal_mask.repeat(self.num_heads, 1,1)
        mask_list = []
        for i in range(bsz):
            mask_head = proposal_mask_nhead[i::bsz, :,:]
            mask_list.append(mask_head)
        mask_list_stack = torch.stack(mask_list)
        mask_stack_shape = mask_list_stack.shape
        mask_nhead = mask_list_stack.view(mask_stack_shape[0] * mask_stack_shape[1], \
            mask_stack_shape[2], mask_stack_shape[3])

        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = F.dropout(attn_output_weights, p=self.dropout_p, training=training)
        
        # soft(k*v^T/sqrt(d))后添加mask
        attn_output_weights = attn_output_weights * mask_nhead
        # import pdb; pdb.set_trace()
        is_inf_attn = torch.isinf(attn_output_weights).any()
        if is_inf_attn:
            import pdb; pdb.set_trace()
            print("attn_output_weight line861 is inf")
            pass

        attn_output = torch.bmm(attn_output_weights, v)

        is_inf_output = torch.isinf(attn_output).any()
        if is_inf_output:
            import pdb; pdb.set_trace()
            print("attn_output line869 is inf")
            pass

        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        # import pdb; pdb.set_trace()
        if not self.outDecoderLayer:
            attn_output = linear(attn_output, self.out_proj.weight, self.out_proj.bias)
            

        if need_weights:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
            return attn_output, attn_output_weights.sum(dim=1) / self.num_heads
        else:
            return attn_output, None


# 2023-01-24 decoderoutput no atten
class MultiHeadAttentionMyNoattn(nn.Module):
    def __init__(self,embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, \
            kdim=None, vdim=None, batch_first=False, device=None, dtype=None, multiheadAtten = None, outDecoderLayer = False):
        super(MultiHeadAttentionMyNoattn, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout_p = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads

        self.outDecoderLayer = outDecoderLayer

        # self.in_proj_weight = nn.Parameter(torch.empty(embed_dim, embed_dim))
        # self.register_parameter('q_proj_weight', None)
        # self.register_parameter('k_proj_weight', None)
        # self.register_parameter('v_proj_weight', None)

        if bias:
            # self.in_proj_bias = nn.Parameter(torch.empty(embed_dim))
            self.in_proj_bias = None
            pass
        else:
            # self.register_parameter('in_proj_bias', None)
            pass
        # self.out_proj = _LinearWithBias(embed_dim, embed_dim)

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = nn.Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None
        
        self.add_zero_attn = add_zero_attn

        
        '''
        if multiheadAtten is not None:
            self._reset_parameters_(multiheadAtten)
            pass
        else:
            self._reset_parameters()
        '''
        # pytorch 1.12
        # self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        # self.bias_k = self.bias_v = None
        
        # self.out_proj = _LinearWithBias(embed_dim, embed_dim)


        # self.add_zero_attn = add_zero_attn

        # self._reset_parameters()
        pass
    def _reset_parameters_(self, multiheadAtten):
        if self._qkv_same_embed_dim:
            nn.init.xavier_uniform_(self.in_proj_weight)
            self.in_proj_weight = multiheadAtten.in_proj_weight
        else:
            # nn.init.xavier_uniform_(self.q_proj_weight)
            # nn.init.xavier_uniform_(self.k_proj_weight)
            nn.init.xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

        self.out_proj.weight=multiheadAtten.out_proj.weight
    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            # nn.init.xavier_uniform_(self.in_proj_weight)
            pass
        else:
            # nn.init.xavier_uniform_(self.q_proj_weight)
            # nn.init.xavier_uniform_(self.k_proj_weight)
            # nn.init.xavier_uniform_(self.v_proj_weight)
            pass

        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            # nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)
    # def forward(self, query: Tensor, key: Tensor, value: Tensor,in_proj_bias: Tensor,key_padding_mask: Optional[Tensor] = None,need_weights: bool = True, attn_mask: Optional[Tensor] = None,average_attn_weights: bool = True,):
    def forward(self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
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
            proposal_mask: Optional[Tensor] = None,
        ):

        # set up shape vars
        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape

        if isinstance(embed_dim, torch.Tensor):
            # embed_dim can be a tensor when JIT tracing
            head_dim = embed_dim.div(self.num_heads, rounding_mode='trunc')
        else:
            head_dim = embed_dim // self.num_heads
        scaling = float(head_dim) ** -0.5

        '''
        # This is inline in_proj function with in_proj_weight and in_proj_bias
        _b = self.in_proj_bias
        _start = 0
        _end = embed_dim
        _w = self.in_proj_weight[_start:_end, :]
        if _b is not None:
            _b = _b[_start:_end]
        q = linear(query, _w, _b)

        # This is inline in_proj function with in_proj_weight and in_proj_bias
        _b = self.in_proj_bias
        _start = embed_dim
        _end = embed_dim * 2
        _w = self.in_proj_weight[_start:_end, :]
        if _b is not None:
            _b = _b[_start:_end]
        k = linear(key, _w, _b)

        # This is inline in_proj function with in_proj_weight and in_proj_bias
        _b = self.in_proj_bias
        _start = embed_dim * 2
        _end = None
        _w = self.in_proj_weight[_start:, :]
        if _b is not None:
            _b = _b[_start:]

        if not self.outDecoderLayer:
            v = linear(value, _w, _b)
        else:
            # import pdb; pdb.set_trace()
            v = value

        q = q*scaling
        
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
        
        src_len = k.size(1)

        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        '''
        v = value
        v = v.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
        # proposal mask 处理
        mask_shape = proposal_mask.shape
        proposal_mask = proposal_mask.view(mask_shape[0], mask_shape[1],-1)
        proposal_mask_nhead = proposal_mask.repeat(self.num_heads, 1,1)
        mask_list = []
        for i in range(bsz):
            mask_head = proposal_mask_nhead[i::bsz, :,:]
            mask_list.append(mask_head)
        mask_list_stack = torch.stack(mask_list)
        mask_stack_shape = mask_list_stack.shape
        mask_nhead = mask_list_stack.view(mask_stack_shape[0] * mask_stack_shape[1], \
            mask_stack_shape[2], mask_stack_shape[3])

        # attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        # attn_output_weights = F.dropout(attn_output_weights, p=self.dropout_p, training=training)
        
        # soft(k*v^T/sqrt(d))后添加mask
        # attn_output_weights = attn_output_weights * mask_nhead
        attn_output_weights = mask_nhead
        # 

        # attn_output = torch.bmm(attn_output_weights, v)

        attn_output = torch.einsum("qnl, qlc -> qnlc", attn_output_weights, v)
        
        attn_output = attn_output.sum(dim = 2)

        # import pdb; pdb.set_trace()

        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        # import pdb; pdb.set_trace()
        if not self.outDecoderLayer:
            attn_output = linear(attn_output, self.out_proj.weight, self.out_proj.bias)
            

        if need_weights:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
            return attn_output, attn_output_weights.sum(dim=1) / self.num_heads
        else:
            return attn_output, None

# 2022-12-09 添加，修改mask添加方式
class MultiHeadAttentionMyChannelAtten(nn.Module):
    def __init__(self,embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, \
            kdim=None, vdim=None, batch_first=False, device=None, dtype=None, multiheadAtten = None, outDecoderLayer = False):
        super(MultiHeadAttentionMyChannelAtten, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout_p = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads

        self.outDecoderLayer = outDecoderLayer

        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.register_parameter('q_proj_weight', None)
        self.register_parameter('k_proj_weight', None)
        self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = _LinearWithBias(embed_dim, embed_dim)

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = nn.Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None
        
        self.add_zero_attn = add_zero_attn

        

        if multiheadAtten is not None:
            self._reset_parameters_(multiheadAtten)
            pass
        else:
            self._reset_parameters()

        # pytorch 1.12
        # self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        # self.bias_k = self.bias_v = None
        
        # self.out_proj = _LinearWithBias(embed_dim, embed_dim)


        # self.add_zero_attn = add_zero_attn

        # self._reset_parameters()
        pass
    def _reset_parameters_(self, multiheadAtten):
        if self._qkv_same_embed_dim:
            nn.init.xavier_uniform_(self.in_proj_weight)
            self.in_proj_weight = multiheadAtten.in_proj_weight
        else:
            nn.init.xavier_uniform_(self.q_proj_weight)
            nn.init.xavier_uniform_(self.k_proj_weight)
            nn.init.xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

        self.out_proj.weight=multiheadAtten.out_proj.weight
    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            nn.init.xavier_uniform_(self.in_proj_weight)
        else:
            nn.init.xavier_uniform_(self.q_proj_weight)
            nn.init.xavier_uniform_(self.k_proj_weight)
            nn.init.xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)
    # def forward(self, query: Tensor, key: Tensor, value: Tensor,in_proj_bias: Tensor,key_padding_mask: Optional[Tensor] = None,need_weights: bool = True, attn_mask: Optional[Tensor] = None,average_attn_weights: bool = True,):
    def forward(self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            training: bool = True,
            key_padding_mask: Optional[Tensor] = None,
            need_weights: bool = False,
            attn_mask: Optional[Tensor] = None,
            use_separate_proj_weight: bool = False,
            q_proj_weight: Optional[Tensor] = None,
            k_proj_weight: Optional[Tensor] = None,
            v_proj_weight: Optional[Tensor] = None,
            static_k: Optional[Tensor] = None,
            static_v: Optional[Tensor] = None,
            proposal_mask: Optional[Tensor] = None,
        ):

        # set up shape vars
        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape

        if isinstance(embed_dim, torch.Tensor):
            # embed_dim can be a tensor when JIT tracing
            head_dim = embed_dim.div(self.num_heads, rounding_mode='trunc')
        else:
            head_dim = embed_dim // self.num_heads
        scaling = float(head_dim) ** -0.5

        # This is inline in_proj function with in_proj_weight and in_proj_bias
        _b = self.in_proj_bias
        _start = 0
        _end = embed_dim
        _w = self.in_proj_weight[_start:_end, :]
        if _b is not None:
            _b = _b[_start:_end]
        q = linear(query, _w, _b)

        # This is inline in_proj function with in_proj_weight and in_proj_bias
        _b = self.in_proj_bias
        _start = embed_dim
        _end = embed_dim * 2
        _w = self.in_proj_weight[_start:_end, :]
        if _b is not None:
            _b = _b[_start:_end]
        k = linear(key, _w, _b)

        # This is inline in_proj function with in_proj_weight and in_proj_bias
        _b = self.in_proj_bias
        _start = embed_dim * 2
        _end = None
        _w = self.in_proj_weight[_start:, :]
        if _b is not None:
            _b = _b[_start:]

        if not self.outDecoderLayer:
            v = linear(value, _w, _b)
        else:
            # import pdb; pdb.set_trace()
            v = value

        q = q*scaling
        
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)

        src_len = k.size(1)

        # attn_output_weights = torch.bmm(q, k.transpose(1, 2))

        # proposal mask 处理
        mask_shape = proposal_mask.shape
        proposal_mask = proposal_mask.view(mask_shape[0], mask_shape[1],-1)
        proposal_mask_nhead = proposal_mask.repeat(self.num_heads, 1,1)
        mask_list = []
        for i in range(bsz):
            mask_head = proposal_mask_nhead[i::bsz, :,:]
            mask_list.append(mask_head)
        mask_list_stack = torch.stack(mask_list)
        mask_stack_shape = mask_list_stack.shape
        mask_nhead = mask_list_stack.view(mask_stack_shape[0] * mask_stack_shape[1], \
            mask_stack_shape[2], mask_stack_shape[3])
        
        attn_channel=torch.einsum("bnl, bnc -> bncl", mask_nhead, q)
        attn_chanel = attn_channel.sigmoid()
        
        attn_output = torch.einsum("bncl, blc -> bnc", attn_chanel, v)
        

        # attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        # attn_output_weights = F.dropout(attn_output_weights, p=self.dropout_p, training=training)
        
        # soft(k*v^T/sqrt(d))后添加mask
        # attn_output_weights = attn_output_weights * mask_nhead

        # attn_output = torch.bmm(attn_output_weights, v)

        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        # import pdb; pdb.set_trace()
        if not self.outDecoderLayer:
            attn_output = linear(attn_output, self.out_proj.weight, self.out_proj.bias)
            

        if need_weights:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
            return attn_output, attn_output_weights.sum(dim=1) / self.num_heads
        else:
            return attn_output, None


# This class exists solely for Transformer; it has an annotation stating
# that bias is never None, which appeases TorchScript
class _LinearWithBias(nn.Linear):
    bias: Tensor  # type: ignore

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__(in_features, out_features, bias=True)  # type: ignore

#
# multihead attention
#

def _in_projection_packed(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w: Tensor,
    b: Optional[Tensor] = None,
) -> List[Tensor]:
    r"""
    Performs the in-projection step of the attention operation, using packed weights.
    Output is a triple containing projection tensors for query, key and value.
    Args:
        q, k, v: query, key and value tensors to be projected. For self-attention,
            these are typically the same tensor; for encoder-decoder attention,
            k and v are typically the same tensor. (We take advantage of these
            identities for performance if they are present.) Regardless, q, k and v
            must share a common embedding dimension; otherwise their shapes may vary.
        w: projection weights for q, k and v, packed into a single tensor. Weights
            are packed along dimension 0, in q, k, v order.
        b: optional projection biases for q, k and v, packed into a single tensor
            in q, k, v order.
    Shape:
        Inputs:
        - q: :math:`(..., E)` where E is the embedding dimension
        - k: :math:`(..., E)` where E is the embedding dimension
        - v: :math:`(..., E)` where E is the embedding dimension
        - w: :math:`(E * 3, E)` where E is the embedding dimension
        - b: :math:`E * 3` where E is the embedding dimension
        Output:
        - in output list :math:`[q', k', v']`, each output tensor will have the
            same shape as the corresponding input tensor.
    """
    E = q.size(-1)
    if k is v:
        if q is k:
            # self-attention
            return _LinearWithBias(q, w, b).chunk(3, dim=-1)
        else:
            # encoder-decoder attention
            w_q, w_kv = w.split([E, E * 2])
            if b is None:
                b_q = b_kv = None
            else:
                b_q, b_kv = b.split([E, E * 2])
            return (_LinearWithBias(q, w_q, b_q),) + _LinearWithBias(k, w_kv, b_kv).chunk(2, dim=-1)
    else:
        w_q, w_k, w_v = w.chunk(3)
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = b.chunk(3)
        return _LinearWithBias(q, w_q, b_q), _LinearWithBias(k, w_k, b_k), _LinearWithBias(v, w_v, b_v)


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
    # if has_torch_function_variadic(input, weight):
    #     return handle_torch_function(linear, (input, weight), input, weight, bias=bias)
    return torch._C._nn.linear(input, weight, bias)