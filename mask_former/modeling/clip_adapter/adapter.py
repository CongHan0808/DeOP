from typing import List
import torch
from torch import nn
from torch.nn import functional as F
from detectron2.structures import BitMasks
from .utils import build_clip_model, crop_with_mask, CLIP
from .text_prompt import PromptExtractor


class ClipAdapter(nn.Module):
    def __init__(self, clip_model_name: str, prompt_learner: PromptExtractor, layermaskvit= [11]):
        super().__init__()
        self.clip_model = build_clip_model(clip_model_name, layermaskvit=layermaskvit)
        self.prompt_learner = prompt_learner
        self.prompt_learner.init_buffer(self.clip_model)
        self.text_feature_buffer = {}

    def forward(self, image: torch.Tensor, text: List[str], **kwargs):
        image = self._preprocess_image(image, **kwargs)
        text_feature = self.get_text_features(text)  # k,feat_dim
        image_features = self.get_image_features(image)
        return self.get_sim_logits(text_feature, image_features)

    def _preprocess_image(self, image: torch.Tensor):
        return image

    def _get_text_features(self, noun_list: List[str]):
        if not self.prompt_learner.with_trainable_params:

            left_noun_list = [
                noun for noun in noun_list if noun not in self.text_feature_buffer
            ]
            if len(left_noun_list) > 0:
                left_text_features = self.prompt_learner(
                    left_noun_list, self.clip_model
                )
                self.text_feature_buffer.update(
                    {
                        noun: text_feature
                        for noun, text_feature in zip(
                            left_noun_list, left_text_features
                        )
                    }
                )
            return torch.stack([self.text_feature_buffer[noun] for noun in noun_list])
        else:
            text_features = self.prompt_learner(noun_list, self.clip_model)
            self.text_feature_buffer.update(
                {
                    noun: text_feature.detach()
                    for noun, text_feature in zip(noun_list, text_features)
                }
            )
            return text_features

    def get_text_features(self, noun_list: List[str]):
        return self._get_text_features(noun_list)

    def get_image_features(self, image: torch.Tensor):
        image_features,_ = self.clip_model.visual(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def get_sim_logits(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor,
        temperature: float = 100,
    ):
        return temperature * image_features @ text_features.T

    def normalize_feature(self, feat: torch.Tensor):
        return feat / feat.norm(dim=-1, keepdim=True)


class MaskFormerClipAdapter(ClipAdapter):
    def __init__(
        self,
        clip_model_name: str,
        prompt_learner: PromptExtractor,
        mask_fill: str = "mean",
        mask_expand_ratio: float = 1.0,
        mask_thr: float = 0.5,
        mask_matting: bool = False,
        region_resized: bool = True,
        layermaskvit=[],
    ):
        super().__init__(clip_model_name, prompt_learner,layermaskvit)
        self.non_object_embedding = nn.Parameter(
            torch.empty(1, self.clip_model.text_projection.shape[-1])
        )
        nn.init.normal_(
            self.non_object_embedding.data,
            std=self.clip_model.transformer.width ** -0.5,
        )
        # for test
        self.mask_fill = mask_fill
        if self.mask_fill == "zero":
            self.mask_fill = (0.0, 0.0, 0.0)
        elif self.mask_fill == "mean":
            self.mask_fill = [255.0 * c for c in CLIP.PIXEL_MEAN]
        else:
            raise NotImplementedError(
                "Unknown mask_fill method: {}".format(self.mask_fill)
            )
        self.mask_expand_ratio = mask_expand_ratio
        self.mask_thr = mask_thr
        self.mask_matting = mask_matting
        self.region_resized = region_resized

        self.register_buffer(
            "pixel_mean", torch.Tensor(CLIP.PIXEL_MEAN).reshape(1, 3, 1, 1) * 255.0
        )
        self.register_buffer(
            "pixel_std", torch.Tensor(CLIP.PIXEL_STD).reshape(1, 3, 1, 1) * 255.0
        )

    def forward(
        self,
        image: torch.Tensor,
        text: List[str],
        mask: torch.Tensor,
        normalize: bool = True,
    ):
        image, valid_flag = self._preprocess_image(image, mask, normalize=normalize)
        if image is None:
            return None, valid_flag
        if isinstance(image, list):
            image_features = torch.cat(
                [self.get_image_features(image_i) for image_i in image], dim=0
            )
        else:
            image_features = self.get_image_features(image)
        text_feature = self.get_text_features(text)  # k,feat_dim
        return self.get_sim_logits(text_feature, image_features), valid_flag
     
    # 获取每个mask对应region feature 
    def get_region_features(
            self,
            image: torch.Tensor,
            mask: torch.Tensor,
            normalize: bool = True
        ):
        image, valid_flag = self._preprocess_image_kd(image, mask, normalize=normalize)
        if image is None:
            return None, valid_flag
        if isinstance(image, list):
            image_features = torch.cat(
                [self.get_image_features(image_i) for image_i in image], dim=0
            )
        else:
            image_features = self.get_image_features(image)
        
        return image_features, valid_flag


    def _preprocess_image(
        self, image: torch.Tensor, mask: torch.Tensor, normalize: bool = True
    ):
        """crop, mask and normalize the image

        Args:
            image ([type]): [C,H,W]
            mask ([type]): [K,H,W
            normalize (bool, optional): [description]. Defaults to True.
        """
        dtype = mask.dtype
        bin_mask = mask > self.mask_thr
        valid = bin_mask.sum(dim=(-1, -2)) > 0
        bin_mask = bin_mask[valid]
        mask = mask[valid]
        if not self.mask_matting:
            mask = bin_mask
        bin_mask = BitMasks(bin_mask)
        bboxes = bin_mask.get_bounding_boxes()
        # crop,mask
        regions = [
            crop_with_mask(
                image.type(dtype),
                single_mask.type(dtype),
                bbox,
                fill=self.mask_fill,
                expand_ratio=self.mask_expand_ratio,
            )[None, ...]
            for bbox, single_mask in zip(bboxes, mask)
        ]
        if len(regions) == 0:
            return None, valid
        if normalize:
            regions = [(r - self.pixel_mean) / self.pixel_std for r in regions]
        # resize
        if self.region_resized:
            regions = [
                F.interpolate(r, size=(224, 224), mode="bicubic") for r in regions
            ]
            regions = torch.cat(regions)
        return regions, valid
    def _preprocess_image_kd(
        self, image: torch.Tensor, mask: torch.Tensor, normalize: bool = True
    ):
        """crop, mask and normalize the image

        Args:
            image ([type]): [C,H,W]
            mask ([type]): [K,H,W
            normalize (bool, optional): [description]. Defaults to True.
        """
        dtype = mask.dtype
        bin_mask = mask > self.mask_thr
        valid = bin_mask.sum(dim=(-1, -2)) > 0
        # bin_mask = bin_mask[valid]
        # mask = mask[valid]
        if not self.mask_matting:
            mask = bin_mask
        bin_mask = BitMasks(bin_mask)
        bboxes = bin_mask.get_bounding_boxes()
        # crop,mask
        regions = [
            crop_with_mask(
                image.type(dtype),
                single_mask.type(dtype),
                bbox,
                fill=self.mask_fill,
                expand_ratio=self.mask_expand_ratio,
            )[None, ...]
            for bbox, single_mask in zip(bboxes, mask)
        ]
        if len(regions) == 0:
            return None, valid
        if normalize:
            regions = [(r - self.pixel_mean) / self.pixel_std for r in regions]
        # resize
        if self.region_resized:
            regions = [
                F.interpolate(r, size=(224, 224), mode="bicubic") for r in regions
            ]
            regions = torch.cat(regions)
        return regions, valid


    def get_text_features(self, noun_list: List[str]):
        object_text_features = self._get_text_features(noun_list)
        non_object_text_features = (
            self.non_object_embedding
            / self.non_object_embedding.norm(dim=-1, keepdim=True)
        )
        return torch.cat([object_text_features, non_object_text_features], dim=0)


class PerPixelClipAdapter(ClipAdapter):
    def __init__(self, *args, **kwargs):
        super(PerPixelClipAdapter, self).__init__(*args, **kwargs)
        self.register_buffer(
            "pixel_mean", torch.Tensor(CLIP.PIXEL_MEAN).reshape(1, 3, 1, 1) * 255.0
        )
        self.register_buffer(
            "pixel_std", torch.Tensor(CLIP.PIXEL_STD).reshape(1, 3, 1, 1) * 255.0
        )

    def _preprocess_image(self, image: torch.Tensor):
        return (image.to(self.pixel_mean.device) - self.pixel_mean) / self.pixel_std

    def get_image_features(self, image: torch.Tensor, per_pixel: bool = False):
        if per_pixel:
            image_features = self.clip_model.visual(image, return_cls=False)  # b,h,w,c
        else:
            image_features = self.clip_model.visual(image)[:, None, None, :].expand(
                image.shape[0], 2, 2, -1
            )  # b,c
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def forward(
        self, image: torch.Tensor, text: List[str], per_pixel: bool = True, **kwargs
    ):
        image = self._preprocess_image(image, **kwargs)
        text_feature = self.get_text_features(text)  # k,feat_dim
        image_features = self.get_image_features(image)
        return self.get_sim_logits(text_feature, image_features)

class MaskFormerClipFeatureAdapter(nn.Module):
    def __init__(self, clip_model_name: str, prompt_learner: PromptExtractor,learn_token=False, \
    prompt_num_token=5,learn_position=False,position_layers=[], input_resolution_run = 384,\
    layermaskvit=[11],maskselfattn=False):
        super().__init__()
        self.clip_model = build_clip_model(clip_model_name,maskselfattn = maskselfattn,layermaskvit=layermaskvit,learn_token=learn_token, prompt_num_token=prompt_num_token,learn_position=learn_position,position_layers=position_layers, input_resolution_run = input_resolution_run)
        self.prompt_learner = prompt_learner
        self.prompt_learner.init_buffer(self.clip_model)
        self.text_feature_buffer = {}

        self.non_object_embedding = nn.Parameter(
            torch.empty(1, self.clip_model.text_projection.shape[-1])
        )

        nn.init.normal_(
            self.non_object_embedding.data,
            std=self.clip_model.transformer.width ** -0.5,
        )

    def forward(self, image: torch.Tensor, text: List[str],mask: torch.Tensor, **kwargs):
        image = self._preprocess_image(image, **kwargs)
        # import pdb; pdb.set_trace()
        text_feature = self.get_text_features(text)  # k,feat_dim
        image_features = self.get_image_features(image, mask, return_cls=False)
        return self.get_sim_logits(text_feature, image_features)

    def _preprocess_image(self, image: torch.Tensor):
        return image

    def _get_text_features(self, noun_list: List[str]):
        if not self.prompt_learner.with_trainable_params:

            left_noun_list = [
                noun for noun in noun_list if noun not in self.text_feature_buffer
            ]
            if len(left_noun_list) > 0:
                left_text_features = self.prompt_learner(
                    left_noun_list, self.clip_model
                )
                self.text_feature_buffer.update(
                    {
                        noun: text_feature
                        for noun, text_feature in zip(
                            left_noun_list, left_text_features
                        )
                    }
                )
            return torch.stack([self.text_feature_buffer[noun] for noun in noun_list])
        else:
            text_features = self.prompt_learner(noun_list, self.clip_model)
            self.text_feature_buffer.update(
                {
                    noun: text_feature.detach()
                    for noun, text_feature in zip(noun_list, text_features)
                }
            )
            return text_features

    def get_text_features(self, noun_list: List[str]):
        object_text_features = self._get_text_features(noun_list)
        non_object_text_features = (
            self.non_object_embedding
            / self.non_object_embedding.norm(dim=-1, keepdim=True)
        )
        return torch.cat([object_text_features, non_object_text_features], dim=0)
    
    def get_image_features(self, image: torch.Tensor, mask: torch.tensor, return_cls: bool):
        image_features = self.clip_model.visual(image, return_cls=return_cls)
        h, w = image_features.shape[1], image_features.shape[2]
        image_features = image_features/(h*w)
        
        upsample = False
        upDownSampel = False
        if upsample and not upDownSampel:
            image_features_upsample = F.interpolate(image_features.permute(0,3,1,2), size = (mask.shape[1], mask.shape[2]), mode="bilinear")
            image_features = image_features_upsample.permute(0,2,3,1)
            mask_downsample = mask.unsqueeze(1)
            mask_downsample = mask_downsample.to(image_features.dtype)
        elif not upsample and upDownSampel:
            image_features_upsample = F.interpolate(image_features.permute(0,3,1,2), size = (mask.shape[1]//4, mask.shape[2]//4), mode="bilinear")
            image_features = image_features_upsample.permute(0,2,3,1)
            mask = mask.to(image_features.dtype)
            mask_downsample = F.interpolate(mask.unsqueeze(1), size=(image_features.shape[1], image_features.shape[2]), mode = "nearest")
        else:
            mask = mask.to(image_features.dtype)
            mask_downsample = F.interpolate(mask.unsqueeze(1), size=(image_features.shape[1], image_features.shape[2]), mode = "nearest")
        mask_new = []
        
        if((mask_downsample.sum(dim=(2,3))==0).sum() > 0):
            for m, m_src in zip(mask_downsample, mask):
                # import pdb; pdb.set_trace()
                if(m.sum(dim =(1,2)) == 0):
                    # import pdb; pdb.set_trace()
                    idxMax = m_src.argmax()
                    idx_x = idxMax // m_src.shape[1]
                    idx_y = idxMax % m_src.shape[1]
                    x_ratio = m_src.shape[0]// m.shape[1] +1
                    y_ratio = m_src.shape[1] // m.shape[2] +1
                    
                    idx_x, idx_y =mask2pt(m_src)
                    m[:, idx_x//x_ratio, idx_y // y_ratio] = 1
                    
                mask_new.append(m)
            mask_downsample = torch.stack(mask_new)
        
        
        image_features = image_features.repeat((mask_downsample.shape[0],1,1,1))
        image_features=torch.einsum("nhwd, nchw->ncd", image_features, mask_downsample)
        image_features = image_features.squeeze(1)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def get_sim_logits(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor,
        temperature: float = 100,
    ):
        return temperature * image_features @ text_features.T

    def normalize_feature(self, feat: torch.Tensor):
        return feat / feat.norm(dim=-1, keepdim=True)

def mask2pt(mask: torch.Tensor):
    col = mask.sum(dim = 0).nonzero()
    y1= col.min()
    y2 = col.max()
    row = mask.sum(dim = 1).nonzero()
    x1 = row.min()
    x2 = row.max()
    return (x1+x2)//2, (y1+y2)//2