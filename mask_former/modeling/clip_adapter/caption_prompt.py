from typing import List, Tuple

import clip
from torch import nn
import torch
from .utils import CLIP

class CaptionPromptExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self._buffer_init = False
        self.with_trainable_params = False
    
    def init_buffer(self, clip_model):
        self._buffer_init = True
    def forward(self, caption:List[str], clip_model: nn.Module()):
        raise NotImplementedError()

class PredefinedCaptionPromptExtractor(CaptionPromptExtractor):
    def __init__(self,):
        super().__init__()
        # self.templates = templates
    def forward(self, captions, clip_model: nn.Module, clip_tokenize_truncate=False, clip_tokenize_context_length=77):

        caption_tokens = [clip.tokenize(caption, truncate = clip_tokenize_truncate, context_length=clip_tokenize_context_length ) \
            for caption in captions]
        
        caption_inputs = torch.cat(caption_tokens).to(
            clip_model.text_projection.data.device
        )
        caption_features = clip_model.encode_text(caption_inputs)
        caption_features /= caption_features.norm(dim = -1, keepdim = True)
        del caption_inputs
        return caption_features

class PredefinedCaptionPromptExtractorAllSpace(CaptionPromptExtractor):
    def __init__(self,):
        super().__init__()
        # self.templates = templates
    def forward(self, captions, clip_model: nn.Module, clip_tokenize_truncate=False, clip_tokenize_context_length=77):

        caption_tokens = [clip.tokenize(caption, truncate = clip_tokenize_truncate, context_length=clip_tokenize_context_length ) \
            for caption in captions]
        
        caption_inputs = torch.cat(caption_tokens).to(
            clip_model.text_projection.data.device
        )
       
        caption_features = clip_model.encode_text_all(caption_inputs)
        caption_features /= caption_features.norm(dim = -1, keepdim = True)
        del caption_inputs
        return caption_features