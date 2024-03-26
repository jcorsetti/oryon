import sys
import os
import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, List
sys.path.append(os.getcwd())
from einops import rearrange
import clip
import torchvision
from .tokenizer import SimpleTokenizer
from omegaconf import DictConfig

class CLIPEncoder(nn.Module):

    def __init__(self, device : str):

        super().__init__()
        self.clip_model, clip_trans = clip.load("ViT-L/14@336px", device, jit=False)
        clip_trans = clip_trans.transforms
        self.clip_trans = torchvision.transforms.Compose([clip_trans[0], clip_trans[1], clip_trans[-1]])
        self.clip_model.to(torch.float32)
        self.clip_model.eval()
        self.tokenizer = SimpleTokenizer('pretrained_models/bpe_simple_vocab_16e6.txt.gz')
        self.device = device
        self.feature_size = 768

        for param in self.clip_model.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        
        self.training = False
        
        return self

    def eval(self):
        
        self.train(False)
        
        return self

    def encode_image(self, image: Tensor) -> Tensor:
        
        clip_rgb = self.clip_trans(image)
        x = self.clip_model.visual.conv1(clip_rgb)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.clip_model.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.clip_model.visual.positional_embedding.to(x.dtype)
        x = self.clip_model.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip_model.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        clip_v_toks = self.clip_model.visual.ln_post(x[:, 1:, :])

        BS = clip_v_toks.shape[0]
        clip_v_toks = clip_v_toks.transpose(1,2).reshape(BS,-1, 24, 24)

        return clip_v_toks

    def encode_prompt(self, prompts : List[List[str]]) -> Tensor:
        '''
        Processes a vector of token indexes with a certain length
        '''
        prompts = [prompt_list[1:] for prompt_list in prompts]
        tokenized_text = torch.stack([self.tokenizer(prompt) for prompt in prompts]) # this should result in [B,80,77]
        tokenized_text = tokenized_text.to(self.device)

        B,T = tokenized_text.shape[:2]
        
        tokenized_text = rearrange(tokenized_text, 'B T L -> (B T) L', B=B,T=T)
        x = self.clip_model.token_embedding(tokenized_text).type(self.clip_model.dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.clip_model.positional_embedding.type(self.clip_model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip_model.ln_final(x).type(self.clip_model.dtype)
        # this is used to consider for the projection only the valid tokens
        eof = tokenized_text.argmax(dim=-1)
        x = x[torch.arange(x.shape[0]), eof]
        x = x @ self.clip_model.text_projection
        x = rearrange(x, '(B T) D -> B T D', B=B,T=T)

        return x

    def forward(self, image: Tensor, prompts : List[List[str]]) -> Tuple[Tensor,Tensor]:
        '''
        image: [B,3,H,W]
        prompts : List[List[str]]
        '''
        # skip first prompt, which is just the object prompt without any template, as this version works with templates
        
        text_embs = self.encode_prompt(prompts)
        visual_featmap = self.encode_image(image)

        return visual_featmap, text_embs


def get_vlm(args: DictConfig, device: str) -> CLIPEncoder:

    if args.image_encoder.vlm == 'clip':
        return CLIPEncoder(device)
    else:
        raise RuntimeError(f"VLM {args.vlm} not implemented.")
