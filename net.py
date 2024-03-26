import torch
import numpy as np
from typing import OrderedDict, Tuple, List
from omegaconf import DictConfig
from torch import Tensor
from torchvision.models import swin_b, Swin_B_Weights
from torchvision.transforms.functional import normalize
from torch.nn.functional import interpolate
from torchvision.models.feature_extraction import create_feature_extractor
from models.vlm import get_vlm
from models.fusion import get_fusion_module
from models.decoder import get_decoder
from models.vlm import get_vlm
from torch import nn

def weights_init_kaiming(m):

    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Upsample) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv1d):
        torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class Oryon(torch.nn.Module):
    def __init__(self, args : DictConfig, device : str):
        
        super().__init__()
        
        self.args = args.model
        self.device = device
        self.vlm = get_vlm(self.args, self.device)
        self.guidance_backbone = self.init_guidance_backbone(self.device)
        self.fusion = get_fusion_module(self.args, self.device)
        self.decoder = get_decoder(self.args, self.device)
        self.init_all()

    def get_trainable_parameters(self) -> list:

        param_list = []
        param_list.extend(self.fusion.parameters())
        param_list.extend(self.decoder.parameters())

        return param_list

    def init_guidance_backbone(self, device):
        swin = swin_b(weights=Swin_B_Weights.DEFAULT)
        for param in swin.parameters():
            param.requires_grad = False
        return_nodes = {
            'features.1.1.add_1' : 'guidance3', # [128,96,96]
            'features.2.reduction' : 'guidance2', # [256,48,48]
            'features.4.reduction' : 'guidance1' #  [512,24,24]
        }

        backbone = create_feature_extractor(swin, return_nodes=return_nodes)
        backbone.eval()
        backbone = backbone.to(device)
        return backbone

    def get_guidance_embeds(self, img_ : Tensor) -> List[Tensor]:
        '''
        Return guidance embeddings as in CATSeg, from Swin_b transformer
        normalization from https://pytorch.org/vision/main/models/generated/torchvision.models.swin_b.html
        '''
        img = img_.clone()

        img = interpolate(img, size=(384,384), mode='bicubic',align_corners=True)
        img = normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        outs = self.guidance_backbone(img)

        guid1 = outs['guidance1'].transpose(2,3).transpose(1,2)
        guid2 = outs['guidance2'].transpose(2,3).transpose(1,2)
        guid3 = outs['guidance3'].transpose(2,3).transpose(1,2)

        return [guid1,guid2,guid3]


    def train(self, mode=True):
        
        self.training = mode
        self.vlm.train(mode)
        self.fusion.train(mode)
        self.decoder.train(mode)
        
        return self

    def eval(self):

        self.train(False)

    def get_image_input(self, xs : dict) -> Tuple[dict, dict]:
        
        # create input with RGB channels
        input_a = {'rgb': xs['anchor']['rgb'].to(self.device)}
        input_q = {'rgb': xs['query']['rgb'].to(self.device)}

        return (input_a, input_q)

    def init_all(self):
        self.fusion.clip_conv.apply(weights_init_kaiming)
        
        if self.args.use_catseg_ckpt:
            #print("Loading CATSeg checkpoint")
            ckpt = torch.load('pretrained_models/catseg.pth', map_location=self.device)
            # set checkpoint names
            new_state_dict = dict()
            # this is necessary because of the refactoring we carried out
            old_fusion_key = 'sem_seg_head.predictor.transformer'
            new_fusion_key = 'fusion'
            old_dec_key = 'fusion.decoder'
            new_dec_key = 'decoder.decoder'
            
            # changing prefix of fusion and decoder keys
            for k,v in ckpt['model'].items():
                if k.startswith(old_fusion_key):
                    new_k = k.replace(old_fusion_key, new_fusion_key)
                    if new_k.startswith(old_dec_key):
                        new_k = new_k.replace(old_dec_key, new_dec_key)
                    if new_k.startswith('fusion.head'):
                        new_k = new_k.replace('fusion.head', 'decoder.head')
                    new_state_dict[new_k] = v
                
            # if using CLIP, we are also loading CATSeg's finetuned CLIP    
            if self.args.image_encoder.vlm == 'clip':
                old_clip_key = 'sem_seg_head.predictor.clip_model'
                new_clip_key = 'vlm.clip_model'
            
                for k,v in ckpt['model'].items():
                    if k.startswith(old_clip_key):
                        new_k = k.replace(old_clip_key,new_clip_key)
                        new_state_dict[new_k] = v
                            
            inco_keys = self.load_state_dict(new_state_dict,strict=False)
            #print(inco_keys)

        else:
            #print("Training from scratch")            
            self.fusion.apply(weights_init_kaiming)
            self.decoder.apply(weights_init_kaiming)


    def forward(self, xs: dict):
            
        # extracting CLIP features        
        visual_a = self.vlm.encode_image(xs['anchor']['rgb'])
        visual_q = self.vlm.encode_image(xs['query']['rgb'])
        prompt_emb = self.vlm.encode_prompt(xs['prompt'])

        guid_a = self.get_guidance_embeds(xs['anchor']['rgb'])        
        guid_q = self.get_guidance_embeds(xs['query']['rgb'])        

        # get encoded feature maps [D,N,N]
        prompt_emb = prompt_emb.unsqueeze(1)
        feats_a = self.fusion.forward(visual_a, prompt_emb, guid_a)
        feats_q = self.fusion.forward(visual_q, prompt_emb, guid_q)

        mask_a, featmap_a = self.decoder.forward(feats_a, guid_a)
        mask_q, featmap_q = self.decoder.forward(feats_q, guid_q)
        
        assert featmap_a.shape[2:] == self.args.image_encoder.img_size

        return {
            'featmap_a' : featmap_a,
            'featmap_q' : featmap_q,
            'mask_a' : mask_a,
            'mask_q' : mask_q
        }
