import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange, repeat
from typing import Tuple, List, Optional


class DoubleConv(nn.Module):
    """(convolution => [GN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(mid_channels // 16, mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(mid_channels // 16, mid_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, guidance_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels - guidance_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, guidance=None): 
        x = self.up(x)
        if guidance is not None:
            x = torch.cat([x, guidance], dim=1)
        return self.conv(x)

class StandardDecoder(nn.Module):
    '''
    Refactored version of the decoder of CATSeg
    '''

    def __init__(self, device: str, extra_upsampling: bool, use_guidance: bool, input_dim: int, decoder_dims: List[int]) -> None:
        '''
        extra_upsampling: None, single o double to use extra upsampling layers
        input_dim: channel input dimension
        decoder_dims: channel output of each decoder layer
        guidance_input_dims: channel input of each guidance featmap
        guidance_output_dims: channel output (after projection) of each guidance featmap
        '''
        super().__init__()
        self.out_size = (192,192) if extra_upsampling else (96,96)
        self.extra_upsampling = extra_upsampling
        self.use_guidance = use_guidance
        guidance_input_dims = (256,128)
        guidance_output_dims = (32,16) if self.use_guidance else (0,0)
        self.device = device
        
        if self.use_guidance:
            self.decoder_guidance_projection = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(d, dp, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                ) for d, dp in zip(guidance_input_dims, guidance_output_dims)
            ]).to(device)

        self.decoder1 = Up(input_dim, decoder_dims[0], guidance_output_dims[0]).to(device)
        self.decoder2 = Up(decoder_dims[0], decoder_dims[1], guidance_output_dims[1]).to(device)

        if self.extra_upsampling:
            # upsampling with same feature dimension, no guidance
            self.decoder3 = Up(decoder_dims[1], decoder_dims[1], 0).to(device)

        self.head = nn.Conv2d(decoder_dims[-1], 1, kernel_size=3, stride=1, padding=1).to(device)

    def forward(self, x: Tensor, guidance: List[Tensor]) -> Tuple[Tensor,Tensor]:
        
        if self.decoder_guidance_projection is not None:
            proj_guidance = [proj(g) for proj, g in zip(self.decoder_guidance_projection, guidance[1:])]
        else:
            proj_guidance = (None,None,None)
        
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = self.decoder1(corr_embed, proj_guidance[0])
        corr_embed = self.decoder2(corr_embed, proj_guidance[1])
        
        if self.extra_upsampling:
            corr_embed = self.decoder3(corr_embed, None) 

        featmap = rearrange(corr_embed.clone(), '(B T) C H W -> B (T C) H W', B=B)
        corr_embed = self.head(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) () H W -> B T H W', B=B)

        if corr_embed.shape[2:] != self.out_size:
            print("Warning: this should not happen: ", self.out_size, corr_embed.shape)
            corr_embed = F.interpolate(corr_embed, size=tuple(self.out_size), mode='bilinear', align_corners=True)
            featmap = F.interpolate(featmap, size=tuple(self.out_size), mode='bilinear', align_corners=True)

        assert featmap.shape[2:] == tuple(self.out_size), f' problem with {featmap.shape} and {self.out_size}'

        return corr_embed, featmap


def get_decoder(args, device) -> StandardDecoder:    
    '''
    Return a Decoder type based on argument
    '''

    if args.image_encoder.decoder_type == 'standard':
        return StandardDecoder(device, args.image_encoder.extra_upsampling, args.image_encoder.use_decoder_guidance, input_dim=128, decoder_dims=[64,32])
    else:
        raise RuntimeError(f"Decoder type {args.image_encoder.decoder_type} not supported.")

