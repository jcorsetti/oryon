import os
import sys
import torch
from torch import Tensor, nn
from easydict import EasyDict
sys.path.append(os.getcwd())
from models.pointdsc.PointDSC import PointDSC
import json

def get_pointdsc_pose(pointdsc_model : nn.Module, pcd1 : Tensor, pcd2 : Tensor, device : str) -> Tensor:
    '''
    pointdsc_model: pretrained model of PointDSC
    pcd1: N,3 Tensor
    pcd2: N,3 Tensor
    NB: pcd1 and pcd2 are points belonging to correspondences
    '''
    
    corr_pos = torch.cat([pcd1, pcd2], axis=-1)
    corr_pos = corr_pos - corr_pos.mean(0)
    data = {
            'corr_pos': corr_pos.unsqueeze(0).to(device).float(),
            'src_keypts': pcd1.unsqueeze(0).to(device).float(),
            'tgt_keypts': pcd2.unsqueeze(0).to(device).float(),
            'testing': True,
            }
    pointdsc_model.eval()
    with torch.no_grad():
        res = pointdsc_model(data)
    return res['final_trans'].squeeze(0).cpu().to(torch.float32)


def get_pointdsc_solver(ckpt_path : str, device : str) -> Tensor:
    '''
    Initializes pretrained PointDSC module
    '''

    config_path = f'{ckpt_path}/snapshot/PointDSC_3DMatch_release/config.json'
    config = json.load(open(config_path, 'r'))
    config = EasyDict(config)

    model = PointDSC(
        in_dim=config.in_dim,
        num_layers=config.num_layers,
        num_channels=config.num_channels,
        num_iterations=config.num_iterations,
        ratio=config.ratio,
        sigma_d=config.sigma_d,
        k=config.k,
        nms_radius=config.inlier_threshold,
    ).to(device)
    model.load_state_dict(torch.load(f'{ckpt_path}/snapshot/PointDSC_3DMatch_release/models/model_best.pkl', map_location=device), strict=False)
    model.eval()
    
    for param in model.parameters():
        param.requires_grad = False
    
    return model
