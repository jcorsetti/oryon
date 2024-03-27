import torch
import numpy as np
from torch.nn.functional import interpolate
from typing import Tuple, Sequence, Optional
from utils.misc import box_from_mask
from utils import coordinates
from torch import Tensor
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode as mode

def is_item_empty(item: dict) -> bool:
    return (len(item['metadata']['cls_ids']) == 0 or len(item['metadata']['cls_names']) == 0 or len(item['metadata']['boxes']) == 0)

def scale_and_square_bbox(box: Sequence, scale:float) -> Sequence:
    
    y,x,h,w = box
    h,w = max(h,2), max(w,2)
    # get centers
    cx, cy = x + w/2, y + h/2
    new_w, new_h = w*scale, h*scale
    # recompute first point
    new_h = max(new_h, 2)
    new_w = max(new_w, 2)
    x = cx - new_w/2  
    y = cy - new_h/2
    
    # square bbox
    max_dim = max(new_h,new_w)
    # recompute if needed
    if new_w < max_dim:
        extra_w = max_dim - new_w
        x = x - extra_w/2.
    
    elif new_h < max_dim:
        extra_h = max_dim - new_h
        y = y - extra_h/2.

    return [int(y),int(x),int(max_dim),int(max_dim)]

def preprocess_item(item: dict) -> dict:
    '''
    Converts numpy arrays to tensor, fixes masks.
    '''

    assert len(item['metadata']['mask_ids']) == 1, f" Problem with instance {item['instance_id']}: no objects found. Check cls_id!"

    # move channels to dim 0
    item['rgb'] = item['rgb'].transpose(2, 0, 1) / 255.
    item['hw_size'] = item['mask'].shape
    
    # move to tensor
    for k, v in item.items():
        if isinstance(v, np.ndarray):
            item[k] = torch.tensor(v)
    
    # rgb useful for visualization, depth is necessary for testing
    item['orig_rgb'] = item['rgb'].clone()
    item['orig_depth'] = item['depth'].clone()
    item['eval_depth'] = item['depth'].clone()

    if 'poses' in item['metadata'].keys():
        item['metadata']['poses'] = [torch.tensor(v) for v in item['metadata']['poses']]

    mask_id = item['metadata']['mask_ids'][0]
    mask = torch.where(item['mask'] == mask_id, 1, 0)
    item['mask'] = mask

    y1,x1,y2,x2 = box_from_mask(mask.numpy(),id=1)
    item['metadata']['boxes'] = torch.tensor([y1,x1,y2-y1,x2-x1])

    return item

def get_resized_item(item: dict, coords: np.ndarray, size: Tuple) -> Tuple[dict, np.ndarray]:
    '''
    Rescales all image data (depth, mask, RGB) to a new dimension.
    Coordinates and bounding boxes are rescaled accordingly.
    '''
    rgb = item['rgb'].clone().unsqueeze(0)
    item['orig_rgb'] = rgb.clone()
    # interpolate wants batch and channel dimensions
    mask = item['mask'].clone().unsqueeze(0).unsqueeze(1).to(torch.uint8)
    depth = item['depth'].clone().unsqueeze(0).unsqueeze(1)

    H,W = rgb.shape[2:]
    # resize everything to training size
    rgb = interpolate(rgb, size=size, mode='bilinear').squeeze()
    mask = interpolate(mask, size=size, mode='nearest').squeeze()
    depth = interpolate(depth.float(), size=size, mode='nearest').squeeze()
    # add to item
    item['rgb'] = rgb
    item['cropped_mask'] = mask
    item['cropped_depth'] = depth
    # rescale correspondences
    coords = coordinates.scale_coords(torch.tensor(coords), (H,W), size).numpy()
    y,x,h,w = item['metadata']['boxes']
    box_coords = torch.tensor([[y,x],[h,w]])
    box_coords = coordinates.scale_coords(box_coords, (H,W), size).numpy()
    x,y,w,h = box_coords[0,1],box_coords[0,0],box_coords[1,1],box_coords[1,0]
    w = max(w,2)
    h = max(h,2)
    item['metadata']['boxes'] = [int(y),int(x),int(h),int(w)]

    return item, coords.astype(np.int16)

def check_validity(item: dict) -> bool:
    '''
    Return false if an item is not valid (e.g., empty mask)
    '''
    mask = torch.count_nonzero(item['mask']).item()

    return mask > 0
