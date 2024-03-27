import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import torch
from PIL import Image
from os.path import join
from matplotlib import pyplot as plt
from matplotlib import collections as mc
from os.path import join
from utils.pcd import lift_pcd, get_diameter
from bop_toolkit_lib.misc import get_symmetry_transformations
from typing import Tuple, Optional, List
import pickle
import json
from plyfile import PlyData


def get_camera() -> np.ndarray:
    camera = np.asarray([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])
    return camera

def get_pair_instance_list(root: str, split: str) -> List:

    path_split = join(root,'fixed_split',split)
    with open(join(path_split,'instance_list.txt')) as f:
        instances = f.readlines()
    
    new_list = list()
    for instance in instances:

        _, id_a, id_q, obj_id = instance.strip('\n').split(',')
        scene_a, img_a = id_a.strip(' ').split(' ')
        scene_q, img_q = id_q.strip(' ').split(' ')
        obj_id = obj_id.strip(' ')
            
        new_list.append((int(scene_a), int(img_a), int(scene_q), int(img_q), int(obj_id)))

    return new_list

def get_single_instance_list(root: str, split: str) -> list:

    instances = set()
    with open(join(root,'fixed_split',split,'instance_list.txt')) as f:
        for line in f.readlines():
            _, id_a, id_q, obj_id = line.strip('\n').split(',')
            scene_a, img_a = id_a.strip(' ').split(' ')
            scene_q, img_q = id_q.strip(' ').split(' ')
            obj_id = obj_id.strip(' ')
            instances.add((scene_a,img_a,obj_id))
            instances.add((scene_q,img_q,obj_id))
    return list(instances)

def get_obj_rendering(root: str, obj_id: int) -> dict:
    '''
    returns object usable for vispy rendering
    argument obj_model is expected to have the following fields:
      - pts : (N,3) xyz points in mm
      - normals: (N,3) normals
      - faces: (M,3) polygon faces needed for rendering 
    '''

    pcd = PlyData.read(os.path.join(root,'models_bop','obj_{:06d}.ply'.format(obj_id)))
    # these are already in mm    
    xs = pcd['vertex']['x']
    ys = pcd['vertex']['y']
    zs = pcd['vertex']['z']
    nxs = pcd['vertex']['nx']
    nys = pcd['vertex']['ny']
    nzs = pcd['vertex']['nz']

    raw_vertexs = np.asarray(pcd['face']['vertex_indices'])
    faces = np.stack([vert for vert in raw_vertexs],axis=0)

    xyz = np.stack((xs,ys,zs), axis=1)
    normals = np.stack((nxs,nys,nzs), axis=1)
    
    return {
        'pts': xyz,
        'normals': normals,
        'faces': faces
    }



def get_obj_names(root: str) -> dict:

    with open(join(root,'models_bop/models_name.json')) as f:
        return json.load(f)

def get_part_data(root: str) -> dict:
    '''
    Get all object poses of a split
    Translation here are in meters
    Boxed in [x,y,w,h]
    '''

    new_data = dict()

    for scene_folder in os.listdir(join(root,'split','test')):

        fa = open(join(root,'split','test',scene_folder,'scene_gt.json'))
        fm = open(join(root,'split','test',scene_folder,'scene_gt_info.json'))
            
        data = json.load(fa)
        meta_data = json.load(fm)
        
        for img_k, img_data in data.items():
            
            for i, (obj_data, obj_metadata) in enumerate(zip(img_data, meta_data[img_k])):
                
                pose = np.eye(4)
                r = np.asarray(obj_data['cam_R_m2c']).reshape(3,3)
                t = np.asarray(obj_data['cam_t_m2c']) / 1000. # in annotation is in mm!
                pose[:3,:3] = r
                pose[:3, 3] = t
                cls_id = int(obj_data['obj_id'])

                item = {
                    'pose' : pose,
                    'cls_id': cls_id,
                    'box': obj_metadata['bbox_visib'],
                    'mask_idx': i + 1
                }
                
                img_dk = f'{int(scene_folder)}_{int(img_k)}'
                cls_dk = f'{int(cls_id)}'
                
                if i == 0:
                    new_data[img_dk] = {}
                new_data[img_dk][cls_dk] = item
        
        fa.close()
        fm.close()

    return new_data

def get_item_metadata(root: str, scene_id: int, img_id: int, pose_annots: dict, cls_names_dict: dict, cls_id: Optional[int] = None) -> dict:   

    img_annots = pose_annots[f'{scene_id}_{img_id}']
    # list of objects in current image
    img_cls_list = list(img_annots.keys())
    cls_ids, mask_ids, cls_names, cls_descs, poses, boxes = list(),list(),list(),list(),list(), list()

    for obj_id in img_cls_list:
        if cls_id is not None:
           if int(obj_id) != int(cls_id):
                continue

        cls_ids.append(int(obj_id))
        mask_ids.append(img_annots[obj_id]['mask_idx'])
        cls_names.append(cls_names_dict[obj_id][0])
        cls_descs.append(cls_names_dict[obj_id][1:])
        poses.append(img_annots[obj_id]['pose'])
        boxes.append(img_annots[obj_id]['box'])

    
    return {
        'cls_ids': cls_ids,
        'mask_ids': mask_ids,
        'cls_names': cls_names,
        'cls_descs': cls_descs,
        'poses': poses,
        'boxes': boxes,
    }

def get_item_data(root: str, scene_id: int, img_id: int, pose_annots: dict, cls_names: dict, cls_id: Optional[int] = None, mask_type:Optional[str]='oracle', hf_depth:bool=False) -> dict:
    '''
    Return TOYL data given root and ifs of image and scene
    '''
    metadata = get_item_metadata(root, scene_id, img_id, pose_annots, cls_names, cls_id=cls_id)

    base_path = join(root, 'split', 'test', f'{scene_id:06d}') 
    img = np.asarray(Image.open(join(base_path,'rgb',f'{img_id:06d}' + '.png')).convert('RGB'))
    if mask_type == 'oracle':
        mask = np.asarray(Image.open(join(base_path,'mask_visib',f'{img_id:06d}' + '.png')).convert('L'))
        #raise RuntimeError("Not yer supported for TOYL")
    elif mask_type == 'ovseg':
        mask = np.asarray(Image.open(join(base_path,'mask_pred',f'{img_id:06d}' + '.png')).convert('L'))
    elif mask_type == 'san':
        # this is 1 for class and 255 in the others
        path = join(root, 'san_name', f'{scene_id} {img_id} {cls_id}.png')
        mask = np.asarray(Image.open(path).convert('L'))
        mask_id = metadata['mask_ids'][0]
        # convert to NOCS format
        mask = np.where(mask == 1, mask_id, 255)
    elif mask_type == 'oryon':
        # this is 1 for class and 255 in the others
        path = join(root, 'oryon', f'{scene_id} {img_id} {cls_id}.png')
        mask = np.asarray(Image.open(path).convert('L'))
        mask_id = metadata['mask_ids'][0]
        # convert to NOCS format
        mask = np.where(mask == 1, mask_id, 255)
    else:
        raise RuntimeError(f'Mask type {mask_type} not implemented.')
    
    if hf_depth:
        depth = np.asarray(Image.open(join(base_path,'hf_depth',f'{img_id:06d}' + '.png')))
    else:
        depth = np.asarray(Image.open(join(base_path,'depth',f'{img_id:06d}' + '.png')))
    
    instance_id = f'{scene_id} {img_id} {cls_id}'

    data = {
        'rgb': img,
        'mask': mask,
        'depth': depth, # account for TOYL(?) object scale!
        'metadata': metadata,
        'instance_id': instance_id
    }
    
    return data

def get_obj_data(root: str) -> Tuple[dict,dict,dict]:

    '''
    Returns complete information about a dataset point clouds
    '''

    obj_files = [file for file in os.listdir(join(root, 'models_bop')) if '.ply' in file]
    obj_models, obj_diams, obj_symm = dict(), dict(), dict()

    with open(join(root,'models_bop','models_info.json')) as f:
        models_info = json.load(f)

    for obj_file in obj_files:

        obj_id = int(os.path.splitext(obj_file[4:])[0])
        model_info = models_info[str(obj_id)]
        
        obj_models[obj_id] = get_obj_rendering(root, obj_id)
        obj_diams[obj_id] = model_info['diameter']
        obj_symm[obj_id] = get_symmetry_transformations(model_info, max_sym_disc_step=0.05)
        
    return (obj_models, obj_diams, obj_symm)

def get_pcd(root : str, split : str, scene_id : int, img_id : int) -> dict:
    '''
    Get NOCS data of a single image in point cloud form, return in Meters
    '''
    K = np.asarray([[572.4114, 0.0, 325.2611], [0.0, 573.5704, 242.0489], [0.0, 0.0, 1.0]])
    id = join(root, 'split', split, f'{scene_id:06d}')
    rgb = Image.open(f'{id}/rgb/{img_id:06d}.png').convert('RGB')
    depth = Image.open(f'{id}/depth/{img_id:06d}.png')
    mask = Image.open(f'{id}/mask_visib/{img_id:06d}.png').convert('L')
    
    rgb = torch.tensor(np.asarray(rgb) / 255.)
    depth = torch.tensor(np.asarray(depth)).unsqueeze(2)
    mask = torch.tensor(np.asarray(mask)).unsqueeze(2)

    H,W = mask.shape[:2]
    xs = torch.linspace(0, W-1, steps=W)
    ys = torch.linspace(0, H-1, steps=H)
    xmap, ymap = torch.meshgrid(xs,ys, indexing='xy')
    xmap, ymap = xmap.unsqueeze(-1), ymap.unsqueeze(-1)

    to_lift = torch.cat((depth, rgb, mask, ymap, xmap),dim=-1)
    pcd = lift_pcd(to_lift, torch.tensor(K).flatten())

    return {
        'xyz' : pcd[:,:3] / 1000.,
        'rgb' : pcd[:,3:6],
        'mask' : pcd[:,6],
        'yx_map' : pcd[:,7:]
    }

def filter_pcd(pcd : dict, mask_idx : int) -> dict:
    '''
    Crops a NOCS point cloud to retain only a specified object
    '''
    idxs = torch.nonzero(pcd['mask'] == mask_idx).squeeze()

    return {
        'xyz' : pcd['xyz'][idxs],
        'rgb' : pcd['rgb'][idxs],
        'mask' : pcd['mask'][idxs],
        'yx_map' : pcd['yx_map'][idxs],
    }