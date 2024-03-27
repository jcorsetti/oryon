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
import json
import pickle

def get_camera() -> np.ndarray:
    return np.asarray([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])


def get_pair_instance_list(root: str, split: str) -> List:

    path_split = join(root,'fixed_split',split)
    with open(join(path_split,'instance_list.txt')) as f:
        instances = f.readlines()
    
    new_list = list()
    for instance in instances:

        _, idx_a, idx_q, cat_id = instance.split(',')
        
        cat_id_a, obj_name_a = cat_id.strip().split(' ')
        cat_id_a = int(cat_id_a)
        scene_a, img_a = [int(n) for n in idx_a.split(' ') if n != '']
        scene_q, img_q = [int(n) for n in idx_q.split(' ') if n != '']
        
        new_list.append((scene_a, img_a, scene_q, img_q, obj_name_a))

    return new_list

def get_single_instance_list(root: str, split: str) -> list:

    instances = set()
    with open(join(root,'fixed_split',split,'instance_list.txt')) as f:
        for line in f.readlines():
            _, id_a, id_q, obj_id = line.strip('\n').split(',')
            scene_a, img_a = id_a.strip(' ').split(' ')
            scene_q, img_q = id_q.strip(' ').split(' ')
            obj_id = obj_id.strip(' ').split(' ')[1].strip('\n')
            instances.add((scene_a,img_a,obj_id))
            instances.add((scene_q,img_q,obj_id))
    return list(instances)

def get_obj_names(root: str) -> dict:

    with open(join(root,'obj_names.json')) as f:
        return json.load(f)

def get_obj_rendering(root: str, obj_id: str) -> dict:
    '''
    returns object usable for vispy rendering
    argument obj_model is expected to have the following fields:
      - pts : (N,3) xyz points in mm
      - normals: (N,3) normals
      - faces: (M,3) polygon faces needed for rendering 
    '''

    pts, normals, faces = list(), list(), list()
    basepath = join(root,'obj_models','real_test', obj_id)

    with open(basepath + '_vertices.txt') as f:
        lines = [line.split(' ') for line in f.readlines()]
        for line in lines:
            pts.append([float(line[0]), float(line[1]), float(line[2])])

    with open(basepath + '_normals.txt') as f:
        lines = [line.split(' ') for line in f.readlines()]
        for line in lines:
            normals.append([float(line[0]), float(line[1]), float(line[2])])

    with open(basepath + '.obj') as f:
        lines = [line.split(' ')[1:] for line in f.readlines() if line.startswith('f')]
        for line in lines:
            f1, f2, f3 = int(line[0].split('/')[0]),int(line[1].split('/')[0]),int(line[2].split('/')[0])
            faces.append([f1,f2,f3])

    return {
        'pts': np.asarray(pts) * 1000,
        'normals': np.asarray(normals),
        'faces': np.asarray(faces),
    }

def get_part_data(root: str) -> dict:
    '''
    Get all object poses of a split
    '''

    poses = dict()

    for img_file in os.listdir(join(root,'gts','real_test')):

        f = open(join(root,'gts','real_test',img_file),'rb')
        data = pickle.load(f)['gt_RTs']
        scene_id, img_id = os.path.splitext(img_file)[0].split('_')[-2:]
        poses[f'{int(scene_id)}_{int(img_id)}'] = data
    
    return poses

def get_partition_instances(root: str, split: str) -> set:
    '''
    Returns a set of instances used in a fixed partition
    '''

    part_set = set()
    with open(join(root,'fixed_split',split,'instance_list.txt')) as f:
        lines = f.readlines()

    for line in lines:

        src_part, scene_a, img_a, scene_q, img_q, cat_id, cat_name = line.strip('\n').split(' ')
        part_set.add((src_part.strip(','), int(scene_a), int(img_a.strip(',')), int(cat_id.strip(','))))
        part_set.add((src_part.strip(','), int(scene_q), int(img_q.strip(',')), int(cat_id.strip(','))))

    return part_set

def get_obj_data(root: str) -> Tuple[dict,dict,dict]:
    '''
    Returns complete information about a dataset point clouds
    '''

    obj_models, obj_diams, obj_symm = dict(), dict(), dict()

    with open(join(root,'obj_models','real_test','models_info.json')) as f:
        models_info = json.load(f)
    
    for obj_name, model_info in models_info.items():
        obj_models[obj_name] = get_obj_rendering(root, obj_name) #get_obj_pcd(root, split, obj_file)
        obj_diams[obj_name] = model_info['diameter']
        obj_symm[obj_name] = get_symmetry_transformations(model_info, max_sym_disc_step=0.05)
        
    return (obj_models, obj_diams, obj_symm)

def get_obj_pcd(root: str, filename: str) -> np.ndarray:
    '''
    Returns the point cloud model of a NOCS object
    '''

    xyz = list()
    with open(join(root,'obj_models','real_test',f'{filename}')) as f:
        lines = [point.split(' ') for point in f.readlines()]
    
    xyz = [[float(line[0]), float(line[1]), float(line[2])] for line in lines]
    return np.asarray(xyz)

def make_detections(root: str, scene_id: int, img_id: int):
    '''
    Makes 2D detections annots for given instance
    '''

    mask = np.asarray(Image.open(join(root, f'scene_{scene_id}/{img_id:04d}_mask.png')).convert('L'))
    classes = np.unique(mask)
    with open(join(root, f'scene_{scene_id}/{img_id:04d}_meta.txt'), 'r') as f:
        lines = f.readlines()
        # list of mask ids 
        mask_ids = [int(line.split(' ')[0]) for line in lines]

    with open(join(root, f'scene_{scene_id}/{img_id:04d}_meta.txt'), 'w') as fm:
        with open(join(root, f'scene_{scene_id}/{img_id:04d}_detection.txt'), 'w') as f:
            for mask_idx, mask_id in enumerate(mask_ids):
                if mask_id in classes:
                    ys, xs = np.nonzero(mask == mask_id)
                    x, y = np.min(xs), np.min(ys)
                    w, h = np.max(xs) - x, np.max(ys) - y
                    # x,y,w,h = nodef_crop((x,y,w,h))
                    # if x < 0 or y < 0 or x+w > 639 or y+h > 479:
                    #    print("Watch out for ", join(root, f'scene_{scene_id}/{img_id:04d}_detection.txt'))
                    f.write(f'{mask_id} {x} {y} {w} {h}\n')
                    fm.write(lines[mask_idx])

def get_item_metadata(root: str, scene_id: int, img_id: int, pose_annots: dict, obj_names:dict, obj_name:Optional[str] = None) -> dict:
    '''
    Return NOCS dataset metadata given root, and ids of scene and image
    '''

    poses = list()
    for pose in pose_annots[f'{scene_id}_{img_id}']:
        # NOCS include scale, which must be removed!
        new_pose = pose.copy()
        new_pose[:3,:3] = new_pose[:3,:3] / np.linalg.norm(new_pose[:3,:3],axis=1)
        poses.append(new_pose)
    
    cls_ids, mask_ids, cls_names, cls_descs, dets = list(), list(), list(), list(), list()
    with open(join(root, 'split/real_test', f'scene_{scene_id}/{img_id:04d}_meta.txt')) as fm:

        det_file = join(root, 'split/real_test', f'scene_{scene_id}/{img_id:04d}_detection.txt')

        with open(det_file) as fd:
            for i, (meta_line, det_line) in enumerate(zip(fm.readlines(), fd.readlines())):
                mask_id, cls_id, cur_obj_name = meta_line.split(' ')
                cur_obj_name = cur_obj_name.strip()
                # can optionaly return only the given cat info
                if obj_name is not None:
                    if cur_obj_name != obj_name:
                        continue
                    else:
                        poses = [poses[i]]
                    
                cls_ids.append(int(cls_id))
                mask_ids.append(int(mask_id))
                # ignore "norm" which is the last token
                cls_name = obj_names[cur_obj_name][0]
                cls_desc = obj_names[cur_obj_name][1:]
                cls_names.append(cls_name)
                cls_descs.append(cls_desc)
                # get detection, ignore first value as it is the id
                x, y, w, h = [int(x) for x in det_line.split(' ')[1:]]
                dets.append((x, y, w, h))

    return {
        'cls_ids': cls_ids,
        'mask_ids': mask_ids,
        'cls_names': cls_names,
        'cls_descs': cls_descs,
        'poses': poses,
        'boxes': dets,
    }

def get_item_data(root: str, scene_id: int, img_id: int, pose_annots: dict, obj_names: dict, obj_name:Optional[str] = None, mask_type:Optional[str]='oracle', hf_depth:bool = False) -> dict:
    '''
    Return NOCS data given root and ifs of image and scene
    '''

    metadata = get_item_metadata(root, scene_id, img_id, pose_annots, obj_names, obj_name)
    
    base_path = join(root, 'split/real_test', f'scene_{scene_id}/{img_id:04d}') 
    img = np.asarray(Image.open(base_path + '_color.png').convert('RGB'))
    if mask_type == 'oracle':
        mask = np.asarray(Image.open(base_path + '_mask.png').convert('L'))
    elif mask_type == 'ovseg':
        # this is already in nocs format
        mask = np.asarray(Image.open(base_path + '_pred_mask.png').convert('L'))
    elif mask_type == 'san':
        # this is 1 for class and 255 in the others
        path = join(root, 'san_name', f'{scene_id} {img_id} {obj_name}.png')
        mask = np.asarray(Image.open(path).convert('L'))
        mask_id = metadata['mask_ids'][0]
        # convert to NOCS format
        mask = np.where(mask == 1, mask_id, 255)
    elif mask_type == 'oryon':
        # this is 1 for class and 255 in the others
        path = join(root, 'oryon', f'{scene_id} {img_id} {obj_name}.png')
        mask = np.asarray(Image.open(path).convert('L'))
        mask_id = metadata['mask_ids'][0]
        # convert to NOCS format
        mask = np.where(mask == 1, mask_id, 255)
    else:
        raise RuntimeError(f'Mask type {mask_type} not implemented.')
    
    if hf_depth:
        depth = np.asarray(Image.open(base_path + '_hfdepth.png'))
    else:
        depth = np.asarray(Image.open(base_path + '_depth.png'))
    
    instance_id = f'{scene_id} {img_id} {obj_name}'
    
    data = {
        'rgb': img,
        'mask': mask,
        'depth': depth,
        'metadata': metadata,
        'instance_id': instance_id
    }

    #if os.path.exists(base_path + '_cp_featmap.npy'):
    #    featmap = np.load(base_path + '_cp_featmap.npy').transpose(2, 0, 1)
    #    data['featmap'] = featmap
    
    return data

def get_pcd(root : str, split : str, scene_id : int, img_id : int) -> dict:
    '''
    Get NOCS data of a single image in point cloud form, return in Meters
    '''
    K = np.asarray([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])
    id = join(root, 'split', split, f'scene_{scene_id}', f'{img_id:04d}')
    rgb = Image.open(f'{id}_color.png').convert('RGB')
    depth = Image.open(f'{id}_depth.png')
    mask = Image.open(f'{id}_mask.png')
    nocs = Image.open(f'{id}_coord.png')
    
    rgb = torch.tensor(np.asarray(rgb) / 255.)
    nocs = torch.tensor(np.asarray(nocs) / 255.)
    depth = torch.tensor(np.asarray(depth)).unsqueeze(2)
    mask = torch.tensor(np.asarray(mask)).unsqueeze(2)

    H,W = mask.shape[:2]
    xs = torch.linspace(0, W-1, steps=W)
    ys = torch.linspace(0, H-1, steps=H)
    xmap, ymap = torch.meshgrid(xs,ys, indexing='xy')
    xmap, ymap = xmap.unsqueeze(-1), ymap.unsqueeze(-1)

    to_lift = torch.cat((depth, rgb, nocs, mask, ymap, xmap),dim=-1)
    pcd = lift_pcd(to_lift, torch.tensor(K).flatten())

    return {
        'xyz' : pcd[:,:3] / 1000.,
        'rgb' : pcd[:,3:6],
        'nocs' : pcd[:,6:9],
        'mask' : pcd[:,9],
        'yx_map' : pcd[:,10:]
    }

def filter_pcd(nocs_pcd : dict, mask_idx : int) -> dict:
    '''
    Crops a NOCS point cloud to retain only a specified object
    '''
    idxs = torch.nonzero(nocs_pcd['mask'] == mask_idx).squeeze()

    return {
        'xyz' : nocs_pcd['xyz'][idxs],
        'rgb' : nocs_pcd['rgb'][idxs],
        'nocs' : nocs_pcd['nocs'][idxs],
        'mask' : nocs_pcd['mask'][idxs],
        'yx_map' : nocs_pcd['yx_map'][idxs],
    }
