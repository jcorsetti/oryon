import os
import sys
sys.path.append(os.getcwd())
import csv
import json
import numpy as np
from typing import Tuple
from os.path import join
from PIL import Image
import pickle
from utils.pcd import get_diameter

def get_instance2cat_id(root: str) -> dict:
    with open(f'{root}/instance2cat.json') as f:
        return json.load(f)

def get_partition_instances(root: str, split: str) -> set:
    '''
    Returns a set of instances used in a fixed partition
    '''

    part_set = set()
    with open(join(root,'fixed_split',split,'instance_list.txt')) as f:
        lines = f.readlines()

    for line in lines:

        img_a, img_q, cat_id = line.strip('\n').split(',')
        part_set.add((int(img_a), int(cat_id)))
        part_set.add((int(img_q), int(cat_id)))

    return part_set

def load_annotations(root : str) -> dict:
    '''
    Loads annotations
    '''
    f = open(join(root,'scene_gt.pkl'),'rb')
    return pickle.load(f)

def load_object_splits(root : str) -> dict:
    '''
    Loads object splits
    '''
    with open(join(root,'object_split.json')) as f:
        return json.load(f)

def get_metadata(root : str) -> Tuple[dict,dict,dict]:

    cat_map = dict()    # map each shapenet object id to a set of textual info
    id_new2old = dict() # maps each new id to the respective shapenet id

    with open(join(root,'metadata.csv'),'r') as f:
        data = csv.reader(f)

        for i, tokens in enumerate(data):
            if i > 0:
                obj_id = tokens[0].split('.')[1]
                cat_id = tokens[2]
                lemmas = tokens[3].split(',')
                obj_name = tokens[-2]

                cat_map[obj_id] = {
                    'obj_id' : obj_id,
                    'cat_id' : cat_id,
                    'obj_syn' : lemmas,
                    'obj_name' : obj_name
                }

    with open(join(root,'objnm2clsid.json')) as f:
        data = json.load(f)
        id_new2old = {new:old.split('_')[0] for old, new in data.items()}    

    with open(join(root,'obj2img.json')) as f:
        id_occ = json.load(f)
        
    return (cat_map, id_new2old, id_occ)

def get_item_data(root: str, annots: dict, metadata: Tuple, img_id: int, cat_id: int=None) -> dict:
    '''
    Returns ShapeNet item
    '''
    cat_map, id_new2old, _ = metadata

    img = np.asarray(Image.open(join(root,'raw_data','rgb',f'{img_id:06d}.jpg')).convert('RGB'))
    mask = np.asarray(Image.open(join(root,'raw_data','mask',f'{img_id:06d}.png')).convert('L'))
    depth = np.asarray(Image.open(join(root,'raw_data','depth',f'{img_id:06d}.png')))

    img_annot = annots[img_id]
    camera = img_annot['K']
    instance_id = f'{img_id} {cat_id}'
    
    cls_ids, mask_ids, cls_names, cls_descs, boxes, poses = list(), list(), list(), list(), list(), list()

    for obj_idx, obj_annot in enumerate(img_annot['obj_info_lst']):

        # skip empty annotations
        if len(obj_annot.keys()) > 0:

            # can optionally return only the given cat info
            obj_id = int(obj_annot['cls_id'])
            if cat_id is not None:
                if obj_id != cat_id:
                    continue

            # index of object is also mask value on the segm mask
            ys, xs = np.nonzero(mask == obj_idx)
            y,x = np.min(ys), np.min(xs)
            h,w = np.max(ys) - y, np.max(xs) - x

            lemmas = cat_map[id_new2old[obj_id]]['obj_syn']
            cls_ids.append(obj_annot['cls_id'])
            mask_ids.append(obj_idx)
            cls_names.append(lemmas[0])
            cls_descs.append(lemmas)
            boxes.append((x,y,w,h))
            pose = np.eye(4)
            pose[:3,:] = obj_annot['RT']
            poses.append(pose)

    metadata = {
        'cls_ids' : cls_ids,
        'mask_ids' : mask_ids,
        'cls_names' : cls_names,
        'cls_descs' : cls_descs,
        'poses': poses,
        'boxes' : boxes
    }

    data = {
        'rgb' : img,
        'mask' : mask,
        'depth' : depth,
        'camera' : camera,
        'metadata' : metadata,
        'instance_id' : instance_id
    }        

    if os.path.exists(join(root,'raw_data','featmap',f'{img_id:06d}.npz')):
        featmap = np.load(join(root,'raw_data','featmap',f'{img_id:06d}.npz'))
        data['featmap'] = featmap['arr_0'].astype(np.float32)

    return data

def get_obj_info(root: str, obj_id : str) -> Tuple:
    '''
    Return info about an object model
    '''
    
    xyz = []
    with open(join(root,'raw_data/models',obj_id+'.obj')) as f:
        lines = [line[2:].strip('\n') for line in f.readlines() if line.startswith('v ')]

    for line in lines:
        x,y,z = line.split(' ')
        xyz.append([float(x), float(y), float(z)])

    pcd = np.asarray(xyz) / 1000.
    diam = get_diameter(pcd)

    return pcd, diam, False




