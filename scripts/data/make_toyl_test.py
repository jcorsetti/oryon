import os
import sys
from os.path import join
sys.path.append(os.getcwd())
import torch
import hashlib
import numpy as np
from os import mkdir, readlink
import pickle
from typing import Tuple, Union
from torch import Tensor
from utils.data import common, toyl
#import open3d as o3d
import argparse

def parse_test_args():
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument('--root', type=str, default='data/toyl', help='Root of dataset')
    parser.add_argument('--src_split', type=str, default='test', help='Split used to get images')
    parser.add_argument('--dest_split', type=str, default='overfit_self', help='Split to be created')
    parser.add_argument('--exclude_split', type=str, default=None, help='If set, exclude samples in this split')
    parser.add_argument('--n_elems', type=int, default=5, help='Number of pairs to generate')
    parser.add_argument('--scene_type', type=str, default='self', help='None, different or same')
    
    args = parser.parse_args()
    return args


def np_transform(g: np.ndarray, pts: np.ndarray):
    """ Applies the SE3 transform

    Args:
        g: SE3 transformation matrix of size ([B,] 3/4, 4)
        pts: Points to be transformed ([B,] N, 3)

    Returns:
        transformed points of size (N, 3)
    """
    rot = g[..., :3, :3]  # (3, 3)
    trans = g[..., :3, 3]  # (3)

    transformed = pts[..., :3] @ np.swapaxes(rot, -1, -2) + trans[..., None, :]
    return transformed

def pcd_correspondences(feats1: Tensor, feats2: Tensor, threshold: float, max_corrs: int) -> Tuple[Tensor,Tensor]:
    '''
    Correspondences between point cloud features
    '''
    feats1_idxs = torch.arange(0, feats1.shape[0])
    feats2_idxs = torch.arange(0, feats2.shape[0])
    SAMPLE = 20000

    # filter if necessary to avoid out of memory
    if feats1.shape[0] >= SAMPLE:
        uniform_dist = torch.ones(feats1.shape[0], dtype=float).to(feats1.device)
        choosen = torch.multinomial(uniform_dist, SAMPLE, replacement=False)
        feats1_idxs = feats1_idxs[choosen]
        feats1 = feats1[choosen]
    
    if feats2.shape[0] >= SAMPLE:
        uniform_dist = torch.ones(feats2.shape[0], dtype=float).to(feats2.device)
        choosen = torch.multinomial(uniform_dist, SAMPLE, replacement=False)
        feats2_idxs = feats2_idxs[choosen]
        feats2 = feats2[choosen]
    
    dist = torch.cdist(feats1.double(), feats2.double(), p=2)
    min_dist = torch.amin(dist, dim=1)

    # these are the points from points2 selected as NN
    points2_idxs = torch.argmin(dist, dim=1)
    # additional filter to see which ones respect the threshold
    valid_corr = torch.nonzero(min_dist <= threshold)
    # get roi2 choosen as minimum
    feats1_idxs = feats1_idxs[valid_corr].squeeze(1)
    feats2_idxs = feats2_idxs[points2_idxs[valid_corr]].squeeze(1)
    
    if valid_corr.shape[0] > max_corrs:
        uniform_dist = torch.ones(valid_corr.shape[0], dtype=float).to(valid_corr.device)
        choosen = torch.multinomial(uniform_dist, max_corrs, replacement=False)
        feats1_idxs = feats1_idxs[choosen]
        feats2_idxs = feats2_idxs[choosen]

    return (feats1_idxs.cpu(), feats2_idxs.cpu())

def make_toyl_test(root: str, src_split : str, dest_split : str, n_elems : int, scene_type : Union['str',None], exclude_set : list = None):
    '''
    Samples a set of view pairs to be used as test partitions. 
    Relative pose is computed by RANSAC on NOCS map. 
    '''

    poses, mask_idxs, scenes, images, objects = list(), list(), list(), list(), list()

    # make list of all object instances within a NOCS split
    gt_dict = dict()

    annots = toyl.get_part_data(root, src_split)
    names = toyl.get_obj_names(root)

    for k, data in annots.items():

        scene_id, img_id = k.split('_')
        scene_id, img_id = int(scene_id), int(img_id)

        for obj_k in data.keys():
            scenes.append(scene_id)
            images.append(img_id)
            objects.append(int(obj_k))
            mask_idxs.append(data[obj_k]['mask_idx'])
            poses.append(data[obj_k]['pose'])

    scenes = np.asarray(scenes)
    images = np.asarray(images)
    objects = np.asarray(objects)
    mask_idxs = np.asarray(mask_idxs)

    #58, 1224, 48, 1662, 8
    
    mkdir(join(root,'fixed_split',dest_split))
    f = open(join(root,'fixed_split',dest_split,'instance_list.txt'),'w')
    
    i = 0
    fail_i = 0
    MAX_FAIL = 200000

    choosen = set()
    while i < n_elems and fail_i < MAX_FAIL:
        
        cur_len = scenes.shape[0]
        # choose a random instance as A
        idx_a = np.random.randint(0, cur_len)
        # get its object id
        obj_id = objects[idx_a]
        # get img and scene id
        scene_a, image_a = scenes[idx_a], images[idx_a]

        # standard : select among indexes with same object
        idxs_pool = objects == obj_id
        same_scene = scenes == scene_a
        idxs_pool = np.logical_and(idxs_pool, same_scene)
    
        if np.count_nonzero(idxs_pool) > 0:
            idx_q = np.random.choice(np.nonzero(idxs_pool)[0])
        else:
            print("Nothing found for {scene_a} {image_a}")
            continue
        
        # skip if I got the same object
        if idx_a == idx_q:
            fail_i += 1
            continue

        scene_q, image_q = scenes[idx_q], images[idx_q]

        # scene_q and scene_a are the same, but image_a and image_q will have to be more than 10 idxs far away
        if abs(int(image_q)-int(image_a) < 10):
            continue

        if exclude_set is not None:
            if (src_split, int(scene_a), int(image_a), int(obj_id)) in exclude_set:
                continue
            if (src_split, int(scene_q), int(image_q), int(obj_id)) in exclude_set:
                continue

        instance_id_a = f'{scene_a} {image_a} {objects[idx_a]}'
        instance_id_q = f'{scene_q} {image_q} {objects[idx_q]}'
        total_instance = f'{scene_a} {image_a} {scene_q} {image_q} {objects[idx_a]}'

        # skip if any of the element have already been choosen
        if (total_instance in choosen):
            fail_i += 1
            continue

        pcd_a = toyl.get_pcd(root, src_split, scene_a, image_a)
        pcd_q = toyl.get_pcd(root, src_split, scene_q, image_q)
        pcd_a = toyl.filter_pcd(pcd_a, mask_idxs[idx_a])
        pcd_q = toyl.filter_pcd(pcd_q, mask_idxs[idx_q])

        if pcd_a['xyz'].shape[0] == 0:
            print("Idx used: ", mask_idxs[idx_a])
            print("Problem with ", instance_id_a)
            exit(0)
            continue
        if pcd_q['xyz'].shape[0] == 0:
            print("Idx used: ", mask_idxs[idx_q])
            print("Problem with ", instance_id_q)
            exit(0)
            continue

        item_a = toyl.get_item_data('data/toyl','test',scene_a, image_a, annots, names, obj_id)
        item_q = toyl.get_item_data('data/toyl','test',scene_q, image_q, annots, names, obj_id)
        
        if len(item_a['metadata']['boxes']) == 0 or len(item_q['metadata']['boxes']) == 0:
            print("Trouble with pair ", instance_id_a, instance_id_q)
            exit(0)

        
        pose_a = poses[idx_a]
        pose_q = poses[idx_q]
        '''
        o3d_pcd_a = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_a['xyz']))
        o3d_pcd_a.paint_uniform_color([1.,0.,0.])

        o3d_pcd_q = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_q['xyz']))
        o3d_pcd_q.paint_uniform_color([0.,1.,0.])

        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0., 0., 0.])
        o3d.visualization.draw_geometries([o3d_pcd_a, o3d_pcd_q, mesh])
        '''
        
        pose_aq = pose_q @ np.linalg.inv(pose_a) # obtained pose A->Q
        xyz_q_inv = np_transform(pose_aq, pcd_a['xyz'])
        xyz_q = pcd_q['xyz']
        
        '''
        o3d_pcd_inv_q = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz_q_inv))
        o3d_pcd_inv_q.paint_uniform_color([0.,0.,1.])
        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0., 0., 0.])
        o3d.visualization.draw_geometries([o3d_pcd_a, o3d_pcd_q, o3d_pcd_inv_q, mesh])
        '''
        
        idxs_a, idxs_q = pcd_correspondences(xyz_q_inv, xyz_q, threshold=0.002, max_corrs=10000)
        # skip if low number of correspondences
        if idxs_a.shape[0] < 100:
            print("Not enough corrs: ", idxs_a.shape[0])
            fail_i += 1
            continue
        
        yx_a, yx_q = pcd_a['yx_map'][idxs_a], pcd_q['yx_map'][idxs_q]

        yx_corrs = torch.cat([yx_a, yx_q],dim=1)
        pair_id = '_'.join([str(elem).strip('\n') for elem in [scene_a,image_a,scene_q,image_q, obj_id]])

        f.write('{}, {} {}, {} {}, {}\n'.format(src_split,
            scene_a,image_a,scene_q,image_q, obj_id
        ))

        pose_aq[:3,3] = pose_aq[:3,3] * 1000.
        gt_dict[pair_id] = {
            'gt' : pose_aq,
            'corrs' : yx_corrs.numpy()
        }

        choosen.add(total_instance)
        i += 1

    if fail_i >= MAX_FAIL:
        print(f"Stopped at {i} elements due to failures D:")
#    else:
    f = open(join(root,'fixed_split',dest_split,'annots.pkl'),'wb')
    pickle.dump(gt_dict, f)


if __name__ == '__main__':

    args = parse_test_args()
    
    #exclude_set = None
    #if args.exclude_split is not None:
    #    exclude_set = nocs.get_partition_instances(args.root, args.exclude_split)

    print("Using exclude set ", args.exclude_split)
    make_toyl_test(args.root, args.src_split, args.dest_split, n_elems=args.n_elems, scene_type=args.scene_type)