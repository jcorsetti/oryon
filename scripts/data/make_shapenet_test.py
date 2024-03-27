import os
import sys
from os.path import join
sys.path.append(os.getcwd())
import torch
import numpy as np
from utils.pcd import lift_pcd
from os import mkdir
import pickle
from typing import Tuple
from torch import Tensor
from utils.data import shapenet

import argparse
import os
from utils.misc import boolean_string
from os import readlink

def parse_test_args():
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument('--root', type=str, default='data/shapenet6d', help='Root of dataset')
    parser.add_argument('--dest_split', type=str, default=None, help='Split to be created')
    parser.add_argument('--exclude_split', type=str, default=None, help='If set, exclude samples in this split')
    parser.add_argument('--n_elems', type=int, default=None, help='Number of pairs to generate')
    
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


def make_shapenet_test(root: str, dest_split : str, n_elems : int, exclude_set : list = None ):
    '''
    Samples a set of view pairs to be used as test partitions. 
    Relative pose is computed by RANSAC on NOCS map. 
    '''

    # make list of all object instances within a NOCS split
    gt_dict = dict()

    # load a bunch of annotations and metadata
    object_split = shapenet.load_object_splits(root)['all']
    annots = shapenet.load_annotations(root)

    metadata = shapenet.get_metadata(root)
    _,_, id2img = metadata
    
    obj_ids = [int(k) for k in id2img.keys() if int(k) in object_split]   # list of object ids

    mkdir(join(root,'fixed_split',dest_split))
    f = open(join(root,'fixed_split',dest_split,'instance_list.txt'),'w')
    i = 0
    fail_i = 0
    MAX_FAIL = 200000

    print("Exclude set: ", exclude_set)
    choosen = set()
    while i < n_elems and fail_i < MAX_FAIL:
        
        # choose a random object, then two random images from the list of images with the object in it
        obj_id = 11131#np.random.choice(obj_ids) # chair id: 11131
        img_id_a = np.random.choice([0,1,2,3,117,3990])#id2img[str(obj_id)])
        img_id_q = np.random.choice([0,1,2,3,117,3990])#id2img[str(obj_id)])

        # skip if I got the same object
        if img_id_a == img_id_q:
            fail_i += 1
            continue

        if exclude_set is not None:
            print("Evaluating ", (int(img_id_a), int(obj_id)))
            if ((int(img_id_a), int(obj_id)) in exclude_set):
                print("Is in the set!")
                continue
            print("Evaluating ", (int(img_id_q), int(obj_id)))
            if ((int(img_id_q), int(obj_id)) in exclude_set):
                print("Is in the set!")
                continue

        instance_id_a = f'{img_id_a} {obj_id}'
        instance_id_q = f'{img_id_q} {obj_id}'
        total_instance = f'{img_id_a} {img_id_q} {obj_id}'

        # skip if any of the element have already been choosen
        if (total_instance in choosen):
            fail_i += 1
            continue
        
        item_a = shapenet.get_item_data(root, annots, metadata, img_id=img_id_a, cat_id=obj_id)
        item_q = shapenet.get_item_data(root, annots, metadata, img_id=img_id_q, cat_id=obj_id)

        pcd_a = shapenet.get_pcd(item_a)
        pcd_q = shapenet.get_pcd(item_q)
        # this is the position of the object in the list but also the mask value
        mask_idx_a = item_a['metadata']['mask_ids'][0]
        mask_idx_q = item_q['metadata']['mask_ids'][0]
        pcd_a = shapenet.filter_pcd(pcd_a, mask_idx_a)
        pcd_q = shapenet.filter_pcd(pcd_q, mask_idx_q)

        if pcd_a['xyz'].shape[0] == 0:
            print("Problem with ", instance_id_a)
        if pcd_q['xyz'].shape[0] == 0:
            print("Problem with ", instance_id_q)

        pose_a_ = annots[img_id_a]['obj_info_lst'][mask_idx_a]['RT'].copy()
        pose_a_[:3,3] = pose_a_[:3,3] * 1000
        pose_a = np.eye(4)
        pose_a[:3,:] = pose_a_

        pose_q_ = annots[img_id_q]['obj_info_lst'][mask_idx_q]['RT'].copy()
        pose_q_[:3,3] = pose_q_[:3,3] * 1000
        pose_q = np.eye(4)
        pose_q[:3,:] = pose_q_

        '''
        o3d_pcd_a = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_a['xyz']))
        o3d_pcd_a.paint_uniform_color([1.,0.,0.])

        o3d_pcd_q = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_q['xyz']))
        o3d_pcd_q.paint_uniform_color([0.,1.,0.])

        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0., 0., 0.])
        o3d.visualization.draw_geometries([o3d_pcd_a, o3d_pcd_q, mesh])
        '''
        
        pose_aq = pose_q @ np.linalg.inv(pose_a) # obtained pose A->Q
        
        xyz_q_inv = np_transform(pose_aq, pcd_a['xyz'])
        xyz_q = pcd_q['xyz']

        '''
        o3d_pcd_inv_q = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz_q_inv))
        o3d_pcd_inv_q.paint_uniform_color([0.,0.,1.])

        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0., 0., 0.])
        o3d.visualization.draw_geometries([o3d_pcd_a, o3d_pcd_q, o3d_pcd_inv_q, mesh])
        '''
        
        try:
            idxs_a, idxs_q = pcd_correspondences(xyz_q_inv, xyz_q, threshold=2.0, max_corrs=10000)
        except:
            print(xyz_q_inv.shape, xyz_q.shape)
            print("Error with {}".format(total_instance))
            continue

        # skip if low number of correspondences
        if idxs_a.shape[0] < 500:
            print(f'{total_instance} failed due to {idxs_a.shape[0]} correspondences!')
            fail_i += 1
            continue
        
        yx_a, yx_q = pcd_a['yx_map'][idxs_a], pcd_q['yx_map'][idxs_q]

        yx_corrs = torch.cat([yx_a, yx_q],dim=1)
        pair_id = '_'.join([str(elem) for elem in [img_id_a,img_id_q,obj_id]])

        f.write('{}, {}, {}\n'.format(img_id_a, img_id_q, obj_id))

        gt_dict[pair_id] = {
            'gt' : pose_aq,
            'corrs' : yx_corrs.numpy()
        }

        choosen.add(total_instance)
        i += 1

    if fail_i >= MAX_FAIL:
        print(f"Stopped at {i} elements due to failures D:")
    else:
        f = open(join(root,'fixed_split',dest_split,'annots.pkl'),'wb')
        pickle.dump(gt_dict, f)


if __name__ == '__main__':

    args = parse_test_args()
    
    exclude_set = None
    if args.exclude_split is not None:
        exclude_set = shapenet.get_partition_instances(args.root, args.exclude_split)

    print("Using exclude set ", args.exclude_split)

    make_shapenet_test(args.root, args.dest_split, n_elems=args.n_elems, exclude_set=exclude_set)