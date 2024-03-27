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
from utils.data import nocs
import open3d as o3d
import argparse

def parse_test_args():
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument('--root', type=str, default=readlink('data/nocs'), help='Root of dataset')
    parser.add_argument('--src_split', type=str, default='real_test', help='Split used to get images')
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

def make_nocs_test(root: str, src_split : str, dest_split : str, n_elems : int, scene_type : Union['str',None], exclude_set : list = None):
    '''
    Samples a set of view pairs to be used as test partitions. 
    Relative pose is computed by RANSAC on NOCS map. 
    '''

    mask_idxs, scenes, images, objects, prompts = list(), list(), list(), list(), list()

    # make list of all object instances within a NOCS split
    gt_dict = dict()
    prompt_dict = dict()

    with open(join(root,'split', src_split, 'instance_list.txt')) as f:
        instances = f.readlines()

    for instance in instances:

        scene_id, img_id = instance.split(' ')

        with open(join(root,'split',src_split,f'scene_{int(scene_id)}',f'{int(img_id):04d}_meta.txt')) as f:

            meta = f.readlines()
            for obj_meta in meta:
                mask_idx, cat_id, cat_prompt = obj_meta.split(' ')
                #if int(cat_id) in [1]:
                scenes.append(int(scene_id))
                images.append(int(img_id))
                objects.append(int(cat_id))
                mask_idxs.append(int(mask_idx))
                hash = int(hashlib.sha256(cat_prompt.encode('utf-8')).hexdigest(), 16) % 10**8
                prompts.append(hash)

                if hash not in prompt_dict.keys():
                    prompt_dict[hash] = cat_prompt

    scenes = np.asarray(scenes)
    images = np.asarray(images)
    objects = np.asarray(objects)
    prompts = np.asarray(prompts)
    mask_idxs = np.asarray(mask_idxs)

    mkdir(join(root,'fixed_split',dest_split))
    f = open(join(root,'fixed_split',dest_split,'instance_list.txt'),'w')
    i = 0
    fail_i = 0
    MAX_FAIL = 200000

    choosen = set()
    while i < n_elems and fail_i < MAX_FAIL:
        
        # choose a random instance as A and a random instance as Q
        cur_len = scenes.shape[0]
        idx_a = np.random.randint(0, cur_len)
        obj_hash = prompts[idx_a]
        obj_id = objects[idx_a]
        # the prompt (i.e., object) must be the same

        scene_a, image_a = scenes[idx_a], images[idx_a]

        # standard : select among indexes with same prompt (i.e. object instance)
        idxs_pool = prompts == obj_hash
        if scene_type is not None:
            if scene_type == 'same':
                # with this setting, restrict to same scene
                same_scene = scenes == scene_a
                idxs_pool = np.logical_and(idxs_pool, same_scene)
            elif scene_type == 'different':
                # with this setting, restrinct to different scene
                diff_scene = scenes != scene_a
                idxs_pool = np.logical_and(idxs_pool, diff_scene)

        if np.count_nonzero(idxs_pool) > 0:
            idx_q = np.random.choice(np.nonzero(idxs_pool)[0])
        else:
            print(f"Apparently object {prompt_dict[obj_hash]} is only in scene {scene_a}")
            continue
        
        # skip if I got the same object
        if idx_a == idx_q:
            fail_i += 1
            continue

        scene_q, image_q = scenes[idx_q], images[idx_q]

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

        pcd_a = nocs.get_pcd(root, src_split, scene_a, image_a)
        pcd_q = nocs.get_pcd(root, src_split, scene_q, image_q)
        pcd_a = nocs.filter_pcd(pcd_a, mask_idxs[idx_a])
        pcd_q = nocs.filter_pcd(pcd_q, mask_idxs[idx_q])

        if pcd_a['xyz'].shape[0] == 0:
            print("Problem with ", instance_id_a)
            continue
        if pcd_q['xyz'].shape[0] == 0:
            print("Problem with ", instance_id_q)
            continue

        try:
            f_a = open(f'{root}/gts/{src_split}/results_{src_split}_scene_{scene_a}_{image_a:04d}.pkl','rb')
            pose_a = pickle.load(f_a)['gt_RTs'][mask_idxs[idx_a]-1].copy()
            #pose_a[:3,3] = pose_a[:3,3] * 1000
        except:
            print("Problem with {}".format(f'{root}/gts/{src_split}/results_{src_split}_scene_{scene_a}_{image_a:04d}.pkl', mask_idxs, idx_a))
            continue
        try:            
            f_q = open(f'{root}/gts/{src_split}/results_{src_split}_scene_{scene_q}_{image_q:04d}.pkl','rb')
            pose_q = pickle.load(f_q)['gt_RTs'][mask_idxs[idx_q]-1].copy()
            #pose_q[:3,3] = pose_q[:3,3] * 1000
        except:
            print("Problem with {}".format(f'{root}/gts/{src_split}/results_{src_split}_scene_{scene_q}_{image_q:04d}.pkl', mask_idxs, idx_a))
            continue
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
        print(idxs_a.shape)
        # skip if low number of correspondences
        if idxs_a.shape[0] < 100:
            fail_i += 1
            continue
        
        yx_a, yx_q = pcd_a['yx_map'][idxs_a], pcd_q['yx_map'][idxs_q]
        obj_name = prompt_dict[obj_hash]

        yx_corrs = torch.cat([yx_a, yx_q],dim=1)
        pair_id = '_'.join([str(elem).strip('\n') for elem in [scene_a,image_a,scene_q,image_q, obj_id, obj_name]])

        f.write('{}, {} {}, {} {}, {} {}'.format(src_split,
            scene_a,image_a,scene_q,image_q, obj_id, obj_name
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
    
    exclude_set = None
    if args.exclude_split is not None:
        exclude_set = nocs.get_partition_instances(args.root, args.exclude_split)

    print("Using exclude set ", args.exclude_split)
    make_nocs_test(args.root, args.src_split, args.dest_split, n_elems=args.n_elems, scene_type=args.scene_type, exclude_set=exclude_set)