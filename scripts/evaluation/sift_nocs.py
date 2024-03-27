import os
import sys
sys.path.append(os.getcwd())
from utils.data import nocs
import numpy as np
import hydra
from omegaconf import DictConfig
from utils.pcd import project_points, np_transform_pcd
from utils.metrics import compute_RT_distances, get_entropy
from torch.nn.functional import interpolate
import numpy as np
import cv2 as cv

from datasets import NOCSDataset
from utils.evaluator import Evaluator
from utils.pointdsc.init import get_pointdsc_pose, get_pointdsc_solver
from utils.misc import torch_sample_select
from torch import Tensor
import torch
from utils.pcd import pdist
from utils.pcd import lift_pcd
from tqdm import tqdm
from PIL import Image

def nn_correspondences(feats1: Tensor, feats2: Tensor, kp1: Tensor, kp2: Tensor, threshold: float, max_corrs: int, subsample_source:int = None) -> Tensor:
    '''
    Finds matches between two [N,D] feature sets
    Return correspondences in shape (y1,x1,y2,x2)
    '''

        
    dist = pdist(feats1, feats2, 'inv_norm_cosine')
    min_dist = torch.amin(dist, dim=1)
    ro12_idxs = torch.argmin(dist, dim=1)
    valid_corr = torch.nonzero(min_dist < threshold)
    # get roi2 choosen as minimum
    kp2 = kp2[ro12_idxs]
    final_corrs = torch.cat((kp1[valid_corr.squeeze(1)], kp2[valid_corr.squeeze(1)]), dim=1)

    idxs = torch_sample_select(final_corrs, max_corrs)
    final_corrs = final_corrs[idxs]

    # these are in format (y1,x1,y2,x2)
    return final_corrs



@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def run_pipeline(args : DictConfig) -> None:

    args.dataset.test.name='nocs'
    args.dataset.test.split='cross_scene_test'
    dataset = NOCSDataset(args, eval=True)
    evaluator = Evaluator(f'NOCS SIFT ({args.test.mask})', compute_vsd=True, compute_iou=False)

    sift = cv.SIFT_create()
    if args.test.mask == 'ovseg':
        USE_PRED_SEGM = True
    else:
        USE_PRED_SEGM = False

    path = 'data/nocs/split/real_test'
    pointdsc_solver = get_pointdsc_solver(args.pretrained.pointdsc, args.device)
    
    K = torch.tensor(nocs.get_camera()).flatten()
    evaluator.add_object_info(*dataset.get_object_info())
    evaluator.init_test()
    
    print("Mask used: ", args.test.mask)
    fp = open(f'sift_nocs_{args.test.mask}.txt','w')
    for i in tqdm(range(dataset.__len__())):
        split, scene_a, img_a, scene_q, img_q, _, obj_name = dataset.instances[i]
        instance_id = f'{scene_a}_{img_a}_{scene_q}_{img_q}_{obj_name}'

        item_a = nocs.get_item_data(path, scene_a, img_a, dataset.abs_poses, dataset.obj_names, obj_name, USE_PRED_SEGM)
        item_q = nocs.get_item_data(path, scene_q, img_q, dataset.abs_poses, dataset.obj_names, obj_name, USE_PRED_SEGM)

        gt_q = torch.tensor(item_q['metadata']['poses'][0])
        gt_a = torch.tensor(item_a['metadata']['poses'][0])

        # go with SIFT

        gray_a = cv.cvtColor(item_a['rgb'], cv.COLOR_BGR2GRAY)
        gray_q = cv.cvtColor(item_q['rgb'], cv.COLOR_BGR2GRAY)

        # get kp and descriptor
        kp_a, feats_a = sift.detectAndCompute(gray_a, None)
        kp_q, feats_q = sift.detectAndCompute(gray_q, None)
        
        feats_a = np.asarray(feats_a)
        feats_q = np.asarray(feats_q)
        kp_a = np.asarray([key_point.pt for key_point in kp_a]).reshape(-1, 2)
        kp_q = np.asarray([key_point.pt for key_point in kp_q]).reshape(-1, 2)        
        kp_a = kp_a.astype(np.int16)
        kp_q = kp_q.astype(np.int16)
        
        # filter based on validity of the mask

        if len(item_a['metadata']['mask_ids']) > 0 and len(item_q['metadata']['mask_ids']) > 0:

            if args.masks == 'ours':
                mask_a = Image.open(f'data/toyl/catseg_masks/{scene_a} {img_a} {obj_name}.png')
                mask_a = np.asarray(mask_a)
                mask_q = Image.open(f'data/toyl/catseg_masks/{scene_q} {img_q} {obj_name}.png')
                mask_q = np.asarray(mask_q)
                item_a['mask'] = mask_a
                item_q['mask'] = mask_q
                mask_id_a = 1
                mask_id_q = 1
            else:
                mask_id_a = item_a['metadata']['mask_ids'][0]
                mask_id_q = item_q['metadata']['mask_ids'][0]
            mask = np.where(item_a['mask'] == mask_id_a, 1, 0)
            item_a['mask'] = mask
            mask = np.where(item_q['mask'] == mask_id_q, 1, 0)
            item_q['mask'] = mask
            
            valid_a = item_a['mask'][kp_a[:,1], kp_a[:,0]]
            valid_q = item_q['mask'][kp_q[:,1], kp_q[:,0]]
    
            # this may happen with predicted masks
            if np.count_nonzero(valid_a) > 0 and np.count_nonzero(valid_q) > 0:

                kp_a = torch.tensor(kp_a[valid_a == 1])
                kp_q = torch.tensor(kp_q[valid_q == 1])
                feats_a = torch.tensor(feats_a[valid_a == 1])
                feats_q = torch.tensor(feats_q[valid_q == 1])

                # return as XY, XY
                corrs = nn_correspondences(feats_a, feats_q, kp_a, kp_q, 0.25, 500)
                corrs_a, corrs_q = corrs[:,:2].to(torch.long), corrs[:,2:].to(torch.long)

                depth_a = torch.tensor(item_a['depth'])
                depth_q = torch.tensor(item_q['depth'])
                # functions lifts pcd in millimeters, here it is converted to meters
                pcd_a = lift_pcd(depth_a.unsqueeze(-1), K, (corrs_a[:,0],corrs_a[:,1])) / 1000.
                pcd_q = lift_pcd(depth_q.unsqueeze(-1), K, (corrs_q[:,0],corrs_q[:,1])) / 1000.
                pred_pose = get_pointdsc_pose(pointdsc_solver, pcd_a, pcd_q, args.device)

                pred_q = pred_pose @ gt_a.to(torch.float32)
                evaluator.register_test({
                    'gt_pose': gt_q.unsqueeze(0),
                    'pred_pose': pred_q.unsqueeze(0),
                    'pred_pose_rel': pred_pose.unsqueeze(0),
                    'cls_id': [obj_name],
                    'camera' : [K.cpu().numpy()],
                    'depth' : [depth_q.cpu().numpy()],
                    'instance_id': [instance_id]
                })

                pred_pose = ' '.join([str(n.item()) for n in pred_pose[:3,:].flatten()])
                line = ','.join([item_a['instance_id'], item_q['instance_id'], pred_pose])
                line += '\n'
                fp.write(line)

            else:
                print("Problem with pair ", item_a['instance_id'], item_q['instance_id'], ' : missing mask')
                print("I hope you are using predicted masks...")
                evaluator.register_test_failure({
                    'cls_id': [obj_name],
                    'instance_id': [instance_id]
                })
        else:
            print("Problem with pair ", item_a['instance_id'], item_q['instance_id'], ' : missing mask')
            print("I hope you are using predicted masks...")
            evaluator.register_test_failure({
                'cls_id': [obj_name],
                'instance_id': [instance_id]
            })
    
    fp.close()
    evaluator.test_summary()
    print(evaluator.get_latex_str())

if __name__ == '__main__':
    run_pipeline()
    