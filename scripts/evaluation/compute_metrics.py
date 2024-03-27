import os
import sys
sys.path.append(os.getcwd())
from omegaconf import DictConfig, OmegaConf, open_dict
import numpy as np
from os.path import join
#import open3d as o3d
from utils.evaluator import Evaluator
from datasets import NOCSDataset, TOYLDataset
import torch
from typing import Optional
import json

def dict_from_preds(perf_file):
    '''
    Produces a dictionart from a prediction file
    '''    
    preds = dict()
    ious_a = dict()
    ious_q = dict()
    
    with open(perf_file,'r') as f:
        preds_lines = f.readlines()

    iou_present = True
    for line in preds_lines:
        tokens = line.split(',')
        if len(tokens) == 3:
            id_a, id_q, pose_line = tokens
            iou_present = False
        elif len(tokens) == 5:
            iou_present = True
            id_a, id_q, pose_line, iou_a, iou_q = tokens
        else:
            raise RuntimeError(' Anomaly in line: ' + line)
        
        scene_a, img_a, obj_a = id_a.split(' ') 
        scene_q, img_q, _ = id_q.split(' ')
        pose = np.asarray([float(p) for p in pose_line.split(' ')]).reshape(3,4)
        new_id = '{}_{}_{}_{}_{}'.format(scene_a, img_a, scene_q, img_q, obj_a)
        
        preds[new_id] = pose
        if iou_present:
            ious_a[new_id] = float(iou_a)
            ious_q[new_id] = float(iou_q)
    
    return preds, ious_a, ious_q, iou_present

def compute_metrics(results_file: str, print_summary: bool = False, out_file: Optional[str] = None):

    metric_file = os.path.splitext(results_file)[0] + '.json'

    res_path, res_file = os.path.split(results_file)
    res_key = 'config_' + '_'.join(os.path.splitext(res_file)[0].split('_')[-3:]) + '.yaml'
    config_path = os.path.join(res_path,res_key)
    args = OmegaConf.load(config_path)
    preds, ious_a, ious_q, iou_present = dict_from_preds(results_file)
    
    if not iou_present:
        print(f'IoU not found in prediction file {results_file}.')

    with open_dict(args):
        print(args.test)
        if 'mask' not in args.test.keys():
            args.test.mask = 'predicted'
            if 'GT' in args.exp_tag:
                args.test.mask = 'oracle'
            elif 'pred' in args.exp_tag:
                args.test.mask = 'predicted'
        if 'hf_depth' not in args.test.keys():
            args.test.hf_depth = False
        if 'prior_crop' not in args.test.keys():
            args.test.prior_crop = False

    args.test.prior_crop=False
    if 'nocs' in results_file:
        dataset = NOCSDataset(args, eval=True)
    elif 'toyl' in results_file:
        dataset = TOYLDataset(args, eval=True)
    else:
        raise RuntimeError("Dataset not supported")
    
    evaluator = Evaluator(exp_tag=args.exp_tag, compute_vsd=True, compute_iou=iou_present)
    evaluator.init_test()
    evaluator.add_object_info(*dataset.get_object_info())
    for idx in range(dataset.__len__()):
        item_a, item_q, _, _, _, gt_pose, cls_id, instance_id, valid = dataset.__getitem__(idx)

        gt_q = item_q['metadata']['poses'][0].numpy()
        gt_a = item_a['metadata']['poses'][0].numpy()
        pred_pose_rel = preds[instance_id]
        
        pred_pose_rel = np.concatenate([pred_pose_rel,np.asarray([[0.,0.,0.,1.]])],axis=0)
        pred_q = pred_pose_rel @ gt_a

        if valid:
            result = {
                'gt_pose': torch.tensor(gt_q).unsqueeze(0),
                'pred_pose': torch.tensor(pred_q).unsqueeze(0),
                'pred_pose_rel': torch.tensor(pred_pose_rel).unsqueeze(0),
                'cls_id': [cls_id],
                'instance_id': [instance_id],
                'camera': [item_q['camera'].numpy()],
                'depth': [item_q['orig_depth'].squeeze().numpy()]
            }
            if iou_present:
                result['iou_a'] = torch.tensor(ious_a[instance_id]).unsqueeze(0)
                result['iou_q'] = torch.tensor(ious_q[instance_id]).unsqueeze(0)
                
            evaluator.register_test(result)
        else:
            evaluator.register_test_failure({
                'cls_id': [cls_id],
                'instance_id': [instance_id]
            })

    if print_summary:
        print(evaluator.test_summary())

    latex_str = evaluator.get_latex_str()
    if out_file is None:
        print(latex_str)
    else:
        with open(out_file, 'a') as f:
            f.write(latex_str)
    # if VSD and AR have not been computed previously, override them
    with open(metric_file,'w') as f:
        evaluator.save(f)



def main():

    path = sys.argv[1]
    # called from function: print to file, no summary
    if len(sys.argv) > 2:
        outfile = sys.argv[2]
        print_summary=False
    # called from scratch: print to stdout, and summary
    else:
        outfile = None
        print_summary = True

    compute_metrics(path, print_summary, outfile)
    
if __name__ == '__main__':
    main()
