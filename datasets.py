import os
import json
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset
from omegaconf.dictconfig import DictConfig
from typing import Any, Dict, Tuple, Sequence, Union, List
from os.path import join
import torchvision
from torch.nn.functional import interpolate
from utils.data import nocs, shapenet, common, toyl
from utils.misc import torch_sample_select, unique_matches
from utils.augmentations import *
from torch import Tensor
from PIL import Image
from utils import viz

def set_seed(seed: int):
    print('SETTING SEED: ', seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

def get_mask_type(mask:str, eval:bool) -> str:
    '''
    Decodes mask type, one of oracle, predicted, ovseg, other possible priors
    '''
    res_mask = None
    # we are in evaluation
    if eval:
        # mask will be obtained with prediction.
        # setting oracle to get correct ground truth
        if mask == 'predicted':
            res_mask = 'oracle'
        # else just leave the input
        else:
            res_mask = mask
    # if not evaluating, is always oracle
    else:
        res_mask = 'oracle'
    
    return res_mask

def clone_batch(batch:dict) -> dict:
    '''
    Clones a batch
    '''
    new_batch = {
        'anchor' : dict(),
        'query' : dict()
    }
    
    for k,v in batch['anchor'].items():
        if isinstance(v, Tensor):
            new_batch['anchor'][k] = v.clone()
        elif isinstance(v, str):
            new_batch['anchor'][k] = v
        elif isinstance(v, list):
            new_batch['anchor'][k] = v.copy()
        else:
            raise RuntimeError(f"Unrecognized type {type(v)}")
    
    for k,v in batch['query'].items():
        if isinstance(v, Tensor):
            new_batch['query'][k] = v.clone()
        elif isinstance(v, str):
            new_batch['query'][k] = v
        elif isinstance(v, list):
            new_batch['query'][k] = v.copy()
        else:
            raise RuntimeError(f"Unrecognized type {type(v)}")

    for k,v in batch.items():
        if k != 'anchor' and k != 'query':
            if isinstance(v, Tensor):
                new_batch[k] = v.clone()
            elif isinstance(v, str):
                new_batch[k] = v
            elif isinstance(v, list):
                new_batch[k] = v.copy()
            else:
                raise RuntimeError(f"Unrecognized type {type(v)}")

    return new_batch

def build_test_preproc(args: DictConfig) -> torchvision.transforms.Compose:

    transform_list = list()
    transform_list.append(resize(args.dataset.img_size))

    if len(transform_list) == 0:
        transform_list.append(identity())
    return torchvision.transforms.Compose(transform_list)

def build_augs(args: DictConfig) -> torchvision.transforms.Compose:

    transform_list = list()
    if args.augs.rgb.jitter:
        transform_list.append(random_jitter(prob=0.5))
    if args.augs.rgb.bright:
        transform_list.append(random_brightness(prob=0.5))
    if args.augs.rgb.hflip:
        transform_list.append(horizontal_flip(prob=0.5))
    if args.augs.rgb.vflip:
        transform_list.append(vertical_flip(prob=0.5))
    
    transform_list.append(resize(args.dataset.img_size))

    if len(transform_list) == 0:
        transform_list.append(identity())
    return torchvision.transforms.Compose(transform_list)
    
def sample_correspondences(corrs: Tensor, instance_id: str, debug_type: str, max_corrs: int) -> Tuple[Tensor, bool]:

    '''
    Sampled correspondencies to a fixed number.
    Returns also a booledn to show validity (i.e. if there are correspondencies)
    '''

    if corrs.shape[0] > 0:
        valid = True
        corrs = unique_matches(corrs.clone())
        if debug_type == 'anchor' or debug_type == 'oracle':
            sampled_corrs = corrs[:max_corrs]
        else:
            idxs = torch_sample_select(corrs, max_corrs)        
            sampled_corrs = corrs[idxs]
    else:
        valid=False
        sampled_corrs = torch.zeros((0,4))
        print(f"Problem with instance {instance_id}")

    return sampled_corrs, valid

class CollateWrapper(object):

    def __init__(self, corr_n : int):
        self.max_corrs = corr_n

    def __call__(self, data : Sequence[Tuple]) -> dict:
        
        orig_rgb_a, rgb_a, mask_a, orig_depth_a, depth_a, eval_depth_a, camera_a, instance_id_a, poses_a, boxes_a, sizes_a = list(), list(), list(), list(), list(), list(), list(), list(), list(), list(), list()
        orig_rgb_q, rgb_q, mask_q, orig_depth_q, depth_q, eval_depth_q, camera_q, instance_id_q, poses_q, boxes_q, sizes_q = list(), list(), list(), list(), list(), list(), list(), list(), list(), list(), list()
        sampled_corr_list, all_corr_list, poses, prompt_list = list(), list(), list(), list()
        valids, instance_ids, cls_ids = list(), list(), list()

        for item_a, item_q, prompt, sampled_corrs, all_corrs, pose, cls_id, instance_id, valid in data:
            
            box_a = item_a['metadata']['boxes']
            box_q = item_q['metadata']['boxes']
            size_a = item_a['hw_size']
            size_q = item_q['hw_size']
            if valid and sampled_corrs.shape[0] > 0:
                valids.append(1.)
            else:
                valids.append(0.)
                sampled_corrs = torch.zeros((self.max_corrs,4)).to(torch.long)
                all_corrs = torch.zeros((self.max_corrs,4)).to(torch.long)
                #print(f"Problem with pair {item_a['instance_id']}, {item_q['instance_id']}")

            box_a = box_a if isinstance(box_a, torch.Tensor) else torch.tensor(box_a)
            box_q = box_q if isinstance(box_q, torch.Tensor) else torch.tensor(box_q)
            size_a = size_a if isinstance(size_a, torch.Tensor) else torch.tensor(size_a)
            size_q = size_q if isinstance(size_q, torch.Tensor) else torch.tensor(size_q)
            rgb_a.append(item_a['rgb'].squeeze())
            orig_rgb_a.append(item_a['orig_rgb'].squeeze())
            mask_a.append(item_a['mask'].squeeze())
            depth_a.append(item_a['depth'].squeeze())
            orig_depth_a.append(item_a['orig_depth'].squeeze())
            eval_depth_a.append(item_a['eval_depth'].squeeze())
            instance_id_a.append(item_a['instance_id'])
            camera_a.append(item_a['camera'].squeeze())
            poses_a.append(item_a['metadata']['poses'][0].squeeze())
            sizes_a.append(size_a)
            boxes_a.append(box_a.squeeze())

            rgb_q.append(item_q['rgb'].squeeze())
            orig_rgb_q.append(item_q['orig_rgb'].squeeze())
            mask_q.append(item_q['mask'].squeeze())
            depth_q.append(item_q['depth'].squeeze())
            orig_depth_q.append(item_q['orig_depth'].squeeze())
            eval_depth_q.append(item_q['eval_depth'].squeeze())
            instance_id_q.append(item_q['instance_id'])
            camera_q.append(item_q['camera'].squeeze())
            poses_q.append(item_q['metadata']['poses'][0].squeeze())
            sizes_q.append(size_q)
            boxes_q.append(box_q.squeeze())
            
            sampled_corr_list.append(sampled_corrs)
            all_corr_list.append(all_corrs)
            prompt_list.append(prompt)
            instance_ids.append(instance_id)
            cls_ids.append(cls_id)

            if pose is not None:
                poses.append(pose)


        final_a = {
            'rgb' : torch.stack(rgb_a,dim=0).to(torch.float32),
            'orig_rgb' : orig_rgb_a,
            'mask' : torch.stack(mask_a,dim=0).to(torch.uint8),
            'orig_depth' : orig_depth_a,
            'eval_depth' : eval_depth_a,
            'depth' : torch.stack(depth_a,dim=0).to(torch.float32),
            'camera' : torch.stack(camera_a,dim=0),
            'pose': torch.stack(poses_a,dim=0),
            'box': torch.stack(boxes_a,dim=0),
            'sizes': torch.stack(sizes_a,dim=0),
            'instance_id' : instance_id_a
        }

        final_q = {
            'rgb' : torch.stack(rgb_q, dim=0).to(torch.float32),
            'orig_rgb' : orig_rgb_q,
            'mask' : torch.stack(mask_q, dim=0).to(torch.uint8),
            'orig_depth' : orig_depth_q,
            'eval_depth' : eval_depth_q,
            'depth' : torch.stack(depth_q,dim=0).to(torch.float32),
            'camera' : torch.stack(camera_q,dim=0),
            'pose': torch.stack(poses_q,dim=0),
            'box': torch.stack(boxes_q,dim=0),
            'sizes': torch.stack(sizes_q,dim=0),
            'instance_id' : instance_id_q
        }


        final_dict = {
            'anchor' : final_a,
            'query' : final_q,
            'corrs' : torch.stack(sampled_corr_list, dim=0).to(torch.long),
            'all_corrs' : all_corr_list, # this is only necessary to compute the pixel distance metric
            'prompt' : prompt_list,
            'valid' : torch.tensor(valids),
            'instance_id' : instance_ids,
            'cls_id' : cls_ids
        }

        if len(poses) > 0:
            final_dict['pose'] = torch.tensor(np.stack(poses, axis=0))

        return final_dict
        
class Shapenet6DDataset(Dataset):

    def __init__(self, args : DictConfig, eval : bool = False):

        self.eval = eval
        self.augs = args.augs
        self.root = args.dataset.root
        self.max_corrs = args.dataset.max_corrs
        self.img_size = tuple(args.dataset.img_size)
        self.collate = CollateWrapper(self.max_corrs)
        self.debug_valid = args.debug_valid

        if eval:
            self.name = args.dataset.test.name
            self.split = args.dataset.test.split
            self.obj = str(args.dataset.test.obj)
            self.augs_fn = build_test_preproc(args)
        else:
            self.name = args.dataset.train.name
            self.split = args.dataset.train.split
            self.obj = str(args.dataset.train.obj)
            self.augs_fn = build_augs(args)
        
        with open(join(self.root, self.name,'templates.json')) as f:
            self.prompt_templates = json.load(f)

        local_root = join(self.root, self.name)
        self.obj_ids = [int(cat) for cat in shapenet.load_object_splits(local_root)[self.obj]]

        #self.augs = self.build_augs()

        self.instances = list()
        self.poses = list()
        self.corrs = list()
        self.init_eval()

        self.annots = shapenet.load_annotations(local_root)
        self.metadata = shapenet.get_metadata(local_root)
        self.cat2instance_id = shapenet.get_instance2cat_id(local_root)

    def init_eval(self):
        '''
        Uses predefined pairs of instances, with precomputed relative poses
        '''
        self.path_split = os.path.join(self.root, self.name, 'fixed_split', self.split)

        with open(join(self.path_split,'instance_list.txt')) as f:
            instances = f.readlines()

        f = open(join(self.path_split,'annots.pkl'),'rb')
        annots = pickle.load(f)

        for instance in instances:

            idx_a, idx_q, obj_id = instance.split(',')
            
            obj_a = int(obj_id)
            obj_q = int(obj_id)

            if obj_a in self.obj_ids:
                pose_annot_id = '_'.join([str(int(e)) for e in (idx_a, idx_q, obj_a)])
                pose = annots[pose_annot_id]['gt']
                pose[:3,3] = pose[:3,3] / 1000.
                self.poses.append(pose)
                self.corrs.append(annots[pose_annot_id]['corrs'])
                self.instances.append((int(idx_a), int(idx_q), obj_q))


    def __getitem__(self, index : int, i=0) -> Tuple:

        img_a, img_q, cat_id = self.instances[index]
        instance_id = f'{img_a}_{img_q}_{cat_id}'
        #print(scene_id_a, img_id_a, ', ', scene_id_q, img_id_q, ', ', cat_id)
        orig_corrs = self.corrs[index]
        pose = self.poses[index]
        
        path = join(self.root, self.name)
        item_a = shapenet.get_item_data(path, self.annots, self.metadata, img_a, cat_id)
        item_q = shapenet.get_item_data(path, self.annots, self.metadata, img_q, cat_id)
                
        item_a = common.preprocess_item(item_a)
        item_q = common.preprocess_item(item_q)

        # prompt is the same by construction
        prompt = self.get_item_prompt(item_a)
        orig_corrs = torch.tensor(orig_corrs)
        #viz.corr_set(item_a['rgb'], item_q['rgb'], res_corrs.numpy(), res_corrs.numpy(), f'tmp/{i}_{instance_id}.png')        
        
        #viz.corr_set(item_a['rgb'], item_q['rgb'], orig_corrs.numpy(), orig_corrs.numpy(), f'tmp/{instance_id}' + post)         
        item_a, item_q, res_corrs = self.augs_fn((item_a, item_q, orig_corrs))
        sampled_corrs, valid_corrs = sample_correspondences(res_corrs, instance_id, self.debug_valid, self.max_corrs)
        #viz.corr_set(item_a['rgb'], item_q['rgb'], sampled_corrs.numpy(), sampled_corrs.numpy(), f'tmp/{instance_id}_aug' + post)     
        
        valid_a = common.check_validity(item_a)
        valid_q = common.check_validity(item_q)
        valid = valid_a and valid_q and valid_corrs

        return item_a, item_q, prompt, sampled_corrs, orig_corrs, pose, cat_id, instance_id, valid

    def __len__(self):
        return len(self.instances)

    def get_obj_info(self, obj_id: int) -> Tuple:
        
        old_id = self.metadata[1][int(obj_id)]
        return shapenet.get_obj_info(join(self.root, self.name), old_id)

    def get_item_prompt(self, item : dict) -> List:

        name = item['metadata']['cls_names'][0]

        if not self.eval:
            # randomly add object description
            if self.augs.text.synset and np.random.rand() > 0.2:
                name = np.random.choice(item['metadata']['cls_descs'][0])


        prompts = [name]
        prompts.extend([template.format(name) for template in self.prompt_templates])
        
        return prompts

class NOCSDataset(Dataset):

    def __init__(self, args : DictConfig, eval : bool = False):
        
        self.eval = eval
        self.augs = args.augs
        self.root = args.dataset.root
        self.max_corrs = args.dataset.max_corrs
        self.debug_valid = args.debug_valid
        self.img_size = tuple(args.dataset.img_size)
        self.collate = CollateWrapper(self.max_corrs)
        # this is only valid at test time
        self.mask_type = args.test.mask

        # used for ablation on prompt
        self.add_description = args.test.add_description    
        
        if eval:
            self.name = args.dataset.test.name
            self.split = args.dataset.test.split
            self.obj = str(args.dataset.test.obj)
            self.augs_fn = build_test_preproc(args)
        else:
            self.name = args.dataset.train.name
            self.split = args.dataset.train.split
            self.obj = str(args.dataset.train.obj)
            self.augs_fn = build_augs(args)

        # ONLY FOR REAL DATA https://github.com/hughw19/NOCS_CVPR2019/issues/54
        self.K = np.asarray([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])
        
        with open(join(self.root,self.name,'templates.json')) as f:
            self.prompt_templates = json.load(f)

        if args.use_seed:
            set_seed(args.seed)

        with open(os.path.join(self.root, self.name, 'object_splits.json')) as f:
            self.obj_ids = [int(cat) for cat in json.load(f)[self.obj]]

        #self.augs = self.build_augs()

        self.instances = list()
        self.poses = list()
        self.corrs = list()
        self.abs_poses = nocs.get_part_data(join(self.root, self.name))
        self.obj_names = nocs.get_obj_names(join(self.root, self.name))

        self.path_split = os.path.join(self.root, self.name, 'fixed_split', self.split)

        self.obj_models, self.obj_diams, self.obj_symms = nocs.get_obj_data(join(self.root,self.name))
        
        with open(join(self.path_split,'instance_list.txt')) as f:
            instances = f.readlines()

        f = open(join(self.path_split,'annots.pkl'),'rb')
        annots = pickle.load(f)

        for instance in instances:

            split, idx_a, idx_q, cat_id = instance.split(',')
            
            cat_id_a, obj_name_a = cat_id.strip().split(' ')
            cat_id_a = int(cat_id_a)
            scene_a, img_a = [int(n) for n in idx_a.split(' ') if n != '']
            scene_q, img_q = [int(n) for n in idx_q.split(' ') if n != '']
            
            if cat_id_a in self.obj_ids:
                pose_annot_id = '_'.join([str(e).strip('\n') for e in (scene_a, img_a, scene_q, img_q, cat_id_a, obj_name_a)])
                pose = annots[pose_annot_id]['gt']
                pose[:3,3] = pose[:3,3] / 1000.
                self.poses.append(pose)
                self.corrs.append(annots[pose_annot_id]['corrs'])
                self.instances.append((split, scene_a, img_a, scene_q, img_q, cat_id_a, obj_name_a))

        # get tracked pair list
        self.tracked_instances = list()
        with open(join(self.path_split,'tracked.txt')) as f:
            tracked = f.readlines()
        
        for instance in tracked:

            split, idx_a, idx_q, cat_id = instance.split(',')    
            cat_id_a, obj_name_a = cat_id.strip().split(' ')
            cat_id_a = int(cat_id_a)
            scene_a, img_a = [int(n) for n in idx_a.split(' ') if n != '']
            scene_q, img_q = [int(n) for n in idx_q.split(' ') if n != '']
            instance_id = f'{scene_a}_{img_a}_{scene_q}_{img_q}_{obj_name_a}'
            self.tracked_instances.append(instance_id)

    def get_item(self, scene_id : int, img_id : int, obj_id: str, mask_type: str = 'oracle') -> dict:
        '''
        Wrapper for NOCS dataset
        '''
        path = join(self.root, self.name)
        item = nocs.get_item_data(path, scene_id, img_id, self.abs_poses, self.obj_names, obj_id, mask_type)
        return item

    def __getitem__(self, index : int) -> Tuple:

        split, scene_a, img_a, scene_q, img_q, cat_id, obj_id = self.instances[index]
        instance_id = f'{scene_a}_{img_a}_{scene_q}_{img_q}_{obj_id}'
        orig_corrs = self.corrs[index]
        pose = self.poses[index]
        path = join(self.root, self.name)
        mask = get_mask_type(self.mask_type, self.eval)
        
        item_a = nocs.get_item_data(path, scene_a, img_a, self.abs_poses, self.obj_names, obj_id, mask)
        item_q = nocs.get_item_data(path, scene_q, img_q, self.abs_poses, self.obj_names, obj_id, mask)

        item_a['camera'] = self.K 
        item_q['camera'] = self.K

        item_a = common.preprocess_item(item_a)
        item_q = common.preprocess_item(item_q)
        # prompt is the same by construction
        prompt = self.get_item_prompt(item_a)
        orig_corrs = torch.tensor(orig_corrs)
        # post = '_test.png' if self.eval else '.png'
        # viz.corr_set(item_a['rgb'], item_q['rgb'], orig_corrs.numpy(), orig_corrs.numpy(), f'tmp/{instance_id}_{self.box_type}' + post)         
        item_a, item_q, res_corrs = self.augs_fn((item_a, item_q, orig_corrs))
        sampled_corrs, valid_corrs = sample_correspondences(res_corrs, instance_id, self.debug_valid, self.max_corrs)
        # viz.corr_set(item_a['rgb'], item_q['rgb'], sampled_corrs.numpy(), sampled_corrs.numpy(), f'tmp/{instance_id}_{self.box_type}_aug' + post)     

        # unvalid objects are skipped at training time and counted as automatic failure at test times 
        # this should only happen when using a predictede segm mask, e.g. ovseg  
        valid_a = common.check_validity(item_a)
        valid_q = common.check_validity(item_q)
        valid = valid_a and valid_q and valid_corrs
        return item_a, item_q, prompt, sampled_corrs, orig_corrs, pose, obj_id, instance_id, valid

    def __len__(self):
        return len(self.instances)

    def get_obj_info(self, obj_id) -> Tuple:
        '''
        Returns the info (object model, object diameter, object symmetry) of a given object name
        '''
        return self.obj_models[obj_id], self.obj_diams[obj_id], self.obj_symms[obj_id]

    def get_object_info(self) -> Tuple:
        '''
        Returns the info (object model, object diameter, object symmetry) of all objects
        '''
        return self.obj_models, self.obj_diams, self.obj_symms

    def get_item_prompt(self, item : dict) -> List:
        
        name = item['metadata']['cls_names'][0]
        
        if self.add_description == 'yes':
            desc = item['metadata']['cls_descs'][0][0]
            name = f'{desc} {name}' 
        elif self.add_description == 'wrong':
            desc = item['metadata']['cls_descs'][0][1]
            name = f'{desc} {name}' 
        elif self.add_description == 'desconly':
            desc = item['metadata']['cls_descs'][0][0]
            name = f'{desc} object' 
        
        prompts = [name]
        prompts.extend([template.format(name) for template in self.prompt_templates])
        
        return prompts

    def save_pred_masks(self, masks : torch.Tensor, instance_ids : list):
        '''
        Saves given mask based on instance_id
        '''
        root = join(self.root, self.name, 'oryon')
        masks = interpolate(masks.to(torch.uint8).unsqueeze(1), size=(480,640), mode='nearest').squeeze(1)
        for instance_id, mask in zip(instance_ids,masks):

            mask[mask==0] = 255
            mask_save = Image.fromarray(mask.cpu().numpy().astype(np.uint8))
            mask_save.save(join(root,instance_id + '.png'))

class TOYLDataset(Dataset):

    def __init__(self, args : DictConfig, eval : bool = False):
        
        self.eval = eval
        self.augs = args.augs
        self.root = args.dataset.root
        self.max_corrs = args.dataset.max_corrs
        self.img_size = tuple(args.dataset.img_size)
        self.debug_valid = args.debug_valid
        self.collate = CollateWrapper(self.max_corrs)        
        self.mask_type = args.test.mask

        # used for ablation on prompt
        self.add_description = args.test.add_description

        if eval:
            self.name = args.dataset.test.name
            self.split = args.dataset.test.split
            self.obj = str(args.dataset.test.obj)
            self.augs_fn = build_test_preproc(args)
        else:
            self.name = args.dataset.train.name
            self.split = args.dataset.train.split
            self.obj = str(args.dataset.train.obj)
            self.augs_fn = build_augs(args)
        
        self.K = np.asarray([[572.4114, 0.0, 325.2611], [0.0, 573.5704, 242.0489], [0.0, 0.0, 1.0]])
        
        with open(join(self.root,self.name,'templates.json')) as f:
            self.prompt_templates = json.load(f)

        if args.use_seed:
            set_seed(args.seed)

        with open(os.path.join(self.root, self.name, 'object_splits.json')) as f:
            self.obj_ids = [int(cat) for cat in json.load(f)[self.obj]]

        #self.augs = self.build_augs()

        self.instances = list()
        self.poses = list()
        self.corrs = list()
        self.local_root = join(self.root, self.name)
        self.part_data = toyl.get_part_data(self.local_root)
        self.obj_names = toyl.get_obj_names(self.local_root)

        self.path_split = os.path.join(self.local_root, 'fixed_split', self.split)

        self.obj_models, self.obj_diams, self.obj_symms = toyl.get_obj_data(join(self.local_root))

        with open(join(self.path_split,'instance_list.txt')) as f:
            instances = f.readlines()

        f = open(join(self.path_split,'annots.pkl'),'rb')
        annots = pickle.load(f)

        for instance in instances:

            split, idx_a, idx_q, cls_id = instance.split(',')
            
            cls_id = int(cls_id)
            scene_a, img_a = [int(n) for n in idx_a.split(' ') if n != '']
            scene_q, img_q = [int(n) for n in idx_q.split(' ') if n != '']
            
            if cls_id in self.obj_ids:
                pose_annot_id = '_'.join([str(e).strip('\n') for e in (scene_a, img_a, scene_q, img_q, cls_id)])
                pose = annots[pose_annot_id]['gt']
                pose[:3,3] = pose[:3,3] / 1000.
                self.poses.append(pose)
                self.corrs.append(annots[pose_annot_id]['corrs'])
                self.instances.append((split, scene_a, img_a, scene_q, img_q, cls_id))

        self.tracked_instances = list()
        with open(join(self.path_split,'tracked.txt')) as f:
            tracked = f.readlines()
        
        for instance in tracked:

            split, idx_a, idx_q, cat_id = instance.split(',')    
            cat_id = int(cat_id)
            scene_a, img_a = [int(n) for n in idx_a.split(' ') if n != '']
            scene_q, img_q = [int(n) for n in idx_q.split(' ') if n != '']
            instance_id = f'{scene_a}_{img_a}_{scene_q}_{img_q}_{cat_id}'
            self.tracked_instances.append(instance_id)

    def get_item(self, scene_id : int, img_id : int, obj_id : int, mask:str = 'oracle') -> dict:
        '''
        Wrapper usable outside the dataset
        '''
        item = toyl.get_item_data(self.local_root, scene_id, img_id, self.part_data, self.obj_names, int(obj_id))
        return item

    def __getitem__(self, index : int) -> Tuple:

        split, scene_a, img_a, scene_q, img_q, cls_id = self.instances[index]
        instance_id = f'{scene_a}_{img_a}_{scene_q}_{img_q}_{cls_id}'
        orig_corrs = self.corrs[index]
        pose = self.poses[index]

        mask_type = get_mask_type(self.mask_type, self.eval)
        item_a = toyl.get_item_data(self.local_root, scene_a, img_a, self.part_data, self.obj_names, cls_id, mask_type)
        item_q = toyl.get_item_data(self.local_root, scene_q, img_q, self.part_data, self.obj_names, cls_id, mask_type)
        
        item_a['camera'] = self.K 
        item_q['camera'] = self.K

        item_a = common.preprocess_item(item_a)
        item_q = common.preprocess_item(item_q)
        
        orig_corrs = torch.tensor(orig_corrs)
        # prompt is the same by construction
        prompt = self.get_item_prompt(item_a)
        
        item_a, item_q, res_corrs = self.augs_fn((item_a, item_q, orig_corrs))
        sampled_corrs, valid_corrs = sample_correspondences(res_corrs, instance_id, self.debug_valid, self.max_corrs)
        
        # unvalid objects are skipped at training time and counted as automatic failure at test times   
        valid_a = common.check_validity(item_a)
        valid_q = common.check_validity(item_q)
        valid = valid_a and valid_q and valid_corrs

        return item_a, item_q, prompt, sampled_corrs, orig_corrs, pose, cls_id, instance_id, valid

    def __len__(self):
        return len(self.instances)

    def get_obj_info(self, obj_id) -> Tuple:
        '''
        Returns the info (object model, object diameter, object symmetry) of a given object name
        '''
        return self.obj_models[int(obj_id)], self.obj_diams[int(obj_id)], self.obj_symms[int(obj_id)]

    def get_object_info(self) -> Tuple:
        '''
        Returns the info (object model, object diameter, object symmetry) of all objects
        '''
        return self.obj_models, self.obj_diams, self.obj_symms

    def get_item_prompt(self, item : dict) -> List:
        
        name = item['metadata']['cls_names'][0]
        
        if self.add_description == 'yes':
            desc = item['metadata']['cls_descs'][0][0]
            name = f'{desc} {name}' 
        elif self.add_description == 'wrong':
            desc = item['metadata']['cls_descs'][0][1]
            name = f'{desc} {name}' 
        elif self.add_description == 'desconly':
            desc = item['metadata']['cls_descs'][0][0]
            name = f'{desc} object' 

        prompts = [name]
        prompts.extend([template.format(name) for template in self.prompt_templates])
                
        return prompts
    
    def save_pred_masks(self, masks : torch.Tensor, instance_ids : list):
        '''
        Saves given mask based on instance_id
        '''
        root = join(self.root, self.name, 'oryon')
        masks = interpolate(masks.to(torch.uint8).unsqueeze(1), size=(480,640), mode='nearest').squeeze(1)
        for instance_id, mask in zip(instance_ids,masks):

            mask[mask==0] = 255
            mask_save = Image.fromarray(mask.cpu().numpy().astype(np.uint8))
            mask_save.save(join(root,instance_id + '.png'))

