import os
import torch
import math
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from omegaconf import DictConfig, OmegaConf
from losses import *
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from datasets import NOCSDataset, Shapenet6DDataset, TOYLDataset
from typing import Dict, Tuple
from torch import Tensor
from net import Oryon
from datetime import datetime
from utils.pcd import lift_pcd
from utils.misc import set_deterministic_seed, args2dict, rescale_coords
from utils.geo6d import best_fit_transform_with_RANSAC
from utils import viz, coordinates
from utils.pointdsc.init import get_pointdsc_pose, get_pointdsc_solver
from utils.pcd import nn_correspondences
from utils.evaluator import Evaluator
import numpy as np

class FPM_Pipeline(LightningModule):
    
    """
    This class is a PyTorch Lightning system and contain the core of the major steps made during the training of a NN
    """

    def __init__(self, args, test_model=False):
        r"""
        This functions setup the model and the NN loss
        """
        super().__init__()

        self.args = args
        self.test_model = test_model
        self.model = self.get_model()
        self.corrs_device = args.corrs_device
        self.feature_loss = self.get_loss()
        self.evaluator = Evaluator(args.exp_tag, compute_vsd=True)
        if self.args.test.solver == 'pointdsc':
            self.pointdsc_solver = get_pointdsc_solver(self.args.pretrained.pointdsc, self.args.device)

############ GETTER METHODS ##################

    def get_callbacks(self):

        args=self.args
        cpt_callback = ModelCheckpoint(
            dirpath=args.tmp.ckpt_out,
            every_n_epochs=args.training.freq_save,
            save_top_k=-1,
            filename='{epoch:04d}'
        )

        bar_callback = TQDMProgressBar(refresh_rate=50)

        return [cpt_callback, bar_callback]

    def get_logger(self):

        args = self.args
        
        wb_logger = WandbLogger(
            save_dir=args.tmp.logs_out,
            project='Oryon-tpami',
            name=args.exp_name,
            offline=True
        )

        self.wb_logger = wb_logger

        return [wb_logger]

    def get_model(self) -> Oryon: 
    
        return Oryon(self.args, self.args.device)
    
    def get_loss(self) -> torch.nn.Module:
                
        return FeatureLoss(self.args, self.args.device)
    
    def get_dataset(self, eval:bool) -> torch.utils.data.Dataset:

        if eval:
            dataset_name = self.args.dataset.test.name
        else:
            dataset_name = self.args.dataset.train.name
        
        if dataset_name == 'nocs':
            return NOCSDataset(self.args, eval)
        elif dataset_name == 'shapenet6d':
            return Shapenet6DDataset(self.args, eval)
        elif dataset_name == 'toyl':
            return TOYLDataset(self.args, eval)
        else:
            raise RuntimeError(f"Dataset {dataset_name} not supported")
        
    def configure_optimizers(self) -> Tuple[list, list]:
        """
        This functions setup the optimizer and the scheduler
        """
        parameters = self.model.get_trainable_parameters()

        if self.args.optimization.optim_type == 'SGD':
            optimizer = torch.optim.SGD(
                params=parameters,
                lr=self.args.optimization.lr,
                momentum=self.args.optimization.momentum,
                weight_decay=self.args.optimization.w_decay,
                nesterov=False)
            
        elif self.args.optimization.optim_type == 'Adam':
            optimizer = torch.optim.AdamW(
                params=parameters,
                lr=self.args.optimization.lr,
                weight_decay=self.args.optimization.w_decay)
        else:
            raise RuntimeError('Optimizer type {} not implemented!'.format(self.args.optimization.optim_type))

        self.optimizer = optimizer

        if self.args.optimization.scheduler_type == 'step':
            # Learning rate is reduced after 50%, 75% and 90% of samples like in Segdriven original implementation
            part_milestones = [math.ceil(self.args.training.n_epochs * step) for step in [0.5,0.75,0.9]]
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=part_milestones,
                gamma=self.args.optimization.gamma)
                
        elif self.args.optimization.scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.args.training.n_epochs - 1,
                eta_min=self.args.optimization.gamma * self.args.optimization.lr)

        elif self.args.optimization.scheduler_type == 'exp':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=self.args.optimization.gamma
                )
        elif self.args.optimization.scheduler_type == 'None':
            # Fake step scheduler
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[self.args.training.n_epochs * 2],
                gamma=self.args.optimization.gamma)
        else:
            raise RuntimeError('Scheduler type {} not implemented!'.format(self.args.optimization.scheduler_type))
        
        return [self.optimizer], [scheduler]

#################### TRAINING ##########################
    
    def on_train_start(self) -> None:
        
        if self.global_rank == 0:
            self.wb_logger.watch(self.model, log_freq=10)
            flat_dict = args2dict(self.args)
            for k, v in flat_dict.items():
                self.wb_logger.experiment.config.update({k:v})
        
        path = os.path.join(self.args.exp_root, self.args.exp_name, 'config.yaml')
        with open(path, 'w') as f:
            OmegaConf.save(self.args, f)

        return super().on_train_start()
    
    def training_step(self, batch, batch_idx):

        #self.evaluator.add_object_info(*self.train_dataset.get_object_info())
        self.evaluator.init_training()
        outputs = self.model.forward(batch)            
        losses, results = self.feature_loss.forward(batch, outputs)        

        loss, losses = self.reduce_losses(losses)
        self.evaluator.register_train(results, clear=True)
        self.structured_log(losses, prefix='train')
        
        return loss

    def on_train_end(self):
    
        # update config file with name of the last epoch
        last_ckpt = self.args.training.n_epochs - 1
        self.args.eval.ckpt = os.path.join(self.args.exp_root, self.args.exp_name, 'models', f'epoch={last_ckpt:04d}.ckpt')
        path = os.path.join(self.args.exp_root, self.args.exp_name, 'config.yaml')
        with open(path, 'w') as f:
            OmegaConf.save(self.args, f)

        return super().on_train_start()

################## VALIDATION ##########################    

    def validation_step(self, batch, batch_idx):
        
        outputs = self.model.forward(batch)
                
        losses, results = self.feature_loss.forward(batch, outputs)
        
        BS = outputs['featmap_a'].shape[0]
        self.evaluator.add_object_info(*self.valid_dataset.get_object_info())
        self.evaluator.init_validation()

        for i_b in range(BS):
    
            if self.is_detection_valid(results, batch, i_b):
                # NB: predicted coords are in featmap coordinates
                pred_corrs, _, _ = self.get_featmap_corrs(batch, outputs,results,idx=i_b)
                if pred_corrs != None:
                    pred_pose = self.get_pose(batch, pred_corrs, idx=i_b)
                    pred_q = pred_pose @ batch['anchor']['pose'][i_b].cpu().detach().to(torch.float32)
                    self.evaluator.register_eval({
                        'iou_a' : results['iou_a'][i_b].unsqueeze(0),
                        'iou_q' : results['iou_q'][i_b].unsqueeze(0),
                        'gt_pose': batch['query']['pose'][i_b].unsqueeze(0),
                        'pred_pose': pred_q.unsqueeze(0),
                        'pred_pose_rel': pred_pose.unsqueeze(0),
                        'camera' : [batch['query']['camera'][i_b].cpu().numpy()],
                        'depth' : [batch['query']['eval_depth'][i_b].squeeze().cpu().numpy()],
                        'cls_id': [batch['cls_id'][i_b]]
                    })

                    #if self.args.viz_valid and ((i_b % 20) == 0):
                    #    pred_corrs = rescale_coords(pred_corrs, (FH,FW), (IH,IW))
                    self.valid_visualization(batch, results, pred_q.clone().numpy(), pred_corrs.cpu().clone().numpy(), i_b)
                else:
                    print("Nothing found :(")
                    self.evaluator.register_valid_failure({
                        'iou_a' : results['iou_a'][i_b].unsqueeze(0),
                        'iou_q' : results['iou_q'][i_b].unsqueeze(0),
                        'cls_id': [batch['cls_id'][i_b]],
                        'instance_id': [batch['instance_id'][i_b]]
                    })
            else:
                print("Nothing found :(")
                self.evaluator.register_valid_failure({
                    'iou_a' : results['iou_a'][i_b].unsqueeze(0),
                    'iou_q' : results['iou_q'][i_b].unsqueeze(0),
                    'cls_id': [batch['cls_id'][i_b]],
                    'instance_id': [batch['instance_id'][i_b]]
                })
        loss, losses = self.reduce_losses(losses)
        self.structured_log(losses, prefix='valid')
    
        return loss

    def valid_visualization(self, batch: dict, loss_results: dict, pred_pose: np.ndarray, pred_corrs: np.ndarray, batch_idx: int, test:bool=False):
        '''
        Visualize various information about a validation sample
        Note: pred_corrs and gt corrs here are all in initial input shape
        '''

        if test:
            dataset = self.test_dataset
        else:
            dataset = self.valid_dataset

        rgb_a = batch['anchor']['rgb'][batch_idx].cpu().squeeze().clone().numpy().transpose(1,2,0) 
        rgb_q = batch['query']['rgb'][batch_idx].cpu().squeeze().clone().numpy().transpose(1,2,0)
        gt_mask_a = batch['anchor']['mask'][batch_idx].cpu().squeeze().numpy()
        gt_mask_q = batch['query']['mask'][batch_idx].cpu().squeeze().numpy()
        pred_mask_a = loss_results['mask_a'][batch_idx].cpu().squeeze().numpy()
        pred_mask_q = loss_results['mask_q'][batch_idx].cpu().squeeze().numpy()
        mask_logits_a = loss_results['logits_a'][batch_idx].cpu().squeeze().numpy()
        mask_logits_q = loss_results['logits_q'][batch_idx].cpu().squeeze().numpy()
        gt_corrs = batch['corrs'][batch_idx].clone().cpu().numpy()
        gt_pose = batch['query']['pose'][batch_idx].clone().cpu().numpy()

        scene_a, img_a, obj_a = batch['anchor']['instance_id'][batch_idx].split(' ')
        scene_q, img_q, _ = batch['query']['instance_id'][batch_idx].split(' ')
        instance_id = batch['instance_id'][batch_idx]
        
        out_path = f'{self.args.tmp.results_out}/viz/{self.args.dataset.test.name}_{self.args.dataset.test.split}_epoch{self.current_epoch}_{instance_id}_{self.args.test.mask}'

        obj_model, _, _ = dataset.get_obj_info(obj_a)
        obj_model = obj_model['pts'] / 1000.

        item_a = dataset.get_item(int(scene_a), int(img_a), obj_a, 'oracle')
        item_q = dataset.get_item(int(scene_q), int(img_q), obj_a, 'oracle')

        #viz.feature_distance(rgb_a, rgb_q, featmap_a, featmap_q, gt_corrs, out_path+'_feats.png')
        viz.pred_mask(rgb_a, rgb_q, gt_mask_a, gt_mask_q, pred_mask_a, pred_mask_q, mask_logits_a, mask_logits_q, out_path + '_mask.png')        
        viz.corr_set(rgb_a, rgb_q, gt_corrs, pred_corrs, out_path+'_corrs.png')
        viz.pred_pose(item_a['rgb'], item_q['rgb'], gt_pose, pred_pose, dataset.K, obj_model, out_path+'_pose.png')
        #viz.corr_neg(rgb_a, rgb_q, gt_corrs, neg_idx_a, neg_idx_q, out_path+'_neg.png')

################ TEST ################################
    
    def on_test_start(self):
        self.pred_file, self.metric_file = self.get_pred_filename()
        if self.args.debug_valid:
            print("WARNING: USING GROUND TRUTH CORRESPONDENCES!!")

        if self.args.seed is not None:
            set_deterministic_seed(self.args.seed)
        else:
            set_deterministic_seed(1)
        # init metrics file
        self.evaluator.add_object_info(*self.test_dataset.get_object_info())
        self.evaluator.init_test()
        # add to the evaluator the object models
        return super().on_test_start()
    
    def test_step(self, batch, batch_idx):
        outputs = self.model.forward(batch)
        BS = outputs['featmap_a'].shape[0]
        FH,FW = self.args.model.image_encoder.img_size
        IH,IW = self.args.dataset.img_size
        _, results = self.feature_loss.forward(batch, outputs)
        
        for i_b in range(BS):
        
            instance_id_a, instance_id_q = batch['anchor']['instance_id'][i_b], batch['query']['instance_id'][i_b]
            if self.is_detection_valid(results, batch, i_b):
                pred_corrs, _, _ = self.get_featmap_corrs(batch, outputs, results, idx=i_b)
                if pred_corrs != None:
                    pred_pose = self.get_pose(batch, pred_corrs, idx=i_b)
                    pred_q = pred_pose @ batch['anchor']['pose'][i_b].cpu().detach().to(torch.float32)
                    self.evaluator.register_test({
                        'iou_a' : results['iou_a'][i_b].unsqueeze(0),
                        'iou_q' : results['iou_q'][i_b].unsqueeze(0),
                        'gt_pose': batch['query']['pose'][i_b].unsqueeze(0),
                        'pred_pose': pred_q.unsqueeze(0),
                        'pred_pose_rel': pred_pose.unsqueeze(0),
                        'cls_id': [batch['cls_id'][i_b]],
                        'camera' : [batch['query']['camera'][i_b].cpu().numpy()],
                        'depth' : [batch['query']['eval_depth'][i_b].squeeze().cpu().numpy()],
                        'instance_id': [batch['instance_id'][i_b]]
                    })
                    if batch['instance_id'][i_b] in self.test_dataset.tracked_instances:
                        pred_corrs = rescale_coords(pred_corrs, (FH,FW), (IH,IW))
                        self.valid_visualization(batch, results, pred_q.clone().numpy(), pred_corrs.cpu().clone().numpy(), i_b, test=True)
                else:
                    self.evaluator.register_test_failure({
                        'iou_a' : results['iou_a'][i_b].unsqueeze(0),
                        'iou_q' : results['iou_q'][i_b].unsqueeze(0),
                        'cls_id': [batch['cls_id'][i_b]],
                        'instance_id': [batch['instance_id'][i_b]]
                    })
                    pred_pose = torch.eye(4)
            else:
                self.evaluator.register_test_failure({
                    'iou_a' : results['iou_a'][i_b].unsqueeze(0),
                    'iou_q' : results['iou_q'][i_b].unsqueeze(0),
                    'cls_id': [batch['cls_id'][i_b]],
                    'instance_id': [batch['instance_id'][i_b]]
                })
                pred_pose = torch.eye(4)

            # adds the current predicted pose to the text file of predictions
            iou_a = results['iou_a'][i_b].cpu().numpy()
            iou_q = results['iou_q'][i_b].cpu().numpy()
            self.add_pred_pose(instance_id_a, instance_id_q, iou_a, iou_q, pred_pose.cpu().numpy())


    def on_test_end(self):
        '''
        Aggregate and print final results
        '''

        self.pred_file.close()

        self.evaluator.test_summary()
        self.evaluator.save(self.metric_file)
        print(self.evaluator.get_latex_str())
        self.metric_file.close()

        return super().on_test_end()

    def is_detection_valid(self, results: dict, batch: dict, idx: int) -> bool:
        '''
        Returns true if detection is valid
        '''
        # then we are evaluating on an external mask (either ground truth or some other backbone)
        if self.args.test.mask != 'predicted':
            mask_a_ = batch['anchor']['mask'][idx]
            mask_q_ = batch['query']['mask'][idx]
            
            copy_a = mask_a_.clone().to(torch.float).unsqueeze(0).unsqueeze(1)
            copy_q = mask_q_.clone().to(torch.float).unsqueeze(0).unsqueeze(1)

            mask_a = F.interpolate(copy_a, tuple(self.args.model.image_encoder.img_size), mode='nearest').squeeze().to(torch.int)
            mask_q = F.interpolate(copy_q, tuple(self.args.model.image_encoder.img_size), mode='nearest').squeeze().to(torch.int)
        # else  check the results
        else:
            mask_a = results['mask_a'][idx]
            mask_q = results['mask_q'][idx]

        valid_a = torch.count_nonzero(mask_a == 1)
        valid_q = torch.count_nonzero(mask_q == 1)
        valid = (valid_a.item() > 0) and (valid_q.item() > 0)
        
        return valid

    def get_featmap_corrs(self, batch: dict, net_output: dict, results: dict, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        '''
        Fuction that compute correspondences from two feature maps
        NB:
        ground truth corrs are in input shape (224x224 by default)
        predicted featmap (e.g., correspondences) are in featmap shape
        '''
        NH, NW = net_output['featmap_a'].shape[2:]   # shape of output feature map

        # external case: select zone for correspondences from batch
        if self.args.test.mask != 'predicted':
            mask_ai = batch['anchor']['mask'][idx].clone().unsqueeze(0).unsqueeze(1)
            mask_qi = batch['query']['mask'][idx].clone().unsqueeze(0).unsqueeze(1)
            mask_ai = F.interpolate(mask_ai.to(torch.float), (NH,NW), mode='nearest').squeeze().to(torch.int)
            mask_qi = F.interpolate(mask_qi.to(torch.float), (NH,NW), mode='nearest').squeeze().to(torch.int)
        else:
        # normal case, use predicted mask
            mask_ai, mask_qi = results['mask_a'][idx], results['mask_q'][idx]
        
        featmap_ai, featmap_qi = net_output['featmap_a'][idx].clone(), net_output['featmap_q'][idx].clone()
        pred_corrs = nn_correspondences(featmap_ai, featmap_qi, mask_ai, mask_qi, self.args.test.dist_th, self.args.test.n_corrs, self.args.test.src_sampling, self.corrs_device)
        
        if pred_corrs is not None:
            # NB: coordinates here are in featmap dimensions. They must now be resized to actual depth
            corr_ai, corr_qi = pred_corrs[:, :2], pred_corrs[:, 2:]
            pos_a = featmap_ai[:, corr_ai[:,0],corr_ai[:,1]].transpose(1,0)       
            pos_q = featmap_qi[:, corr_qi[:,0],corr_qi[:,1]].transpose(1,0)
        else:
            pos_a, pos_q = None, None

        return pred_corrs, pos_a, pos_q

    def get_pose(self, batch : dict, corrs : Tensor, idx : int) -> Tensor:
        '''
        Function used at inference to get 3D-3D correspondences and solve them by open3d registration
        '''
        depth_a, depth_q = batch['anchor']['orig_depth'][idx].squeeze(), batch['query']['orig_depth'][idx].squeeze()    #[H,W]
        camera_a = batch['anchor']['camera'][idx].to(depth_a.device).reshape(9)
        camera_q = batch['query']['camera'][idx].to(depth_a.device).reshape(9)
        
        # original sizes of images and depths
        HA,WA = batch['anchor']['sizes'][idx].cpu()
        HQ,WQ = batch['query']['sizes'][idx].cpu()
        # size output in the feature map
        HO, WO = self.args.model.image_encoder.img_size
        # this corrs are express in featmap dimension, within the crop.
        corrs_a, corrs_q = corrs[:,:2].clone().cpu(), corrs[:,2:].clone().cpu()
        instance_id = batch['instance_id'][idx]
        out_path = f'{self.args.tmp.results_out}/viz/{self.args.dataset.test.name}_{self.args.dataset.test.split}_epoch{self.current_epoch}_{instance_id}_{self.args.test.mask}_bicorrs.png'
        
        corrs_a = coordinates.scale_coords(corrs_a, (HO,WO), (HA,WA))
        corrs_q = coordinates.scale_coords(corrs_q, (HO,WO), (HQ,WQ))
        valid_a = coordinates.get_valid_coords(corrs_a, (HA,WA))
        valid_q = coordinates.get_valid_coords(corrs_q, (HQ,WQ))
        
        valid = torch.logical_and(valid_a, valid_q)
        corrs_a, corrs_q = corrs_a[valid], corrs_q[valid]
        
        corrs_a = corrs_a.to(depth_a.device).to(torch.long)
        corrs_q = corrs_q.to(depth_q.device).to(torch.long)

        # functions lifts pcd in millimeters, here it is converted to meters
        pcd_a = lift_pcd(depth_a.unsqueeze(-1), camera_a, (corrs_a[:,1],corrs_a[:,0])) / 1000.
        pcd_q = lift_pcd(depth_q.unsqueeze(-1), camera_q, (corrs_q[:,1],corrs_q[:,0])) / 1000.
        
        if self.args.test.solver == 'ransac':
            pose = best_fit_transform_with_RANSAC(pcd_a.cpu().numpy(), pcd_q.cpu().numpy(), max_iter=10000, fix_percent=0.9999, match_err=0.001)
            pose4 = np.eye(4)
            pose4[:3,:] = pose
            pose4 = torch.tensor(pose4)
        elif self.args.test.solver == 'pointdsc':
            pose4 = get_pointdsc_pose(self.pointdsc_solver, pcd_a, pcd_q, self.device)
        else:
            raise RuntimeError(f"Solver {self.args.test.solver} not implemented")
            
        return pose4.to(torch.float32)

    def get_pred_filename(self):
        '''
        Generate the prediction file name by using timestamps. Returns file pointers of results file and metrics files
        '''
        now = datetime.now()
        dt_string = now.strftime("%d%m%Y_%H%M")
        rand_seed = np.random.randint(0,1000)
        pred_file = f'{self.args.dataset.test.name}_{self.args.dataset.test.split}_{self.args.dataset.test.obj}_{dt_string}_{rand_seed}.csv'
        metric_file = f'{self.args.dataset.test.name}_{self.args.dataset.test.split}_{self.args.dataset.test.obj}_{dt_string}_{rand_seed}.json'
        fp = open(os.path.join(self.args.tmp.results_out, pred_file),'w')
        fm = open(os.path.join(self.args.tmp.results_out, metric_file),'w')
        dest = os.path.join(self.args.tmp.results_out,f'config_{dt_string}_{rand_seed}.yaml')
        OmegaConf.save(self.args, dest)
        
        return fp, fm

    def add_pred_pose(self, id_a: str, id_q: str, mask_a_iou: np.ndarray, mask_q_iou: np.ndarray, pred_pose: np.ndarray):
        '''
        Saves a precicted pose in the prediction file
        '''
        pred_pose = ' '.join([str(n) for n in pred_pose[:3,:].flatten()])
        line = ','.join([id_a, id_q, pred_pose, str(mask_a_iou), str(mask_q_iou)])#, rle_a, rle_q])
        line += '\n'
        self.pred_file.write(line)

###################### DATALOADERS ######################

    def get_train_dataloader(self):
        
        args = self.args
        train_set = self.get_dataset(eval=False)
        print("TRAINING on {}, split {}, object split {}. Samples: {}".format(train_set.name, train_set.split, train_set.obj, train_set.__len__()))
        self.train_dataset = train_set
        train_loader = DataLoader(
            dataset=train_set,
            batch_size=args.dataset.batch_size,
            collate_fn=train_set.collate,
            shuffle=True,
            num_workers=8
        )
        
        return train_loader

    def get_valid_dataloader(self):
        
        args = self.args
        valid_set = self.get_dataset(eval=True)
        print("VALIDATING on {}, split {}, object split {}. Samples: {}".format(valid_set.name, valid_set.split, valid_set.obj, valid_set.__len__()))
        self.valid_dataset = valid_set
        valid_loader = DataLoader(
            dataset=valid_set,
            batch_size=args.dataset.batch_size,
            collate_fn=valid_set.collate,
            shuffle=False,
            num_workers=8
        )

        return valid_loader

    def get_test_dataloader(self):

        args = self.args
        
        test_set = self.get_dataset(eval=True)
        print("TESTING on {}, split {}, object split {}. Samples: {}".format(test_set.name, test_set.split, test_set.obj, test_set.__len__()))
        self.test_dataset = test_set
        test_loader = DataLoader(
            dataset=test_set,
            batch_size=args.dataset.batch_size,
            collate_fn=test_set.collate,
            shuffle=False,
            num_workers=8
        )

        return test_loader

############## MISC #######################

    def structured_log(self, losses : dict, prefix : str) -> None:
        '''
        Logs all the metrics current tracked by the evaluator
        '''
        mean_metrics  = self.evaluator.get_log_means()
        all_metrics = {}
        # setup losses
        total_loss = 0.
        for k,v in losses.items():
            all_metrics['{}_loss/{}'.format(prefix, k)] = v
            total_loss += v
        # setup other metrics
        for k,v in mean_metrics.items():
            all_metrics['{}_metric/{}'.format(prefix, k)] = v

        all_metrics['{}_loss/total'.format(prefix)] = total_loss
                
        self.log_dict(
            all_metrics, 
            on_step=False, 
            on_epoch=True, 
            logger=True,
            sync_dist=True,
            rank_zero_only=True,
            batch_size=self.args.dataset.batch_size
        )

    def reduce_losses(self, losses: dict) -> Tuple[Tensor, dict]:
    
        w_losses = {}
        final_loss = 0.
        weights = self.args.loss.w

        for k in losses.keys():
            w_loss = losses[k] * weights[k]
            final_loss = final_loss + w_loss     
            w_losses[k] = w_loss

        return final_loss, w_losses
        

    def forward(self, x):
        return self.model(x)