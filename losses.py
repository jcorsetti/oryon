import torch
from torch import Tensor
from typing import Tuple, Dict
from omegaconf import DictConfig
import torch.nn.functional as F
from utils.pcd import pdist
from utils.misc import torch_sample_select, rescale_coords
from utils.metrics import mask_iou
from utils.losses import DiceLoss, LovaszLoss, FocalLoss
import numpy as np

class FeatureLoss(torch.nn.Module):
    '''
    Contrastive learning loss with positive and negative samples
    '''
    def __init__(self, args : DictConfig, device : str):
        super().__init__()
        self.device = device
        self.args = args
        self.pos_margin = args.loss.pos_margin
        self.neg_margin = args.loss.neg_margin
        self.neg_kernel = args.loss.neg_kernel_size
        self.hard_negatives = args.loss.hard_negatives
        self.mask_th = args.test.mask_threshold
        self.mask_type = args.loss.mask_type
        
        if self.mask_type == 'cross_entropy':
            self._mask_loss = torch.nn.BCEWithLogitsLoss()
        elif self.mask_type == 'dice':
            self._mask_loss = DiceLoss(weight=torch.tensor([0.5,0.5]))
        elif self.mask_type == 'lovasz':
            self._mask_loss = LovaszLoss()
        elif self.mask_type == 'focal':
            self._mask_loss = FocalLoss()

        else:
            raise RuntimeError(f"Mask loss function {self.mask_type} not implemented.")

        
    def mask_loss(self, pred_logits: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        '''
        Prediction will probably be lower then ground truth in resolution.
        Ground truth is downsampled if this happens
        pred: [B,N,H1,W1]
        gt:   [B,H2,W2]
        '''

        gt_shape = gt.shape[1:]
        pred_shape = pred_logits.shape[2:]
        gt_c = gt.clone()
        #print(gt.shape, pred.shape)
        # reduce gt dimension if necessary
        if gt_shape != pred_shape:
            gt_c = F.interpolate(gt.unsqueeze(1), pred_shape, mode='nearest').squeeze(1)

        pred_logits  = pred_logits.squeeze(1)
        loss = self._mask_loss(pred_logits, gt_c.to(torch.float))
        with torch.no_grad():
            pred_mask = torch.where(torch.sigmoid(pred_logits) > self.mask_th, 1, 0)
            iou = mask_iou(gt_c, pred_mask)

        return loss, pred_mask, pred_logits, iou

    def forward(self, batch : dict, net_output : dict) -> Tuple[dict,dict]:
        
        featmap_a = net_output['featmap_a']
        featmap_q = net_output['featmap_q']
        gt_corrs = batch['corrs'].clone()
        valid = batch['valid']

        CH,CW = batch['anchor']['rgb'].shape[2:]
        FH,FW = net_output['featmap_a'].shape[2:]
        # ground truth corrs are in resized shape
        # put them in the feature map dimension
        # necessary because image size and feature map size may not correspond

        gt_corrs = rescale_coords(gt_corrs, (CH,CW),(FH,FW))
        gt_corrs = torch.clamp(gt_corrs, min=0, max=FH-1)

        corrs_a, corrs_q = gt_corrs[:,:,:2], gt_corrs[:,:,2:]

        pos_a, pos_q = self.sample_positives(featmap_a, featmap_q, gt_corrs, valid)
        if self.hard_negatives:
            neg_a, neg_a_idxs = self.sample_hardest_negatives(pos_a, featmap_a, corrs_a, valid)
            neg_q, neg_q_idxs = self.sample_hardest_negatives(pos_q, featmap_q, corrs_q, valid)
        else:
            neg_a, neg_a_idxs = self.sample_negatives(pos_a, featmap_a, corrs_a, valid)
            neg_q, neg_q_idxs = self.sample_negatives(pos_q, featmap_q, corrs_q, valid)

        # inverted cosine similarity as measure of distance (norm between 0 and 1)
        dist_pos = 0.5 * (-1*F.cosine_similarity(pos_a, pos_q, dim=2) + 1)
        dist_neg_a =  0.5 * (-1*F.cosine_similarity(pos_a, neg_a, dim=2) + 1)
        dist_neg_q =  0.5 * (-1*F.cosine_similarity(pos_q, neg_q, dim=2) + 1)
        
        pos_loss = F.relu(dist_pos - self.pos_margin)
        neg_loss_a = F.relu(self.neg_margin - dist_neg_a)
        neg_loss_q = F.relu(self.neg_margin - dist_neg_q)
        
        pos_loss = pos_loss.mean(1)
        neg_loss_a = neg_loss_a.mean(1)
        neg_loss_q = neg_loss_q.mean(1)

        if torch.count_nonzero(valid).item() > 0:
            # only consider element in the batch with valid 
            pos_loss = pos_loss[valid==1].mean()
            neg_loss_a = neg_loss_a[valid==1].mean()
            neg_loss_q = neg_loss_q[valid==1].mean()
        else:
            pos_loss = torch.tensor(0).cuda()
            neg_loss_a = torch.tensor(0).cuda()
            neg_loss_q = torch.tensor(0).cuda()

        if torch.isnan(pos_loss).any():
            print("Pos loss NaN!")
        if torch.isnan(neg_loss_a).any():
            print("Neg loss a NaN!")
        if torch.isnan(neg_loss_q).any():
            print("Neg loss q NaN!")

        mask_loss_a, pred_mask_a, pred_logits_a, iou_a = self.mask_loss(net_output['mask_a'], batch['anchor']['mask'])
        mask_loss_q, pred_mask_q, pred_logits_q, iou_q = self.mask_loss(net_output['mask_q'], batch['query']['mask'])

        # this contains loss to be optimized
        losses = {
            'mask': 0.5 * (mask_loss_a + mask_loss_q),
            'pos' : pos_loss,
            'neg' : 0.5 * (neg_loss_a + neg_loss_q)
        }
        # this contains intermediate results used for metrics
        results = {
            'neg_a' : neg_a_idxs,
            'neg_q' : neg_q_idxs,
            'mask_a' : pred_mask_a,
            'mask_q' : pred_mask_q,
            'logits_a' : pred_logits_a,
            'logits_q' : pred_logits_q,
            'iou_a' : iou_a,
            'iou_q' : iou_q
        }

        return losses, results

    def sample_positives(self, featmap_a : Tensor, featmap_q : Tensor, corrs : Tensor, valid : Tensor) -> Tuple[Tensor, Tensor]:
        '''
        Get list of positive features and optionally predicted correspondences 
        '''

        BS, DIM = featmap_a.shape[:2]
        N_CORR = corrs.shape[1]

        pos_a, pos_q = torch.zeros(BS, N_CORR, DIM),  torch.zeros(BS, N_CORR, DIM)
        pos_a, pos_q = pos_a.to(featmap_a.device), pos_q.to(featmap_q.device)
        
        for i_b in range(BS):
            featmap_ai, featmap_qi, corr_i = featmap_a[i_b], featmap_q[i_b], corrs[i_b]
            corr_ai, corr_qi = corr_i[:, :2], corr_i[:, 2:]
            # just leave them empty, this sample will be skipped in the loss for lack of correspondencies
            if valid[i_b] == 1:
                # based on visualizations coordinates should be correct
                pos_a[i_b] = featmap_ai[:, corr_ai[:,0],corr_ai[:,1]].transpose(1,0)
                pos_q[i_b] = featmap_qi[:, corr_qi[:,0],corr_qi[:,1]].transpose(1,0)

        return (pos_a, pos_q)
    
    def sample_hardest_negatives(self, pos : Tensor, featmap : Tensor, pos_coords : Tensor, valid : Tensor) -> Tuple[Tensor,Tensor]:
        '''
        Sample a set of negatives given a set of positives, a feature map and a mask on the feature map.
        Positives are not needed now, but can be useful later to implement an hardest contrastive approach
        '''

        BS, N, D = pos.shape
        _,_,H,W = featmap.shape
        neg_feats = torch.zeros((BS,N,D)).to(self.device)
        neg_coords = torch.zeros((BS,N,2)).to(self.device)

        # make coordinate map from feature map
        xs = torch.linspace(0, W-1, steps=W)
        ys = torch.linspace(0, H-1, steps=H)
        xmap, ymap = torch.meshgrid(xs,ys, indexing='xy')
        xmap = xmap.flatten().to(self.device)
        ymap = ymap.flatten().to(self.device)
        # basically a list of yx coordinates in the pixel space
        featmap_yx = torch.stack([ymap,xmap],dim=-1)
        if self.device == 'cuda':
            featmap_yx = featmap_yx.half()

        for i_b in range(BS):
            
            if valid[i_b] == 1:
                
                # reshape the feature map: [512,H,W] -> [H*W,512] as a list of features
                featmap_i = featmap[i_b].reshape(D,(H*W)).transpose(1,0)
                featmap_yx_i = featmap_yx.clone()
                
                # reduce pool of negatives to reduce memory requirements
                if featmap_i.shape[0] > 2000:
                    idxs = torch_sample_select(featmap_i, 2000)
                    featmap_i = featmap_i[idxs]
                    featmap_yx_i = featmap_yx_i[idxs]

                #torch.save(featmap_i.clone().detach().cpu(), 'featmap.pt')
                #torch.save(pos[i_b].clone().detach().cpu(), 'pos_coords.pt')

                # given positives, compute pixel distances and feature distances
                pixel_dist = pdist(pos_coords[i_b], featmap_yx_i, 'L2')
                feat_dist = pdist(pos[i_b], featmap_i, 'inv_norm_cosine')

                # exclude near point by making their feature distance high
                feat_dist = feat_dist + 1e6 * F.relu(self.neg_kernel - pixel_dist)
                # select points nearest in the feature space
                negs_idxs = torch.argmin(feat_dist,dim=1)

                # select y,x coordinated ans obtain feature points
                neg_y, neg_x = featmap_yx_i[negs_idxs,0], featmap_yx_i[negs_idxs,1]
                neg_y, neg_x = neg_y.to(torch.int64), neg_x.to(torch.int64)
                neg_feats[i_b] = featmap[i_b][:,neg_y,neg_x].transpose(1,0)
                neg_coords[i_b,:,0] = neg_y
                neg_coords[i_b,:,1] = neg_x
                
        return neg_feats, neg_coords

    def sample_negatives(self, pos : Tensor, featmap : Tensor, pos_coords : Tensor, valid : Tensor) -> Tuple[Tensor,Tensor]:
        '''
        Sample a set of negatives given a set of positives, a feature map and a mask on the feature map.
        Positives are not needed now, but can be useful later to implement an hardest contrastive approach
        '''

        BS, N, D = pos.shape
        _,_,H,W = featmap.shape
        neg_feats = torch.zeros((BS,N,D)).to(self.device)
        neg_coords = torch.zeros((BS,N,2)).to(self.device)

        # make coordinate map from feature map
        xs = torch.linspace(0, W-1, steps=W)
        ys = torch.linspace(0, H-1, steps=H)
        xmap, ymap = torch.meshgrid(xs,ys, indexing='xy')
        xmap = xmap.flatten().to(self.device)
        ymap = ymap.flatten().to(self.device)
        # basically a list of yx coordinates in the pixel space
        featmap_yx = torch.stack([ymap,xmap],dim=-1)
        if self.device == 'cuda':
            featmap_yx = featmap_yx.half()


        for i_b in range(BS):
            
            if valid[i_b] == 1:
                
                # reshape the feature map: [512,H,W] -> [H*W,512] as a list of features
                featmap_i = featmap[i_b].reshape(D,(H*W)).transpose(1,0)
                # [H*W, 2]
                featmap_yx_i = featmap_yx.clone()
                
                negs_idxs = torch.randint(0, featmap_i.shape[0], (pos.shape[1],))

                # select y,x coordinated ans obtain feature points
                neg_y, neg_x = featmap_yx_i[negs_idxs,0], featmap_yx_i[negs_idxs,1]
                neg_y, neg_x = neg_y.to(torch.int64), neg_x.to(torch.int64)
                neg_feats[i_b] = featmap[i_b][:,neg_y,neg_x].transpose(1,0)
                neg_coords[i_b,:,0] = neg_y
                neg_coords[i_b,:,1] = neg_x
                
        return neg_feats, neg_coords

