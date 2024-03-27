import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from matplotlib import collections as mc
from matplotlib import cm
from numpy import ndarray
from utils.metrics import np_mask_iou
from utils.pcd import project_points, np_transform_pcd
from utils.metrics import compute_RT_distances, get_entropy
import torch.nn.functional as F
from torch import Tensor
import torch
import cv2
from typing import Tuple, List, Optional, Union



def get_item_rgb(item: dict) -> ndarray:

    return (item['rgb'] * 255).numpy().transpose(1,2,0).astype(np.uint8)

def save_item_rgb(item: dict, path: str):

    rgb = get_item_rgb(item)
    cv2.imwrite(path, rgb)

def save_item_depth(item: dict, path:str):

    depth = item['cropped_depth'].numpy()
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    depth_colored = cm.viridis(depth)
    cv2.imwrite(path, (depth_colored[:,:,:3] * 255).astype(np.uint8))

def pred_mask(rgb1: ndarray, rgb2: ndarray, gt1: ndarray, gt2: ndarray, pred1: ndarray, pred2: ndarray, logits1: ndarray, logits2: ndarray, out_path: str):

    '''
    Visualizes ground truth and predicted masks, along with IoU
    '''
    assert rgb1.shape[:2] == gt1.shape, " Problem with first image shapes"
    assert rgb2.shape[:2] == gt2.shape, " Problem with first image shapes"

    if pred1.shape != gt1.shape:
        pred1 = torch.tensor(pred1).unsqueeze(0).unsqueeze(1)
        pred1 = F.interpolate(pred1.to(torch.float), gt1.shape, mode='nearest').squeeze().numpy()
        logits1 = torch.tensor(logits1).unsqueeze(0).unsqueeze(1)
        logits1 = F.interpolate(logits1.to(torch.float), gt1.shape, mode='bilinear').squeeze().numpy()
    
    if pred2.shape != gt2.shape:
        pred2 = torch.tensor(pred2).unsqueeze(0).unsqueeze(1)
        pred2 = F.interpolate(pred2.to(torch.float), gt2.shape, mode='nearest').squeeze().numpy()
        logits2 = torch.tensor(logits2).unsqueeze(0).unsqueeze(1)
        logits2 = F.interpolate(logits2.to(torch.float), gt1.shape, mode='bilinear').squeeze().numpy()

    colored_logits1 = (cm.bwr(logits1)[:,:,:3] * 255).astype(np.uint8)
    colored_logits2 = (cm.bwr(logits2)[:,:,:3] * 255).astype(np.uint8)

    iou1 = np_mask_iou(pred1, gt1)
    iou2 = np_mask_iou(pred2, gt2)

    pred1, gt1 = (pred1 * 255).astype(np.uint8), gt1 * 255
    pred2, gt2 = (pred2 * 255).astype(np.uint8), gt2 * 255

    fig, axs = plt.subplots(nrows=4, ncols=2)

    axs[0,0].imshow(rgb1, alpha=1.0, interpolation='none')
    axs[0,0].set_axis_off()
    axs[0,0].set_title('Query', fontsize=5)
    
    axs[0,1].imshow(rgb2, alpha=1.0, interpolation='none')
    axs[0,1].set_axis_off()
    axs[0,1].set_title('Anchor', fontsize=5)

    axs[1,0].imshow(gt1, alpha=1.0, interpolation='none')
    axs[1,0].set_axis_off()
    axs[1,0].set_title('GT Mask', fontsize=5)
    
    axs[1,1].imshow(gt2, alpha=1.0, interpolation='none')
    axs[1,1].set_axis_off()
    axs[1,1].set_title('GT Mask', fontsize=5)

    axs[2,0].imshow(pred1, alpha=1.0, interpolation='none')
    axs[2,0].set_axis_off()
    axs[2,0].set_title(f'Pred: {iou1:.2f} IoU', fontsize=5)
    
    axs[2,1].imshow(pred2, alpha=1.0, interpolation='none')
    axs[2,1].set_axis_off()
    axs[2,1].set_title(f'Pred: {iou2:.2f} IoU', fontsize=5)

    axs[3,0].imshow(colored_logits1, alpha=1.0, interpolation='none')
    axs[3,0].set_axis_off()
    axs[3,0].set_title(f'Logits', fontsize=5)
    
    axs[3,1].imshow(colored_logits2, alpha=1.0, interpolation='none')
    axs[3,1].set_axis_off()
    axs[3,1].set_title(f'Logits', fontsize=5)

    plt.gcf().set_size_inches(2, 4)
    fig.savefig(
        out_path,
        bbox_inches='tight', transparent=False, dpi=300)
    plt.close(fig)


def attention_map(rgb1: ndarray, rgb2: ndarray, att1: Tensor, att2: Tensor, out_path: str):
    '''
    Visualizes index (mapped to color) of visual descriptor most considered
    '''

    max1 = torch.argmax(att1, dim=0)
    max2 = torch.argmax(att2, dim=0)
    conf1 = 1 - get_entropy(att1, dim=0,norm=True)
    conf2 = 1 - get_entropy(att2, dim=0,norm=True)

    conf1 = (conf1 - conf1.min()) / (conf1.max() - conf1.min()) 
    conf2 = (conf2 - conf2.min()) / (conf2.max() - conf2.min()) 

    fig, axs = plt.subplots(nrows=3, ncols=2)

    axs[0,0].imshow(rgb1, alpha=1.0, interpolation='none')
    axs[0,0].set_axis_off()
    axs[0,0].set_title('Query', fontsize=5)
    
    axs[0,1].imshow(rgb2, alpha=1.0, interpolation='none')
    axs[0,1].set_axis_off()
    axs[0,1].set_title('Anchor', fontsize=5)

    max1 = (max1 - max1.min()) / (max1.max() - max1.min()) 
    max2 = (max2 - max2.min()) / (max2.max() - max2.min()) 

    color1 = (cm.gist_rainbow(max1.numpy())[:,:,:3] * 255).astype(np.uint8)
    color2 = (cm.gist_rainbow(max2.numpy())[:,:,:3] * 255).astype(np.uint8)
    ent_color1 = (cm.viridis(conf1.numpy())[:,:,:3] * 255).astype(np.uint8)
    ent_color2 = (cm.viridis(conf2.numpy())[:,:,:3] * 255).astype(np.uint8)

    axs[1,0].imshow(color1, alpha=1.0, interpolation='none')
    axs[1,0].set_axis_off()
    axs[1,0].set_title('Most attended prompt', fontsize=5)
    axs[1,1].imshow(color2, alpha=1.0, interpolation='none')
    axs[1,1].set_title('Most attended prompt', fontsize=5)
    axs[1,1].set_axis_off()
    
    axs[2,0].imshow(ent_color1, alpha=1.0, interpolation='none')
    axs[2,0].set_title('Prompt confidence', fontsize=5)
    axs[2,0].set_axis_off()
    axs[2,1].imshow(ent_color2, alpha=1.0, interpolation='none')
    axs[2,1].set_title('Prompt confidence', fontsize=5)
    axs[2,1].set_axis_off()

    fig.savefig(
        out_path,
        bbox_inches='tight', transparent=False, dpi=300)
    plt.close(fig)

def feature_distance(rgb1 : ndarray, rgb2 : ndarray, featmap1 : Tensor, featmap2 : Tensor, corrs : ndarray, out_path: str):

    '''
    rgb1, rgb2 : [3,H,W]
    featmap1, featmap2 : [D,H,W]
    corrs: [N,4]
    '''

    fig, axs = plt.subplots(nrows=4, ncols=2)

    axs[0,0].imshow(rgb1, alpha=1.0, interpolation='none')
    axs[0,0].set_axis_off()
    axs[0,0].set_title('Query', fontsize=5)
    
    axs[0,1].imshow(rgb2, alpha=1.0, interpolation='none')
    axs[0,1].set_axis_off()
    axs[0,1].set_title('Anchor', fontsize=5)

    for i in range(1,4):
        idx = np.random.randint(0, corrs.shape[0])
        corr1, corr2 = corrs[idx,:2], corrs[idx,2:]
        
        feat1 = featmap1[:,corr1[0], corr1[1]]

        dist1 = torch.pow((feat1.unsqueeze(1).unsqueeze(2).cpu() - featmap1.cpu()),2).sum(0).sqrt()
        dist2 = torch.pow((feat1.unsqueeze(1).unsqueeze(2).cpu() - featmap2.cpu()),2).sum(0).sqrt()
        dist1 = (dist1 - dist1.min()) / (dist1.max() - dist1.min()) 
        dist2 = (dist2 - dist2.min()) / (dist2.max() - dist2.min()) 

        color1 = (cm.viridis(dist1.numpy())[:,:,:3] * 255).astype(np.uint8)
        color2 = (cm.viridis(dist2.numpy())[:,:,:3] * 255).astype(np.uint8)
    
        axs[i,0].imshow(color1, alpha=1.0, interpolation='none')
        axs[i,0].set_axis_off()
        
        
        axs[i,1].imshow(color2, alpha=1.0, interpolation='none')
        axs[i,1].set_axis_off()

        axs[i,0].scatter(
            x = corr1[1],
            y = corr1[0],
            s = 1.0,
            alpha= 0.5,
            c = 'red'
        )

        axs[i,1].scatter(
            x = corr2[1],
            y = corr2[0],
            s = 1.0,
            alpha= 0.5,
            c = 'red'
        )
    plt.gcf().set_size_inches(2, 4)
    fig.savefig(
        out_path,
        bbox_inches='tight', transparent=False, dpi=300)
    plt.close(fig)

def corr_neg(rgb1 : ndarray, rgb2 : ndarray, corr_set: ndarray, neg_set1: ndarray, neg_set2: ndarray, out_path : str, max_corrs : int = 20):
    '''
    Given two images and two sets of correspondences between them, visualizes them and negative correspondences as well
    '''

    corr_set_ = corr_set.copy()
    neg_set1_ = neg_set1.copy()
    neg_set2_ = neg_set2.copy()

    H1,W1 = rgb1.shape[:2]

    viz_img = np.concatenate((rgb1, rgb2),axis=1)
    fig, axs = plt.subplots(nrows=1, ncols=1)

    axs.imshow(viz_img, alpha=1.0, interpolation='none')
    axs.set_axis_off()
    
    pos_set1 = corr_set_[:,:2]
    pos_set2 = corr_set_[:,2:]
    pos_set2[:,1] += W1
    neg_set2_[:,1] += W1

    if corr_set_.shape[0] > max_corrs:
        idxs = np.random.choice(np.arange(0, corr_set_.shape[0]), max_corrs, replace=False)
    else:
        idxs = np.arange(0, corr_set_.shape[0])
    
    pos_lines = [[(c1[1],c1[0]),(c2[1],c2[0])] for c1, c2 in zip(pos_set1[idxs], pos_set2[idxs])]
    neg_lines1 = [[(p[1], p[0]),(n[1],n[0])] for p,n in zip(pos_set1[idxs], neg_set1_[idxs])]
    neg_lines2 = [[(p[1], p[0]),(n[1],n[0])] for p,n in zip(pos_set2[idxs], neg_set2_[idxs])]
    pos_colors = ['green' for i in range(len(pos_lines))]
    neg_colors = ['red' for i in range(len(pos_lines))]
    
    pos_lc = mc.LineCollection(pos_lines, colors=pos_colors, linewidths=0.5)
    axs.add_collection(pos_lc)
    
    neg1_lc = mc.LineCollection(neg_lines1, colors=neg_colors, linewidths=0.5)
    axs.add_collection(neg1_lc)
    
    neg2_lc = mc.LineCollection(neg_lines2, colors=neg_colors, linewidths=0.5)
    axs.add_collection(neg2_lc)
    
    for pos_line, neg_line1, neg_line2 in zip(pos_lines, neg_lines1, neg_lines2):
        
        axs.scatter(
            x=[pos_line[0][0]],
            y=[pos_line[0][1]],
            s=3.0,
            alpha=0.8,
            c='green'
        )
        axs.scatter(
            x=[pos_line[1][0]],
            y=[pos_line[1][1]],
            s=3.0,
            alpha=0.8,
            c='green'
        )
        axs.scatter(
            x=[neg_line1[1][0]],
            y=[neg_line1[1][1]],
            s=3.0,
            alpha=0.8,
            c='red'
        )
        axs.scatter(
            x=[neg_line2[1][0]],
            y=[neg_line2[1][1]],
            s=3.0,
            alpha=0.8,
            c='red'
        )
    fig.savefig(
        out_path,
        bbox_inches='tight', transparent=False, dpi=300)
    plt.close(fig)

def pred_pose(rgb_a : ndarray, rgb_q : ndarray, gt_pose : ndarray, pred_pose : ndarray, K: ndarray, obj_model : ndarray, out_path : str):
    fig, axs = plt.subplots(nrows=2, ncols=2)

    axs[0,0].imshow(rgb_q, alpha=1.0, interpolation='none')
    axs[0,0].set_axis_off()
    axs[0,0].set_title('Query', fontsize=5)
    
    axs[0,1].imshow(rgb_a, alpha=1.0, interpolation='none')
    axs[0,1].set_axis_off()
    axs[0,1].set_title('Anchor', fontsize=5)
    
    axs[1,0].imshow(rgb_q, alpha=1.0, interpolation='none')
    axs[1,0].set_axis_off()
    axs[1,0].set_title('Ground truth', fontsize=5)
    
    axs[1,1].imshow(rgb_q, alpha=1.0, interpolation='none')
    axs[1,1].set_axis_off()
    

    r,t = gt_pose[:3,:3], gt_pose[:3,3]
    #print("Gt rotation epoch ", self.current_epoch, r.flatten())
    gt_pcd = np_transform_pcd(obj_model, r, t)
    pts = project_points(gt_pcd, K)
    correct_x1, correct_x2 = pts[:,0] < 640, pts[:,0] >= 0
    correct_y1, correct_y2 = pts[:,1] < 480, pts[:,1] >= 0
    valid = np.logical_and(np.logical_and(correct_y2, correct_y1), np.logical_and(correct_x1, correct_x2))
    pts = pts[valid]
    
    axs[1,0].scatter(
        x = pts[:,0],
        y = pts[:,1],
        s = 0.1,
        alpha= 0.5,
        c = 'red'
    )

    r,t = pred_pose[:3,:3], pred_pose[:3,3] 
    
    r_err, t_err = compute_RT_distances(pred_pose, gt_pose)
    axs[1,1].set_title(f'Rot Err: {r_err[0]:.2f}° T Err: {t_err[0]:.2f}', fontsize=5)
    pred_pcd = np_transform_pcd(obj_model, r, t)
    pts = project_points(pred_pcd, K)
    correct_x1, correct_x2 = pts[:,0] < 640, pts[:,0] >= 0
    correct_y1, correct_y2 = pts[:,1] < 480, pts[:,1] >= 0
    valid = np.logical_and(np.logical_and(correct_y2, correct_y1), np.logical_and(correct_x1, correct_x2))
    pts = pts[valid]

    axs[1,1].scatter(
        x = pts[:,0],
        y = pts[:,1],
        s = 0.1,
        alpha= 0.5,
        c = 'blue'
    )
    fig.savefig(
        out_path,
        bbox_inches='tight', transparent=False, dpi=300)
    plt.close(fig)

def corr_set(rgb_a : Union[ndarray, Tensor], rgb_q : Union[ndarray, Tensor], gt_corrs : ndarray, pred_corrs : ndarray, out_path : str):

    if isinstance(rgb_a,Tensor):
        rgb_a = (rgb_a.clone().cpu().numpy().transpose(1,2,0))
        rgb_q = (rgb_q.clone().cpu().numpy().transpose(1,2,0))

    pred_corrs_ = pred_corrs.copy()
    gt_corrs_ = gt_corrs.copy()
    viz_img = np.concatenate((rgb_a, rgb_q),axis=1)
    fig, axs = plt.subplots(nrows=3, ncols=2)
    
    H1,W1 = rgb_a.shape[:2]
    pred_corrs_[:,3] += W1
    gt_corrs_[:,3] += W1
    COLORS = ['b','g','r','c','m','y','k','w']

    for axs_i in range(3):

        if axs_i == 0:
            axs[axs_i,0].set_title('Predicted')
            axs[axs_i,1].set_title('Ground truth')

        axs[axs_i,0].imshow(viz_img, alpha=1.0, interpolation='none')
        axs[axs_i,0].set_axis_off()
        axs[axs_i,1].imshow(viz_img, alpha=1.0, interpolation='none')
        axs[axs_i,1].set_axis_off()

        if gt_corrs_.shape[0] > 8:
            gt_idxs = np.random.choice(np.arange(0, gt_corrs_.shape[0]), 8, replace=False)
        else:
            gt_idxs = np.arange(gt_corrs_.shape[0])
        
        if pred_corrs_.shape[0] > 8:
            pred_idxs = np.random.choice(np.arange(0, pred_corrs_.shape[0]), 8, replace=False)
        else:
            pred_idxs = np.arange(pred_corrs_.shape[0])
        
        gt_lines = [[(c[1],c[0]),(c[3],c[2])] for c in gt_corrs_[gt_idxs]]
        #gt_lines = [[(c[0],c[1]),(c[2],c[3])] for c in gt_corrs[gt_idxs]]
        gt_colors = [COLORS[i%8] for i in range(len(gt_lines))]
        pred_lines = [[(c[1],c[0]),(c[3],c[2])] for c in pred_corrs_[pred_idxs]]
        #pred_lines = [[(c[0],c[1]),(c[2],c[3])] for c in pred_corrs[pred_idxs]]
        pred_colors = [COLORS[i%8] for i in range(len(pred_lines))]
        
        lc = mc.LineCollection(gt_lines, colors=gt_colors, linewidths=0.3)
        axs[axs_i,1].add_collection(lc)
        lc = mc.LineCollection(pred_lines, colors=pred_colors, linewidths=0.3)
        axs[axs_i,0].add_collection(lc)

        for (gt_line, pred_line, gt_color, pred_color) in zip(gt_lines, pred_lines, gt_colors, pred_colors):
            
            axs[axs_i,0].scatter(
                x=[pred_line[0][0]],
                y=[pred_line[0][1]],
                s=1.0,
                alpha=0.8,
                c=pred_color
            )
            axs[axs_i,0].scatter(
                x=[pred_line[1][0]],
                y=[pred_line[1][1]],
                s=1.0,
                alpha=0.8,
                c=pred_color
            )

            axs[axs_i,1].scatter(
                x=[gt_line[0][0]],
                y=[gt_line[0][1]],
                s=1.0,
                alpha=0.8,
                c=gt_color
            )
            axs[axs_i,1].scatter(
                x=[gt_line[1][0]],
                y=[gt_line[1][1]],
                s=1.0,
                alpha=0.8,
                c=gt_color
            )

    plt.gcf().set_size_inches(8, 6.5)
    fig.savefig(
        out_path,
        bbox_inches='tight', transparent=False, dpi=300)
    plt.close(fig)

def featmap_to_rgb(featmap: ndarray) -> Image:
    '''
    Maps a 2D feature map (D,W,H) in an RGB image using PCA
    '''

    D,W,H = featmap.shape
    lin_feats = np.reshape(featmap,(D,W*H)).transpose(1,0)

    pca = PCA(n_components=3)
    y = pca.fit_transform(lin_feats)
    std, mean = y.std(0), y.mean(0)
    y = 255 * (y - mean) / std
    y = np.reshape(y, (W,H,3))

    img = Image.fromarray(y.astype(np.uint8))
    
    return img

def dual_featmap_to_rgbs(featmap_a: ndarray, featmap_b: ndarray) -> Tuple[object,object]:
    '''
    Maps a 2D feature map (D,W,H) in an RGB image using PCA
    '''

    D1,W1,H1 = featmap_a.shape
    D2,W2,H2 = featmap_b.shape

    assert D1 == D2

    lin_feats_a = np.reshape(featmap_a.astype(np.float16),(D1,W1*H1)).transpose(1,0)
    lin_feats_b = np.reshape(featmap_b.astype(np.float16),(D2,W2*H2)).transpose(1,0)
    lin_feats = np.concatenate((lin_feats_a, lin_feats_b), axis=0)
    pca = PCA(n_components=3, copy=False)
    y = pca.fit_transform(lin_feats)
    std, mean = y.std(0), y.mean(0)
    y = 255 * (y - mean) / std
    y_a, y_b = y[:W1*H1], y[W1*H1:]
    y_a = np.reshape(y_a, (W1,H1,3))
    y_b = np.reshape(y_b, (W2,H2,3))

    img_a = Image.fromarray(y_a.astype(np.uint8))
    img_b = Image.fromarray(y_b.astype(np.uint8))
    
    return img_a,img_b

def featmaps_to_rgbs(featmaps: List[ndarray]) -> List[ndarray]:
    '''
    Maps a list of 2D feature maps (D,W,H) in a list of RGB images using PCA
    '''

    sizes = list()
    feat_sizes = list()
    lens = list()
    lin_feats = list()
    images = list()

    for featmap in featmaps:
        D,H,W = featmap.shape
        sizes.append((H,W))
        lens.append(H*W)
        feat_sizes.append(D)

        lin_feats.append(np.reshape(featmap.astype(np.float16),(D,W*H)).transpose(1,0))
    
    lin_feats = np.concatenate(lin_feats, axis=0)

    pca = PCA(n_components=3, copy=False)
    y = pca.fit_transform(lin_feats)
    std, mean = y.std(0), y.mean(0)
    y = 255 * (y - mean) / std
    
    acc_len = 0
    for feat_len, img_size in zip(lens, sizes):
        
        h,w = img_size
        y_i = y[acc_len:acc_len+feat_len]
        y_i = np.reshape(y_i, (w,h,3))

        img_i = Image.fromarray(y_i.astype(np.uint8))
        images.append(img_i)
        acc_len += feat_len

    return images


def viz_item(item: dict, out_path: str, resized: bool):

    rgb = item['rgb']
    mask = item['cropped_mask'] if resized else item['mask']
    depth = item['cropped_depth'] if resized else item['depth']

    if isinstance(rgb, torch.Tensor):
        rgb = (rgb.clone().numpy().transpose(1,2,0) * 255)
        mask = mask.clone().numpy() * 255
        depth = depth.clone().numpy().astype(np.float32)
        dmax, dmin = depth.max(), depth.min()
        depth = (depth - dmin) / (dmax-dmin)

    depth_color = (cm.viridis(depth)[:,:,:3] * 255).astype(np.uint8)
    rgb = rgb.astype(np.uint8)
    mask = mask.astype(np.uint8)

    fig, axs = plt.subplots(nrows=3, ncols=1)

    axs[0].imshow(rgb, alpha=1.0, interpolation='none')
    axs[0].set_axis_off()
    axs[0].set_title('RGB', fontsize=5)
        
    axs[1].imshow(mask, alpha=1.0, interpolation='none')
    axs[1].set_axis_off()
    axs[1].set_title('Mask', fontsize=5)

    axs[2].imshow(depth_color, alpha=1.0, interpolation='none')
    axs[2].set_axis_off()
    axs[2].set_title('Depth', fontsize=5)

    fig.savefig(
        out_path,
        bbox_inches='tight', transparent=False, dpi=300)
    plt.close(fig)

