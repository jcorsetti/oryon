#!/usr/bin/env python3
import os
import time
from numpy.lib.function_base import append
import torch
import numpy as np
import pickle as pkl
import concurrent.futures
from cv2 import imshow, waitKey
from sklearn.neighbors import NearestNeighbors
import torch.nn.functional as F

# config = Config(ds_name='ycb')
# bs_utils = Basic_Utils(config)
# cls_lst = config.ycb_cls_lst
# try:
#     config_lm = Config(ds_name="linemod")
#     bs_utils_lm = Basic_Utils(config_lm)
# except Exception as ex:
#     print(ex)

def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()

def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
        A: Nxm numpy array of corresponding points, usually points on mdl
        B: Nxm numpy array of corresponding points, usually points on camera axis
    Returns:
    T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
    R: mxm rotation matrix
    t: mx1 translation vector
    '''

    assert A.shape == B.shape
    # get number of dimensions
    m = A.shape[1]
    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    # rotation matirx
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[m-1, :] *= -1
        R = np.dot(Vt.T, U.T)
    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)
    T = np.zeros((3, 4))
    T[:, :3] = R
    T[:, 3] = t
    return T

def best_fit_transform_with_RANSAC(A, B, max_iter = 20, match_err = 0.015, fix_percent = 0.7):

    best_RT = np.zeros((3,4),dtype=np.float32)
    ptsnum, m = A.shape
    if ptsnum < 4:
        return best_RT

    iter = 0
    best_inlier_nums = 0
    
    #extend_A = np.ones((ptsnum,4),dtype=np.float32)
    #extend_A[:,:3] = A
    #extend_B = np.ones((ptsnum,4),dtype=np.float32)
    #extend_B[:,:3] = B
    #extend_RT = np.eye(4,dtype=np.float)
    curr_RT = best_fit_transform(A,B)

    while iter < max_iter:
    # get num of points, get number of dimensions
        curr_R = curr_RT[:,:3]
        curr_T = curr_RT[:,3:4].T

        tran_A = np.dot(A,curr_R.T) + curr_T
        err_dis = np.linalg.norm(tran_A - B,axis=1)
        match_idx = (err_dis <= match_err)
        inliers_num = match_idx.sum()
        if inliers_num > best_inlier_nums:
            best_inlier_nums = inliers_num
            best_RT = curr_RT
        
        if best_inlier_nums > fix_percent * ptsnum:
            best_RT = best_fit_transform(A[match_idx],B[match_idx])
            return best_RT
        #np.random.seed()
        selected_idx = np.random.randint(0,ptsnum,4)
        selected_A = A[selected_idx]
        selected_B = B[selected_idx]
        curr_RT = best_fit_transform(selected_A,selected_B)
        #extend_RT[:3,:] = curr_RT
        #trans_A = np.dot(extend_RT,extend_A.T)
        #err_dis = np.linalg.norm(trans_A - extend_B.T,axis=0)
        

        iter+=1
   
    return best_RT

def best_fit_transform_icp(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
        A: Nxm numpy array of corresponding points, usually points on mdl
        B: Nxm numpy array of corresponding points, usually points on camera axis
    Returns:
    T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
    R: mxm rotation matrix
    t: mx1 translation vector
    '''

    assert A.shape == B.shape
    # get number of dimensions
    m = A.shape[1]
    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    # rotation matirx
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[m-1, :] *= -1
        R = np.dot(Vt.T, U.T)
    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t
    return T

def icp(A, B, init_pose=None, max_iterations=20, tolerance=0.001):
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m+1,A.shape[0]))
    dst = np.ones((m+1,B.shape[0]))
    src[:m,:] = np.copy(A.T)
    dst[:m,:] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T)

        # compute the transformation between the current source and nearest destination points
        T = best_fit_transform_icp(src[:m,:].T, dst[:m,indices].T)

        # update the current source
        src = np.dot(T, src)

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    T = best_fit_transform_icp(A, src[:m,:].T)

    return T[:m,:]
