import torch
import numpy as np
import transforms3d.quaternions as t3d

def np_angles2mat(angles : np.ndarray) -> np.ndarray:
    '''
    Given angles in radiants, return matrix
    '''
    cosx = np.cos(angles[0])
    cosy = np.cos(angles[1])
    cosz = np.cos(angles[2])
    sinx = np.sin(angles[0])
    siny = np.sin(angles[1])
    sinz = np.sin(angles[2])
    Rx = np.array([[1, 0, 0], [0, cosx, -sinx], [0, sinx, cosx]])
    Ry = np.array([[cosy, 0, siny], [0, 1, 0], [-siny, 0, cosy]])
    Rz = np.array([[cosz, -sinz, 0], [sinz, cosz, 0], [0, 0, 1]])
    R_ab = Rx @ Ry @ Rz

    return R_ab

def np_mat2quat(pose : np.ndarray) -> np.ndarray:
    
    '''
    Translates a single rotation pose to a quaternion
    '''

    assert pose.shape[0] == 3 or pose.shape[0] == 4
    assert pose.shape[1] == 3 or pose.shape[1] == 4

    # get rotation and translate it
    rotate = pose[:3, :3]
    new_pose = t3d.mat2quat(rotate)

    # must also include translation, otherwise only rotation
    if pose.shape[1] == 4:
        translate = pose[:3, 3]
        new_pose = np.concatenate([new_pose, translate], axis=0)
    
    return new_pose  # (7, )


def np_quat2mat(pose : np.ndarray) -> np.ndarray:
    
    '''
    Translates a batch of quaternion poses in a batch of rotation poses
    '''

    single_elem = False
    if len(pose.shape) == 1: # no batch
        pose = np.expand_dims(pose, axis=0)
        single_elem = True

    assert pose.shape[1] == 7 or pose.shape[1] == 4

    # Separate each quaternion value.
    q0, q1, q2, q3 = pose[:, 0], pose[:, 1], pose[:, 2], pose[:, 3]
    # Convert quaternion to rotation matrix.
    # Ref: 	http://www-evasion.inrialpes.fr/people/Franck.Hetroy/Teaching/ProjetsImage/2007/Bib/besl_mckay-pami1992.pdf
    # A method for Registration of 3D shapes paper by Paul J. Besl and Neil D McKay.
    R11 = q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3
    R12 = 2 * (q1 * q2 - q0 * q3)
    R13 = 2 * (q1 * q3 + q0 * q2)
    R21 = 2 * (q1 * q2 + q0 * q3)
    R22 = q0 * q0 + q2 * q2 - q1 * q1 - q3 * q3
    R23 = 2 * (q2 * q3 - q0 * q1)
    R31 = 2 * (q1 * q3 - q0 * q2)
    R32 = 2 * (q2 * q3 + q0 * q1)
    R33 = q0 * q0 + q3 * q3 - q1 * q1 - q2 * q2
    
    R = np.stack((np.stack((R11, R12, R13), axis=0), np.stack((R21, R22, R23), axis=0), np.stack((R31, R32, R33), axis=0)), axis=0)
    new_pose = R.transpose((2, 0, 1))  # (B, 3, 3)
    
    if pose.shape[1] == 7:
        translation = pose[:, 4:][:, :, None]  # (B, 3, 1)
        new_pose = np.concatenate((new_pose, translation), axis=2)
    
    # remove single batch dimension
    if single_elem:
        new_pose = new_pose[0,:,:]

    return new_pose  # (B, 3, 4)