import os
import sys
from os.path import join
sys.path.append(os.getcwd())
import numpy as np
import open3d as o3d
import os
import open3d
from tqdm import tqdm
import sys

PATH = os.path.join(sys.argv[1], 'obj_models/real_test')
print("Making BOP-format models from objects in ", PATH)
for file in tqdm(os.listdir(PATH)):
    # read each object file
    if file.endswith('_vertices.txt'):
        # get pcd
        basename = file[:-13]

        pts = list()
        with open(os.path.join(PATH,file)) as f:
            lines = [line.split(' ') for line in f.readlines()]
            for line in lines:
                pts.append([float(line[0]), float(line[1]), float(line[2])])


        pts = np.asarray(pts) * 1000
        pcd = open3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))

        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1000, max_nn=50))
        norms = np.asarray(pcd.normals)
        with open(os.path.join(PATH, basename + "_normals.txt"),'w') as f:
            for i in range(norms.shape[0]):
                f.write(f'{norms[i,0]} {norms[i,1]} {norms[i,2]}\n')

