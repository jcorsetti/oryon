import os
import sys
from os.path import join
sys.path.append(os.getcwd())
from utils.data import nocs
import numpy as np
import os
from os.path import join
from utils.pcd import pdist
import torch
import json
import sys

ROOT = sys.argv[1] # your root to the real275 dataset goes here
SPLIT = 'real_test'
print("Making BOP-format symm annotations from objects in ", join(ROOT,'obj_models',SPLIT))

SYM = [
      {
        "axis": [
          0,
          0,
          1
        ],
        "offset": [
          0,
          0,
          0
        ]
      }
    ]

models_info = dict()
for file in os.listdir(join(ROOT,'obj_models',SPLIT)):

    objname, ext = os.path.splitext(file)
    if ext == '.obj':
        obj_r = nocs.get_obj_rendering(ROOT, objname)
        xyz = obj_r['pts']
        mins, maxs = np.min(xyz,axis=0), np.max(xyz,axis=0)
        
        p1 = xyz[xyz[:,0] == mins[0]]
        p2 = xyz[xyz[:,1] == mins[1]]
        p3 = xyz[xyz[:,2] == mins[2]]
        p4 = xyz[xyz[:,0] == maxs[0]]
        p5 = xyz[xyz[:,1] == maxs[1]]
        p6 = xyz[xyz[:,2] == maxs[2]]

        ps = torch.tensor(np.concatenate([p1,p2,p3,p4,p5,p6],axis=0))
        obj_dist = pdist(ps,ps)
        diameter = torch.max(obj_dist)
        cur_info = {
            'diameter' : diameter.item(),
            'min_x' : mins[0],
            'min_y' : mins[1],
            'min_z' : mins[2],
            'max_x' : maxs[0],
            'max_y' : maxs[1],
            'max_z' : maxs[2]
        }

        if 'bottle' in objname or 'bowl' in objname or 'can' in objname:
            cur_info['symmetries_continuous'] = SYM
        models_info[objname] = cur_info

with open(join(ROOT,'obj_models',SPLIT,'models_info.json'),'w') as f:
    json.dump(models_info,f)






