import os
import numpy as np
from os.path import join
import json
from PIL import Image
import sys

ROOT = sys.argv[1]
SPLIT = 'test'

from tqdm import tqdm

for scene_folder in tqdm(os.listdir(join(ROOT,'split',SPLIT))):

    with open(join(ROOT,'split',SPLIT,scene_folder,'scene_gt.json')) as f:

        data = json.load(f)

    for img_id, img_data in data.items():
        full_mask = np.zeros((480, 640))

        for i in range(len(img_data)):
            mask_i = Image.open(join(ROOT,'split',SPLIT,scene_folder,'mask_visib',f'{int(img_id):06d}_{int(i):06d}.png')).convert('L')
            mask_i = np.asarray(mask_i).copy()
            full_mask[mask_i == 255] = i+1

        full_mask[full_mask == 0] = 255
        out_path = join(ROOT,'split',SPLIT,scene_folder,'mask_visib',f'{int(img_id):06d}.png')
        Image.fromarray(full_mask.astype(np.uint8)).save(out_path)
        

            