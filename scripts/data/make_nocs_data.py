import os
import sys
from os.path import join
sys.path.append(os.getcwd())
from utils.misc import sorted_alphanumeric
from utils.data.nocs import make_detections

ROOT = sys.argv[1]
PATH = join(ROOT,'split','real_test')

f = open(join(PATH, 'instance_list.txt'),'w')

print("Making annotations from ", PATH)
for scene in sorted_alphanumeric(os.listdir(PATH)):

    if os.path.isdir(join(PATH,scene)):
        scene_id = int(scene.split('_')[-1])
        scene_set = set()

        for file in sorted_alphanumeric(os.listdir(join(PATH, scene))):
            
            ext = os.path.splitext(file)[-1]

            if ext == '.png':

                file_id = file.split('_')[0]
                if file_id not in scene_set:
                    f.write(f'{scene_id} {file_id}\n')
                    scene_set.add(file_id)
f.close()
f = open(join(PATH, 'instance_list.txt'),'r')

# make detections
lines = f.readlines()
for line in lines:
    scene_id, img_id = line.split(' ')
    make_detections(PATH,int(scene_id),int(img_id))

f.close()












