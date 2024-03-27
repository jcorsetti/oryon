import numpy as np
import csv
from typing import Tuple, Dict

def get_dict_stats(dict : Dict) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Get mean and std of dictionary with numerical values
    '''

    values = np.asarray(list(dict.values()))
    return np.mean(values), np.std(values)

def perf_from_csv(file : str) -> Tuple[Dict, Dict]:

    '''
    Given a .csv file in BOP prediction format, returns a dictionary with:
    - Instance id (part id + image id + object id)
    - Rotation matrix
    - Translation vector
    And another with object occurrencies
    '''

    obj_occs = {}
    poses = {}

    with open(file) as f:

        reader = list(csv.reader(f, delimiter=','))

        for i_r, row in enumerate(reader):

            if i_r == 0:  # Ignore first row of csv
               continue

            part_id, img_id, obj_id = int(row[0]),int(row[1]),int(row[2])

            # add new class in dict if not present
            if obj_id not in obj_occs.keys():
                obj_occs[obj_id] = 0

            obj_occs[obj_id] += 1
            r = np.resize(np.asarray(row[4].split(), dtype=np.float64), (3,3))
            t = np.asarray(row[5].split(), dtype=np.float64)

            instance_id = '{:06d}_{:06d}_{:02d}'.format(part_id, img_id, obj_id)
            poses[instance_id] = {
                'r': r,
                't': t
            }

    return poses, obj_occs