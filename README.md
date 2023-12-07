# Oryon: Open-Vocabulary Object 6D Pose Estimation

This is the repository that contains source code for the [Oryon website](https://jcorsetti.github.io/oryon-website/) and its implementation to be released.

## Roadmap
- Code release (TBD): February-March 2024 
- Added test and train splits: 7 Dec 2023
- Website and arxiv released: 4 Dec 2023

## Datasets

Our data is based on three publicly available datasets:
- [REAL275](https://github.com/hughw19/NOCS_CVPR2019), used for test. We sample from the real test partition.
- [Toyota-Light](https://bop.felk.cvut.cz/datasets/) (TOYL), used for test. We sample from the test split from the BOP challenge. 
- [ShapeNet6D](https://github.com/ethnhe/FS6D-PyTorch) (SN6D), used for training. Note that SN6D itself does not provide textual annotations, but it uses object models from [ShapeNetSem](https://shapenet.org/), which do provide object names and synsets for each object model.

We sample scenes from each dataset to build the training and testing partition (20000 image pairs for SN6D and 2000 for REAL275 and TOYL), and report in the following folder the scene ids and image ids used for each partition.

    data
    └── real275                  
        ├── obj_names.json
        ├── annots.pkl
        └── instance_list.txt  
    └── toyl                     
        ├── obj_names.json
        ├── annots.pkl
        └── instance_list.txt  
    └── shapenet6d                
        ├── annots.pkl
        └── instance_list.txt  
    └── templates.json         

In `data/templates.json` the templates used to augment the prompt are contained. 
They are the same templates used by [CLIP in ImageNet](https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb).
See the following sections for the format of each dataset.

### REAL275

`obj_names.json` contains the map from each object id in REAL275 to the manually produced textual information used to compute the prompt.
For each object id the associated list reports:
1. The object name
2. The object description used in the default setting
3. The misleading object description used in the ablation study.

`instance_list.txt` contains a list of identifiers of image pair in the following format, for each line:

`
split_name, scene_anchor_id image_anchor_id, scene_query_id image_query_id, category_id object_id
`

`split_name` referes to the split from which the image has been obtained (always `real_test` in our case).
The scenes id refer to the index of the scene in REAL275, while the image ids are indexes of the images within a scene.
The `category_id` is not used, as we only consider the `object_id`.

`annots.pkl` contains the ground truth relative pose and the 2D ground truth matches for each image pair in `instance_list.txt`.
Note that the translation in the ground truth pose is in meters.
The list of matches is an array Nx4, with N the numbers of matches.
Each match is in shape Y1,X1,Y2,X1, where Y1,X1 represent the image coordinate on the anchor image and Y2,X2 the image coordinate on the query image.

### Toyota-Light

`obj_names.json` is structured as in REAL275. 

`instance_list.txt` is structured as REAL275 but without the category id, in the following format for each line:

`
split_name, scene_anchor_id image_anchor_id, scene_query_id image_query_id, object_id
`

`annots.pkl` is structured as in REAL275.

### ShapeNet6D

Each line of `instance_list.txt` is structured as follows:

`
image_anchor_id, image_query_id, object_id
`

`annots.pkl` is structured as in REAL275.

Note that each image of ShapeNet6D shows a different random background, so that we consider each image as being part of a different scene.
ShapeNet6D provides a map from their object ids to the object ids of the original ShapeNetSem: we use this map to associated the object name and synonym sets of ShapeNetSem to each object model in ShapeNet6D.

## Website License
The website template is from [Nerfies](https://github.com/nerfies/nerfies.github.io).

<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.
